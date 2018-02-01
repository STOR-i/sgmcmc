# Define declareDynamics generic, defined separately for each SGMCMC method
# @param sgmcmc a stochastic gradient mcmc object, as defined in the respective modules sgld.r etc.
declareDynamics = function( sgmcmc, seed ) UseMethod("declareDynamics")

# Declare the TensorFlow steps needed for one step of SGLD
# @param sgld is an sgld object
declareDynamics.sgld = function( sgld, seed ) {
    # dynamics is returned, contains list of TensorFlow steps for SGLD
    dynamics = list()
    # Get the correct gradient estimate given the sgld object (i.e. standard sgld or sgldcv) 
    estLogPostGrads = getGradients( sgld )
    # Loop over each parameter in params
    for ( pname in names( sgld$params ) ) {
        # Declare simulation parameters
        theta = sgld$params[[pname]]
        epsilon = sgld$stepsize[[pname]]
        grad = estLogPostGrads[[pname]]
        # Check if gradient is IndexedSlices object, e.g. if tf$gather was used
        isSparse = checkSparse(grad)
        # Predeclare injected noise
        noise = sqrt(0.5) * getNoise(isSparse, grad, epsilon, seed)
        # Declare dynamics, using sparse updates if grad is IndexedSlices object
        if (isSparse) {
            updateCurr = 0.5 * epsilon * grad$values + noise
            dynamics[[pname]] = tf$scatter_add(theta, grad$indices, updateCurr)
        } else {
            dynamics[[pname]] = theta$assign_add(0.5 * epsilon * grad + noise)
        }
    }
    return( dynamics )
}

# Declare the TensorFlow steps needed for one step of SGHMC, input SGHMC object
# @param is an sghmc object
declareDynamics.sghmc = function( sghmc, seed ) {
    dynamics = list( "theta" = list(), "nu" = list(), "refresh" = list(), "grad" = list() )
    # Get the correct gradient estimate given the sgld object (i.e. standard sgld or sgldcv) 
    estLogPostGrads = getGradients( sghmc )
    # Loop over each parameter in params
    for ( pname in names( sghmc$params ) ) {
        dynamics$grad[[pname]] = tf$gradients( sghmc$estLogPost, sghmc$params[[pname]] )[[1]]
        # Declare tuning constants
        stepsize = sghmc$stepsize[[pname]]
        alpha = sghmc$alpha[[pname]]
        # Declare parameters
        theta = sghmc$params[[pname]]
        nu = tf$Variable( sqrt( stepsize ) * tf$random_normal( theta$get_shape(), seed = seed ) )
        # Declare dynamics
        gradU = estLogPostGrads[[pname]]
        # Check if gradient is IndexedSlices object, e.g. if tf$gather was used
        isSparse = checkSparse(gradU)
        # Predeclare injected noise
        noise = getNoise(isSparse, gradU, stepsize, seed)
        # Declare dynamics, using sparse updates if grad is IndexedSlices object
        if (isSparse) {
            dynamics$refresh[[pname]] = tf$scatter_update(nu, gradU$indices, 0.5 * noise)
            nuCurr = tf$gather(nu, gradU$indices)
            updateCurr = stepsize * gradU$values - alpha * nuCurr + sqrt(alpha) * noise
            dynamics$nu[[pname]] = tf$scatter_add(nu, gradU$indices, updateCurr)
            dynamics$theta[[pname]] = tf$scatter_add(theta, gradU$indices, nuCurr)
        } else {
            dynamics$refresh[[pname]] = nu$assign(0.5 * noise)
            updateCurr = stepsize * gradU - alpha * nu + sqrt(alpha) * noise
            dynamics$nu[[pname]] = nu$assign_add(updateCurr)
            dynamics$theta[[pname]] = theta$assign_add(nu)
        }
    }
    return( dynamics )
}

# Declare the TensorFlow steps needed for one step of SGNHT
# @param sgnht is an sgnht object
declareDynamics.sgnht = function( sgnht, seed ) {
    dynamics = list( "theta" = list(), "u" = list(), "alpha" = list() )
    estLogPostGrads = getGradients( sgnht )
    # Loop over each parameter in params
    for ( pname in names(sgnht$params) ) {
        # Get constants for this parameter
        stepsize = sgnht$stepsize[[pname]]
        a = sgnht$a[[pname]]
        rankTheta = sgnht$ranks[[pname]]
        # Declare momentum params
        theta = sgnht$params[[pname]]
        u = tf$Variable( sqrt(stepsize) * tf$random_normal( theta$get_shape(), seed = seed ) )
        alpha = tf$Variable( a, dtype = tf$float32 )
        # Declare dynamics
        gradU = estLogPostGrads[[pname]]
        # Check if gradient is IndexedSlices object, e.g. if tf$gather was used
        isSparse = checkSparse(gradU)
        # Predeclare injected noise
        noise = sqrt(a) * getNoise(isSparse, gradU, stepsize, seed)
        # Declare dynamics, using sparse updates if grad is IndexedSlices object
        if (isSparse) {
            uCurr = tf$gather(u, gradU$indices)
            uUpdate = stepsize * gradU$values - tf$multiply(alpha, uCurr) + noise
            dynamics$u[[pname]] = tf$scatter_add(u, gradU$indices, uUpdate)
            dynamics$theta[[pname]] = tf$scatter_add(theta, gradU$indices, uCurr)
            axes = matrix( rep( 0:( rankTheta - 1 ), each = 2 ), nrow = 2 )
            axes = tf$constant( axes, dtype = tf$int32 )
            aUpdate = tf$tensordot( uCurr, uCurr, axes ) / tf$size( uCurr, out_type = tf$float32 )
            dynamics$alpha[[pname]] = alpha$assign_add(aUpdate - stepsize) 
        } else {
            dynamics$u[[pname]] = u$assign_add(stepsize * gradU - u * alpha + noise)
            dynamics$theta[[pname]] = theta$assign_add(u)
            # Tensordot throws error if rank is 0 so catch this edge case
            # For parameters of higher order than vectors we use tensor contraction
            # to calculate the inner product for the thermostat.
            if ( rankTheta == 0 ) {
                dynamics$alpha[[pname]] = alpha$assign_add( u * u - stepsize )
            } else if( rankTheta >= 1 ) {
                # Declare axes for tensor contraction
                axes = matrix( rep( 0:( rankTheta - 1 ), each = 2 ), nrow = 2 )
                axes = tf$constant( axes, dtype = tf$int32 )
                dynamics$alpha[[pname]] = alpha$assign_add( 
                        tf$tensordot( u, u, axes ) / tf$size( u, out_type = tf$float32 ) - stepsize )
            }
        }
    }
    return( dynamics )
}


# Check if gradient is indexed slices object
checkSparse = function(grad) {
    isSparse = tryCatch({
        temp = grad$indices
        TRUE
    }, error = function (e) { 
        return(FALSE)
    })
    return(isSparse)
}

# Declare injected noise
getNoise = function(gathered, grad, epsilon, seed) {
    if (gathered) {
        noise = sqrt(2 * epsilon) * tf$random_normal(grad$values$get_shape(), seed = seed)
    } else {
        noise = sqrt(2 * epsilon) * tf$random_normal(grad$get_shape(), seed = seed)
    }
    return(noise)
}
