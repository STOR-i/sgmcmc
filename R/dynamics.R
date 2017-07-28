# Declare the TensorFlow steps needed for one step of SGLD
# @param sgld is an sgld object
declareDynamics.sgld = function( sgld ) {
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
        # Declare form of one step of SGLD
        dynamics[[pname]] = theta$assign_add( 0.5 * epsilon * grad + 
                sqrt( epsilon ) * tf$random_normal( theta$get_shape() ) )
    }
    return( dynamics )
}

# Declare the TensorFlow steps needed for one step of SGHMC, input SGHMC object
# @param is an sghmc object
declareDynamics.sghmc = function( sghmc) {
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
        nu = tf$Variable( sqrt( stepsize ) * tf$random_normal( theta$get_shape() ) )
        # Declare dynamics
        gradU = estLogPostGrads[[pname]]
        dynamics$refresh[[pname]] = nu$assign( sqrt( stepsize ) * tf$random_normal( 
                theta$get_shape() ) )
        dynamics$nu[[pname]] = nu$assign_add( stepsize*gradU - alpha*nu + 
                sqrt( 2 * stepsize * alpha ) * tf$random_normal( theta$get_shape() ) )
        dynamics$theta[[pname]] = theta$assign_add( nu )
    }
    return( dynamics )
}

# Declare the TensorFlow steps needed for one step of SGNHT
# @param sgnht is an sgnht object
declareDynamics.sgnht = function( sgnht ) {
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
        u = tf$Variable( sqrt(stepsize) * tf$random_normal( theta$get_shape() ) )
        alpha = tf$Variable( a, dtype = tf$float32 )
        # Declare dynamics
        gradU = estLogPostGrads[[pname]]
        dynamics$u[[pname]] = u$assign_add( stepsize * gradU - u * alpha +  
                sqrt( 2 * a * stepsize ) * tf$random_normal( u$get_shape() ) )
        dynamics$theta[[pname]] = theta$assign_add( u )
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
    return( dynamics )
}
