# Methods for implementing Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) using Tensorflow.
# Gradients are automatically calculated. The main function is sghmc, which implements a full
# SGHMC procedure for a given model, including gradient calculation.
#
# References:
#   1. T. Chen, E.B. Fox, and C. Guestrin.  
#           Stochastic gradient Hamiltonian Monte Carlo.  
#           In Proceeding of 31st International Conference on Machine Learning (ICML’14), 2014.

library(tensorflow)
source("setup.r")
source("update.r")
source("storage.r")
source("controlVariates.r")

declareDynamics = function( estLogPost, estLogPostOpt, gradFull, params, paramsOpt, 
        stepsizes, aList, ranks ) {
    # Initialize SGNHT tensorflow by declaring Dynamics
    dynamics = list( "theta" = list(), "u" = list(), "alpha" = list() )
    grads = calcCVGradient( estLogPost, estLogPostOpt, gradFull, params, paramsOpt )
    for ( pname in names(params) ) {
        # Get constants for this parameter
        eta = stepsizes[[pname]]
        a = aList[[pname]]
        rankTheta = ranks[[pname]]
        # Declare momentum params
        theta = params[[pname]]
        u = tf$Variable( sqrt(eta) * tf$random_normal( theta$get_shape() ) )
        alpha = tf$Variable( a, dtype = tf$float32 )
        gradU = grads[[pname]]
        # Declare dynamics
        dynamics$u[[pname]] = u$assign_sub( u * alpha - eta * gradU - 
                sqrt( 2*a*eta ) * tf$random_normal( u$get_shape() ) )
        dynamics$theta[[pname]] = theta$assign_add( u )
        if ( rankTheta == 0 ) {
            dynamics$alpha[[pname]] = alpha$assign_add( u * u - eta )
        } else {
            dynamics$alpha[[pname]] = alpha$assign_add( 
                    tf$tensordot( u, u, rankTheta ) / tf$size( u, out_type = tf$float32 ) - eta )
        }
    }
    return( dynamics )
}

updateSGNHT = function( sess, dynamics, data, placeholders, minibatch_size ) {
    # Perform one step of SGNHT
    feedCurr = data_feed( data, placeholders, minibatch_size )
    for ( step in dynamics$u ) {
        sess$run( step, feed_dict = feedCurr )
    }
    for ( step in dynamics$theta ) {
        sess$run( step, feed_dict = feedCurr )
    }
    for ( step in dynamics$alpha ) {
        sess$run( step, feed_dict = feedCurr )
    }
}

sgnhtCV = function( logLik, logPrior, data, paramsRaw, stepsize, a, n, n_iters = 10^4 ) {
    # 
    # Get key sizes and declare correction term for log posterior estimate
    N = dim( data[[1]] )[1]
    correction = tf$constant( N / n, dtype = tf$float32 )
    # Convert params and data to tensorflow variables and placeholders
    params = setupParams( paramsRaw )
    placeholders = setupPlaceholders( data, minibatch_size )
    # Declare tensorflow variables for initial optimizer
    paramsOpt = setupParams( paramsRaw )
    placeholdersFull = setupFullPlaceholders( data )
    paramStorage = initStorage( paramsRaw, n_iters )
    # Get ranks for each parameter tensor, required for dynamics
    ranks = getRanks( paramsRaw )
    # Declare container for full gradient
    gradFull = setupFullGradients( paramsRaw )
    # Declare estimated log posterior tensor using declared variables and placeholders
    estLogPost = logPrior(params,placeholders) + correction * logLik(params,placeholders)
    # Declare estimated log posterior tensor for optimization
    estLogPostOpt = logPrior(paramsOpt,placeholders) + correction * logLik(paramsOpt,placeholders)
    # Declare full log posterior for calculation at MAP estimate
    fullLogPostOpt = logPrior(paramsOpt,placeholdersFull) + logLik(paramsOpt,placeholdersFull)
    # Declare optimizer ADD STEPSIZE AS A PARAMETER
    optSteps = declareOptimizer( estLogPostOpt, fullLogPostOpt, paramsOpt, params, gradFull, 1e-5 )
    # Declare SGLD dynamics
    dynamics = declareDynamics( estLogPost, estLogPostOpt, gradFull, params, 
            paramsOpt, stepsize, a, ranks )
    # Initalize tensorflowsession
    sess = initSess()
    # Run Langevin dynamics on each parameter for n_iters
    writeLines( "Finding initial MAP estimates" )
    for ( i in 1:n_iters ) {
        optUpdate( sess, optSteps, data, placeholders, minibatch_size )
        if ( i %% 100 == 0 ) {
            printProgress( sess, estLogPostOpt, data, placeholders, i, minibatch_size, params )
        }
    }
    calcFullGrads( sess, optSteps, data, placeholdersFull )
    # Run SGNHT dynamics on each parameter for n_iters
    writeLines( "Sampling using SGNHT-CV" )
    for ( i in 1:n_iters ) {
        updateSGNHT( sess, dynamics, data, placeholders, n )
        paramStorage = storeState( sess, i, params, paramStorage )
        if ( i %% 100 == 0 ) {
            printProgress( sess, estLogPost, data, placeholders, i, minibatch_size, params )
        }
    }
    return( paramStorage )
}
