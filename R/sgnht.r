# Methods for implementing Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) using Tensorflow.
# Gradients are automatically calculated. The main function is sghmc, which implements a full
# SGHMC procedure for a given model, including gradient calculation.
#
# References:
#   1. T. Chen, E.B. Fox, and C. Guestrin.  
#           Stochastic gradient Hamiltonian Monte Carlo.  
#           In Proceeding of 31st International Conference on Machine Learning (ICML’14), 2014.


library(tensorflow)
library(reticulate)
source("setup.r")
source("update.r")
source("storage.r")

declareDynamics = function( estLogPost, params, stepsizes, aList, ranks ) {
    # Initialize SGNHT tensorflow by declaring Dynamics
    dynamics = list( "theta" = list(), "u" = list(), "alpha" = list() )
    for ( pname in names(params) ) {
        # Get constants for this parameter
        eta = stepsizes[[pname]]
        a = aList[[pname]]
        rankTheta = ranks[[pname]]
        # Declare momentum params
        theta = params[[pname]]
        u = tf$Variable( sqrt(eta) * tf$random_normal( theta$get_shape() ) )
        alpha = tf$Variable( a, dtype = tf$float32 )
        # Declare dynamics
        gradU = tf$gradients( estLogPost, theta )[[1]]
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

updateSGNHT = function( sess, sgnht ) {
    # Perform one step of SGNHT
    feedCurr = data_feed( sgnht$data, sgnht$placeholders, sgnht$n )
    for ( step in sgnht$dynamics$u ) {
        sess$run( step, feed_dict = feedCurr )
    }
    for ( step in sgnht$dynamics$theta ) {
        sess$run( step, feed_dict = feedCurr )
    }
    for ( step in sgnht$dynamics$alpha ) {
        sess$run( step, feed_dict = feedCurr )
    }
}

setupSGNHT = function( logLik, logPrior, data, paramsRaw, stepsize, a, n, gibbsParams ) {
    # 
    # Get dataset size
    N = dim( data[[1]] )[1]
    # Convert params and data to tensorflow variables and placeholders
    params = setupParams( paramsRaw )
    placeholders = setupPlaceholders( data, n )
    # Declare estimated log posterior tensor using declared variables and placeholders
    estLogPost = setupEstLogPost( logLik, logPrior, params, placeholders, N, n, gibbsParams )
    # Get ranks for each parameter tensor, required for dynamics
    ranks = getRanks( paramsRaw )
    # Declare SGNHT dynamics
    dynamics = declareDynamics( estLogPost, params, stepsize, a, ranks )
    # Create SGNHT object
    sgnht = list( "dynamics" = dynamics, "data" = data, "n" = n, "placeholders" = placeholders, 
            "params" = params, "estLogPost" = estLogPost )
    return( sgnht )
}

runSGNHT = function( logLik, logPrior, data, paramsRaw, stepsize, a, n, 
        nIters = 10^4, verbose = TRUE ) {
    # Declare SGNHT object
    sgnht = setupSGNHT( logLik, logPrior, data, paramsRaw, stepsize, a, n, NULL )
    # Initialize parameter storage
    paramStorage = initStorage( paramsRaw, nIters )
    # Initalize tensorflowsession
    sess = initSess()
    # Run Langevin dynamics on each parameter for n_iters
    for ( i in 1:nIters ) {
        updateSGNHT( sess, sgnht )
        paramStorage = storeState( sess, i, sgnht, paramStorage )
        if ( i %% 100 == 0 ) {
            checkDivergence( sess, sgnht, i, verbose )
        }
    }
    return( paramStorage )
}
