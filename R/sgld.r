# Methods for implementing Stochastic Gradient Langevin Dynamics (SGLD) using Tensorflow.
# Gradients are automatically calculated. The main function is sgld, which implements a full
# SGLD procedure for a given model, including gradient calculation.
#
# References:
#   1. Welling, M. and Teh, Y. W. (2011). 
#       Bayesian learning via stochastic gradient Langevin dynamics. 
#       In Proceedings of the 28th International Conference on Machine Learning (ICML-11), 
#       pages 681–688.

library(tensorflow)
source("setup.r")
source("update.r")
source("storage.r")

declareDynamics = function( estLogPost, params, stepsize ) {
    # Initialize SGLD tensorflow by declaring Langevin Dynamics
    #
    # List contains operations consisting of one SGLD update across all parameters
    dynamics = list()
    # Declare SGLD dynamics using tensorflow autodiff
    for ( pname in names( params ) ) {
        theta = params[[pname]]
        epsilon = stepsize[[pname]]
        grad = tf$gradients( estLogPost, theta )[[1]]
        dynamics[[pname]] = theta$assign_add( 0.5 * epsilon * grad + 
                sqrt( epsilon ) * tf$random_normal( theta$get_shape() ) )
    }
    return( dynamics )
}

updateSGLD = function( sess, sgld ) {
    # Perform one step of the declared dynamics
    feedCurr = data_feed( sgld$data, sgld$placeholders, sgld$n )
    for ( step in sgld$dynamics ) {
        sess$run( step, feed_dict = feedCurr )
    }
}

setupSGLD = function( logLik, logPrior, data, paramsRaw, stepsize, n, gibbsParams ) {
    # 
    # Get dataset size
    N = dim( data[[1]] )[1]
    # Convert params and data to tensorflow variables and placeholders
    params = setupParams( paramsRaw )
    placeholders = setupPlaceholders( data, n )
    # Declare estimated log posterior tensor using declared variables and placeholders
    estLogPost = setupEstLogPost( logLik, logPrior, params, placeholders, N, n, gibbsParams )
    # Declare SGLD dynamics
    dynamics = declareDynamics( estLogPost, params, stepsize )
    # Declare SGLD object
    sgld = list( "dynamics" = dynamics, "data" = data, "n" = n, "placeholders" = placeholders, 
            "params" = params, "estLogPost" = estLogPost )
    return( sgld )
}

runSGLD = function( logLik, logPrior, data, paramsRaw, stepsize, n, nIters = 10^4, verbose = TRUE ) {
    sgld = setupSGLD( logLik, logPrior, data, paramsRaw, stepsize, n, NULL )
    paramStorage = initStorage( paramsRaw, nIters )
    # Initialize tensorflow session
    sess = initSess()
    # Run Langevin dynamics on each parameter for n_iters
    for ( i in 1:nIters ) {
        updateSGLD( sess, sgld )
        paramStorage = storeState( sess, i, sgld, paramStorage )
        if ( i %% 100 == 0 ) {
            checkDivergence( sess, sgld, i, verbose )
        }
    }
    return( paramStorage )
}
