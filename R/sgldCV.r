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
source("controlVariates.r")

declareDynamics = function( estLogPost, estLogPostOpt, gradFull, params, paramsOpt, stepsize ) {
    # Initialize SGLD tensorflow by declaring Langevin Dynamics
    #
    # List contains operations consisting of one SGLD update across all parameters
    step_list = list()
    # Calculate optimized gradient
    gradEst = calcCVGradient( estLogPost, estLogPostOpt, gradFull, params, paramsOpt )
    # Declare SGLD dynamics using tensorflow autodiff
    for ( pname in names( params ) ) {
        paramCurr = params[[pname]]
        gradCurr = gradEst[[pname]]
        step_list[[pname]] = paramCurr$assign_add( 0.5 * stepsize[[pname]] * gradCurr + sqrt( stepsize[[pname]] ) * tf$random_normal( paramCurr$get_shape() ) )
    }
    return( step_list )
}

setupSGLDCV = function( logLik, logPrior, data, paramsRaw, stepsize, optStepsize, n, gibbsParams ) {
    # 
    # Get dataset size
    N = dim( data[[1]] )[1]
    # Convert params and data to tensorflow variables and placeholders
    params = setupParams( paramsRaw )
    placeholders = setupPlaceholders( data, n )
    # Declare tensorflow variables for initial optimizer
    paramsOpt = setupParams( paramsRaw )
    placeholdersFull = setupFullPlaceholders( data )
    # Declare container for full gradient
    gradFull = setupFullGradients( paramsRaw )
    # Declare estimated log posterior tensor using declared variables and placeholders
    estLogPost = setupEstLogPost( logLik, logPrior, params, placeholders, N, n, gibbsParams )
    # Declare estimated log posterior tensor for optimization
    estLogPostOpt = setupEstLogPost( logLik, logPrior, paramsOpt, placeholders, N, n, gibbsParams )
    # Declare full log posterior for calculation at MAP estimate
    fullLogPostOpt = setupFullLogPost( logLik, logPrior, paramsOpt, placeholdersFull, gibbsParams )
    # Declare optimizer ADD STEPSIZE AS A PARAMETER
    optimizer = declareOptimizer( estLogPostOpt, fullLogPostOpt, paramsOpt, 
            params, gradFull, optStepsize )
    # Declare SGLD dynamics
    dynamics = declareDynamics( estLogPost, estLogPostOpt, gradFull, params, paramsOpt, stepsize )
    # Declare SGLDCV object
    sgldCV = list( "dynamics" = dynamics, "optimizer" = optimizer, "data" = data, "n" = n, 
            "placeholders" = placeholders, "placeholdersFull" = placeholdersFull, 
            "params" = params, "estLogPost" = estLogPost, "estLogPostOpt" = estLogPostOpt )
    return( sgldCV )
}

updateSGLDCV = function( sess, sgldCV ) {
    # Perform one step of the declared dynamics
    feedCurr = data_feed( sgldCV$data, sgldCV$placeholders, sgldCV$n )
    for ( step in sgldCV$dynamics ) {
        sess$run( step, feed_dict = feedCurr )
    }
}

runSGLDCV = function( logLik, logPrior, data, paramsRaw, stepsize, optStepsize, 
            n, nIters = 10^4, nItersOpt = 10^4, verbose = TRUE ) {
    # Setup SGLDCV object
    sgldCV = setupSGLDCV( logLik, logPrior, data, paramsRaw, stepsize, optStepsize, n, NULL )
    # Initialize storage
    paramStorage = initStorage( paramsRaw, nIters )
    # Initalize tensorflowsession
    sess = initSess()
    # Run initial optimization to find mode of parameters
    getMode( sess, sgldCV, nItersOpt, verbose )
    # Run Langevin dynamics on each parameter for n_iters
    if ( verbose ) {
        writeLines( "\nSampling using SGLD-CV..." )
    }
    for ( i in 1:nIters ) {
        updateSGLDCV( sess, sgldCV )
        paramStorage = storeState( sess, i, sgldCV, paramStorage )
        if ( i %% 100 == 0 ) {
            checkDivergence( sess, sgldCV, i, verbose )
        }
    }
    return( paramStorage )
}
