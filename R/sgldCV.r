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

calcCVGradient = function( estLogPost, estLogPostOpt, gradFull, params, paramsOpt ) {
    # Calculate reduced variance gradient estimate using control variates
    gradEst = list()
    for ( pname in names( params ) ) {
        paramCurr = params[[pname]]
        optParamCurr = paramsOpt[[pname]]
        gradCurr = tf$gradients( estLogPost, paramCurr )[[1]]
        optGradCurr = tf$gradients( estLogPostOpt, optParamCurr )[[1]]
        fullOptGradCurr = gradFull[[pname]]
        gradEst[[pname]] = fullOptGradCurr - optGradCurr + gradCurr
    }
    return( gradEst )
}

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

declareOptimizer = function( estLogPost, fullLogPost, paramsOpt, params, gradFull, optStepsize ) {
    # Initialize optimizer for MAP estimation step
    #
    optSteps = list()
    optimizer = tf$train$AdamOptimizer( 0.001 )
    optSteps[["update"]] = optimizer$minimize( -estLogPost)
    # Steps for calculating full gradient and setting initial parameter values at MAP estimate
    optSteps[["fullCalc"]] = list()
    optSteps[["reassign"]] = list()
    for ( pname in names( paramsOpt ) ) {
        paramOptCurr = paramsOpt[[pname]]
        paramCurr = params[[pname]]
        grad = tf$gradients( fullLogPost, paramOptCurr )[[1]]
        optSteps$fullCalc[[pname]] = gradFull[[pname]]$assign( grad )
        optSteps$reassign[[pname]] = paramCurr$assign( paramOptCurr )
    }
    return( optSteps )
}

# Add option to include a summary measure??
sgldCV = function( calcLogLik, calcLogPrior, data, paramsRaw, stepsize, minibatch_size, 
        n_iters = 10^4 ) {
    # 
    # Get key sizes and declare correction term for log posterior estimate
    n = getMinibatchSize( minibatch_size )
    N = dim( data[[1]] )[1]
    correction = tf$constant( N / minibatch_size, dtype = tf$float32 )
    # Convert params and data to tensorflow variables and placeholders
    params = setupParams( paramsRaw )
    placeholders = setupPlaceholders( data, minibatch_size )
    # Declare tensorflow variables for initial optimizer
    paramsOpt = setupParams( paramsRaw )
    placeholdersFull = setupFullPlaceholders( data )
    paramStorage = initStorage( paramsRaw, n_iters )
    # Declare container for full gradient
    gradFull = setupFullGradients( paramsRaw )
    # Declare estimated log posterior tensor using declared variables and placeholders
    logLik = calcLogLik( params, placeholders )
    logPrior = calcLogPrior( params, placeholders )
    estLogPost = logPrior + correction * logLik
    # Declare estimated log posterior tensor for optimization
    logLikOpt = calcLogLik( paramsOpt, placeholders )
    logPriorOpt = calcLogPrior( paramsOpt, placeholders )
    estLogPostOpt = logPriorOpt + correction * logLikOpt
    # Declare full log posterior for calculation at MAP estimate
    fullLogPostOpt = calcLogLik( paramsOpt, placeholdersFull ) + 
            calcLogPrior( paramsOpt, placeholdersFull )
    # Declare optimizer ADD STEPSIZE AS A PARAMETER
    optSteps = declareOptimizer( estLogPostOpt, fullLogPostOpt, paramsOpt, params, gradFull, 1e-6 )
    # Declare SGLD dynamics
    dynamics = declareDynamics( estLogPost, estLogPostOpt, gradFull, params, paramsOpt, stepsize )
    # Initalize tensorflowsession
    sess = initSess()
    # Initial optimization of parameters
    writeLines( "Finding initial MAP estimates" )
    for ( i in 1:n_iters ) {
        optUpdate( sess, optSteps, data, placeholders, minibatch_size )
        if ( i %% 100 == 0 ) {
            printProgress( sess, estLogPostOpt, data, placeholders, i, minibatch_size, params )
        }
    }
    calcFullGrads( sess, optSteps, data, placeholdersFull )
    # Run Langevin dynamics on each parameter for n_iters
    writeLines( "Sampling using SGLD-CV" )
    for ( i in 1:n_iters ) {
        updateSGLD( sess, dynamics, data, placeholders, minibatch_size )
        paramStorage = storeState( sess, i, params, paramStorage )
        if ( i %% 100 == 0 ) {
            printProgress( sess, estLogPost, data, placeholders, i, minibatch_size, params )
        }
    }
    return( paramStorage )
}
