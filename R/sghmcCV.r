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

declareDynamics = function( estLogPost, estLogPostOpt, gradFull, params, paramsOpt, eta, alpha ) {
    # Initialize SGLD tensorflow by declaring Langevin Dynamics
    #
    # List contains operations consisting of one SGLD update across all parameters
    param_names = names( params )
    step_list = list( "dynamics" = list(), "momentum" = list(), "refresh" = list() )
    vs = list()
    gradEst = calcCVGradient( estLogPost, estLogPostOpt, gradFull, params, paramsOpt )
    for ( param_name in param_names ) {
        # Declare momentum params and reparameterize
        param_current = params[[param_name]]
        vs[[param_name]] = tf$Variable( tf$random_normal( param_current$get_shape() ) )
        v_current = vs[[param_name]]
        step_list$refresh[[param_name]] = v_current$assign( sqrt( eta[[param_name]] ) * tf$random_normal( param_current$get_shape() ) )
        grad = gradEst[[param_name]]
        step_list$momentum[[param_name]] = v_current$assign( eta[[param_name]]*grad + alpha[[param_name]]*v_current + sqrt( eta[[param_name]] * alpha[[param_name]] / 4 ) * tf$random_normal( param_current$get_shape() ) )
        step_list$dynamics[[param_name]] = param_current$assign_add( v_current )
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
sghmcCV = function( calcLogLik, calcLogPrior, data, paramsRaw, eta, alpha, L, minibatch_size, 
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
    optSteps = declareOptimizer( estLogPostOpt, fullLogPostOpt, paramsOpt, params, gradFull, 1e-5 )
    # Declare SGLD dynamics
    dynamics = declareDynamics( estLogPost, estLogPostOpt, gradFull, params, 
            paramsOpt, eta, alpha )
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
    writeLines( "Sampling using SGHMC-CV" )
    for ( i in 1:n_iters ) {
        updateSGHMC( sess, dynamics, data, placeholders, minibatch_size, L )
        if ( i %% 100 == 0 ) {
            printProgress( sess, estLogPost, data, placeholders, i, minibatch_size, params )
        }

    }
}
