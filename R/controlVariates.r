library(tensorflow)

getMode = function( sess, sgmcmc, nIters = 10^4, verbose = TRUE ) {
    # Initial optimization of parameters
    if ( verbose ) {
        writeLines( "\nFinding initial MAP estimates..." )
    }
    for ( i in 1:nIters ) {
        optUpdate( sess, sgmcmc )
        if ( i %% 100 == 0 ) {
            checkOptDivergence( sess, sgmcmc, i, verbose )
        }
    }
    calcFullGrads( sess, sgmcmc )
}

optUpdate = function( sess, sgmcmc ) {
    # Perform one optimization step
    feedCurr = data_feed( sgmcmc$data, sgmcmc$placeholders, sgmcmc$n )
    sess$run( sgmcmc$optimizer$update, feed_dict = feedCurr )
}

setupFullLogPost = function( logLik, logPrior, params, placeholders, gibbsParams ) {
    # Declare full log posterior
    #
    # Separate function from setupEstLogPost avoids float precision errors from correction term
    if ( is.null( gibbsParams ) ) {
        logPost = logPrior( params, placeholders ) + logLik( params, placeholders )
    } else {
        logPost = logPrior( params, placeholders, gibbsParams ) + 
                logLik( params, placeholders, gibbsParams )
    }
    return( logPost )
}

setupFullPlaceholders = function( data ) {
    # Create placeholders to hold full dataset for full log posterior calculation
    data_names = names( data )
    tfPlaceholders = list()
    for ( dname in data_names ) {
        current_size = dim( data[[dname]] )
        tfPlaceholders[[dname]] = tf$placeholder( tf$float32, current_size )
    }
    return( tfPlaceholders )
}

setupFullGradients = function( params ) {
    # Create container for the full gradient value after optimization
    #
    # Declare containers as tensorflow variables. Gradients will be same shape as parameters
    param_names = names( params )
    gradientContainer = list()
    for ( pname in param_names ) {
        gradientContainer[[pname]] = tf$Variable( params[[pname]], dtype = tf$float32 )
    }
    return( gradientContainer )
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

calcFullGrads = function( sess, sgmcmc ) {
    # Calculate full gradient information at MAP estimate
    feedCurr = feedFullDataset( sgmcmc$data, sgmcmc$placeholdersFull )
    for ( pname in names( sgmcmc$params ) ) {
        sess$run( sgmcmc$optimizer$fullCalc[[pname]], feed_dict = feedCurr  )
        sess$run( sgmcmc$optimizer$reassign[[pname]] )
    }
}

checkOptDivergence = function( sess, sgmcmc, iter, verbose ) {
    # Check divergence of optimization procedure and print progress if verbose == TRUE
    currentEstimate = sess$run( sgmcmc$estLogPostOpt, feed_dict = data_feed( 
            sgmcmc$data, sgmcmc$placeholders, sgmcmc$n ) )
    if ( is.nan( currentEstimate ) ) {
        stop("Chain diverged")
    }
    if ( verbose ) {
        writeLines( paste0( "Iteration: ", iter, "\t\tLog posterior estimate: ", currentEstimate ) )
    }
}
