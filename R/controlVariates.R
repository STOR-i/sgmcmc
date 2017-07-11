# Create generic sgmcmccv object from sgmcmc object
createSGMCMCCV = function( logLik, logPrior, dataset, params, stepsize, optStepsize, 
        minibatchSize ) {
    # First create generic sgmcmc object then add specifics
    sgmcmccv = createSGMCMC( logLik, logPrior, dataset, params, stepsize, minibatchSize )
    # If minibatchSize is a proportion, convert to an integer
    minibatchSize = convertProp( minibatchSize, sgmcmccv$N )
    # Declare TensorFlow variables for initial optimizer
    sgmcmccv$paramsOpt = setupParams( params )
    sgmcmccv$placeholdersFull = setupFullPlaceholders( dataset )
    # Declare container for full gradients at mode
    sgmcmccv$logPostOptGrad = setupFullGradients( params )
    # Declare estimated log posterior tensor for optimization
    sgmcmccv$estLogPostOpt = setupEstLogPost( 
        logLik, logPrior, sgmcmccv$paramsOpt, sgmcmccv$placeholders, sgmcmccv$N, minibatchSize )
    # Declare full log posterior for calculation at MAP estimate
    sgmcmccv$fullLogPostOpt = setupFullLogPost( 
        logLik, logPrior, sgmcmccv$paramsOpt, sgmcmccv$placeholdersFull )
    # Declare optimizer
    sgmcmccv$optimizer = declareOptimizer( sgmcmccv$estLogPostOpt, sgmcmccv$fullLogPostOpt, 
        sgmcmccv$paramsOpt, sgmcmccv$params, sgmcmccv$logPostOptGrad, optStepsize )
    return( sgmcmccv )
}

# This function performs a single optimization step on the control variate parameters.
# This initial optimization procedure is needed before SGMCMCCV is applied in order to
# ensure the control variate parameters estimate the mode well.
optUpdate = function( sess, sgmcmc ) {
    feedCurr = dataFeed( sgmcmc$data, sgmcmc$placeholders, sgmcmc$n )
    sess$run( sgmcmc$optimizer$update, feed_dict = feedCurr )
}

# Initial optimization of parameters for Control Variate methods.
# Needed to ensure control variate parameters estimate posterior mode.
getMode = function( sess, sgmcmc, nIters = 10^4, verbose = TRUE ) {
    # If verbose parameter is TRUE, print progress
    if ( verbose ) {
        writeLines( "\nFinding initial MAP estimates..." )
    }
    for ( i in 1:nIters ) {
        # Single update of optimization
        optUpdate( sess, sgmcmc )
        if ( i %% 100 == 0 ) {
            checkOptDivergence( sess, sgmcmc, i, verbose )
        }
    }
    # Calculate the full gradient of the log posterior (i.e. using the full dataset) 
    #   at the mode estimates
    calcFullGrads( sess, sgmcmc )
}

# Declare full log posterior density from logLik and logPrior functions
setupFullLogPost = function( logLik, logPrior, params, placeholders ) {
    # Separate function from setupEstLogPost avoids float precision errors from correction term
    if ( is.null( logPrior ) ) {
        logPost = logLik( params, placeholders )
    } else {
        logPost = logPrior( params ) + logLik( params, placeholders )
    }
    return( logPost )
}

# Create TensorFlow placeholders to hold full dataset for full log posterior calculation
setupFullPlaceholders = function( data ) {
    tfPlaceholders = list()
    for ( dname in names( data ) ) {
        current_size = dim( data[[dname]] )
        tfPlaceholders[[dname]] = tf$placeholder( tf$float32, current_size )
    }
    return( tfPlaceholders )
}

# Create TensorFlow variables to hold estimates of the full log posterior gradient at the mode
setupFullGradients = function( params ) {
    gradientContainer = list()
    for ( pname in names( params ) ) {
        gradientContainer[[pname]] = tf$Variable( params[[pname]], dtype = tf$float32 )
    }
    return( gradientContainer )
}

# Initialize optimizer which finds estimates of the mode of the log posterior
declareOptimizer = function( estLogPost, fullLogPost, paramsOpt, params, gradFull, optStepsize ) {
    optSteps = list()
    optimizer = tf$train$AdamOptimizer( optStepsize )
    optSteps[["update"]] = optimizer$minimize( -estLogPost )
    optSteps[["fullCalc"]] = list()
    optSteps[["reassign"]] = list()
    for ( pname in names( paramsOpt ) ) {
        # Declare current parameters
        paramOptCurr = paramsOpt[[pname]]
        paramCurr = params[[pname]]
        # Declare procedure to calculate the full log posterior gradient at the mode
        grad = tf$gradients( fullLogPost, paramOptCurr )[[1]]
        optSteps$fullCalc[[pname]] = gradFull[[pname]]$assign( grad )
        # This procedure assigns the MCMC starting values to the mode estimates
        optSteps$reassign[[pname]] = paramCurr$assign( paramOptCurr )
    }
    return( optSteps )
}

# Calculates full log posterior gradient estimate at the mode
calcFullGrads = function( sess, sgmcmc ) {
    feedCurr = feedFullDataset( sgmcmc$data, sgmcmc$placeholdersFull )
    for ( pname in names( sgmcmc$params ) ) {
        sess$run( sgmcmc$optimizer$fullCalc[[pname]], feed_dict = feedCurr  )
        sess$run( sgmcmc$optimizer$reassign[[pname]] )
    }
}

# Check divergence of optimization procedure and print progress if verbose == TRUE
checkOptDivergence = function( sess, sgmcmc, iter, verbose ) {
    currentEstimate = sess$run( sgmcmc$estLogPostOpt, feed_dict = dataFeed( sgmcmc$data, 
            sgmcmc$placeholders, sgmcmc$n ) )
    # If log posterior estimate is NAN, chain is diverged, stop
    if ( is.nan( currentEstimate ) ) {
        stop("Chain diverged")
    }
    if ( verbose ) {
        writeLines( paste0( "Iteration: ", iter, "\t\tLog posterior estimate: ", currentEstimate ) )
    }
}

# Feeds the full dataset to the current operation, used by calcFullGrads
feedFullDataset = function( data, placeholders ) {
    feed_dict = dict()
    for ( input in names( placeholders ) ) {
        feed_dict[[ placeholders[[input]] ]] = data[[input]]
    }
    return( feed_dict )
}

