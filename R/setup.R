# Create generic sgmcmc object
createSGMCMC = function( logLik, logPrior, dataset, params, stepsize, minibatchSize, seed ) { 
    # First check tf installation using tf_status. Throw error if it isn't installed.
    #checkTFInstall() #comment this out for now until we set-up a better package check
    # Set seed if required, TensorFlow seeds set inside dynamics
    if ( !is.null( seed ) ) {
        tf$set_random_seed(seed)
        set.seed(seed)
    }
    # Get dataset size
    N = getDatasetSize( dataset )
    # If minibatchSize is a proportion, convert to an integer
    minibatchSize = convertProp( minibatchSize, N )
    # Convert params and dataset to TensorFlow variables and placeholders
    paramstf = setupParams( params )
    placeholders = setupPlaceholders( dataset, minibatchSize )
    # Declare estimated log posterior tensor using declared variables and placeholders
    # Check for tf$float64 errors (we only use tf$float32), if so throw a more explanatory error
    estLogPost = tryCatch({ 
        setupEstLogPost( logLik, logPrior, paramstf, placeholders, N, minibatchSize )
    }, error = function ( e ) getPosteriorBuildError( e ) )
    # Check stepsize tuning constants are in list format
    stepsize = convertList( stepsize, params )
    # Declare sgmcmc object as list, return for custom tuning constants to be added by calling method
    sgmcmc = list( "N" = N, "data" = dataset, "n" = minibatchSize, "placeholders" = placeholders, 
            "stepsize" = stepsize, "params" = paramstf, "estLogPost" = estLogPost )
    return( sgmcmc )
}

# getGradients generic defines different gradient estimates for 
# control variate and non-control variate methods
# @param sgmcmc a stochastic gradient mcmc object, as defined in the respective modules sgld.r etc.
getGradients = function( sgmcmc ) UseMethod("getGradients")

# Get gradient estimates for standard SGMCMC method
getGradients.sgmcmc = function( sgmcmc ) {
    estLogPostGrads = list()
    for ( pname in names( sgmcmc$params ) ) {
        estLogPostGrads[[pname]] = tf$gradients( sgmcmc$estLogPost, sgmcmc$params[[pname]] )[[1]]
    }
    return( estLogPostGrads )
}

# Get gradient estimates for control variate methods, handle IndexedSlices gradients gracefully
getGradients.sgmcmccv = function( sgmcmcCV ) {
    estLogPostGrads = list()
    for ( pname in names( sgmcmcCV$params ) ) {
        gradCurr = tf$gradients( sgmcmcCV$estLogPost, sgmcmcCV$params[[pname]] )[[1]]
        isSparse = gradIsIndexed(gradCurr)
        optGradCurr = tf$gradients( sgmcmcCV$estLogPostOpt, sgmcmcCV$paramsOpt[[pname]] )[[1]]
        optGradFull = sgmcmcCV$logPostOptGrad[[pname]]
        if (isSparse) {
            # Get the current gradient estimate but ensure to keep it as IndexedSlices object
            fullGradCurr = tf$gather(optGradFull, gradCurr$indices)
            currentVals = fullGradCurr - optGradCurr$values + gradCurr$values
            estLogPostGrads[[pname]] = tf$IndexedSlices(currentVals, gradCurr$indices)
        } else {
            estLogPostGrads[[pname]] = optGradFull - optGradCurr + gradCurr
        }
    }
    return( estLogPostGrads )
}

# Check if gradCurr is IndexedSlices object for minibatched parameters
gradIsIndexed = function(grad) {
    isSparse = tryCatch({
        temp = grad$indices
        TRUE
    }, error = function (e) { 
        return(FALSE)
    })
    return(isSparse)
}

# Calculate size of dataset
getDatasetSize = function( data ) {
    N = dim( data[[1]] )[1]
    # Check edge case that data[[1]] is 1d
    if ( is.null( N ) ) {
        N = length( data[[1]] )
    }
    return( N )
}

# Calculate shape of dataset and params
getShape = function( input ) {
    shapeInput = dim( input )
    # Check edge case that input[[1]] is 1d
    if ( is.null( shapeInput ) ) {
        shapeInput = c( length( input ) )
    }
    return( shapeInput )
}

# Redeclare parameters as TensorFlow variables, initialize at starting values
setupParams = function( params ) {
    tfParams = list()
    for ( pname in names( params ) ) {
        tfParams[[pname]] = tf$Variable( params[[pname]], dtype = tf$float32 )
    }
    return( tfParams )
}

# Declare TensorFlow placeholders for each dataset based on data list and minibatch size n
setupPlaceholders = function( data, n ) {
    tfPlaceholders = list()
    for ( dname in names( data ) ) {
        shapeCurr = getShape( data[[dname]] )
        shapeCurr[1] = n
        tfPlaceholders[[dname]] = tf$placeholder( tf$float32, shapeCurr )
    }
    return( tfPlaceholders )
}

# Use defined logLik & logPrior functions to declare unbiased estimate of log posterior
setupEstLogPost = function( logLik, logPrior, params, placeholders, N, n ) {
    correction = tf$constant( N / n, dtype = tf$float32 )
    # If logPrior is NULL then use uninformative prior
    if ( is.null( logPrior ) ) {
        estLogPost = correction * logLik( params, placeholders )
    } else {
        estLogPost = logPrior( params ) + correction * logLik( params, placeholders )
    }
    return( estLogPost )
}

# If minibatch size is a proportion (i.e. < 1) convert to an integer
convertProp = function( minibatchSize, N ) {
    if ( minibatchSize < 1 ) {
        out = round( minibatchSize * N )
        if ( out == 0 ) {
            stop( paste0( "minibatchSize proportion (", minibatchSize, 
                    ") too small for dataset size (", N, ")." ) )
        } else {
            return( round( minibatchSize * N ) )
        }
    } else {
        return( minibatchSize )
    }
}

# Get the rank of each of the parameter objects, needed for sghmc
# @assume paramsRaw object is the original list of numeric R arrays, not TensorFlow tensors
getRanks = function( paramsRaw ) {
    ranks = list()
    for ( pname in names( paramsRaw ) ) {
        param = paramsRaw[[pname]]
        # Catch case that param is a vector
        if ( is.null( dim( param ) ) ) {
            ranks[[pname]] = as.numeric( length( param ) > 1 )
        } else {
            ranks[[pname]] = length( dim( param ) )
        }
    }
    return( ranks )
}

# If stepsizes or a or alpha tuning constants are not in a list
# format then convert to a list format
convertList = function( tuningConst, params ) {
    # Do nothing if already a list
    if( typeof( tuningConst ) == "list" ) {
        return( tuningConst )
    }
    convertedConsts = list()
    for ( pname in names( params ) ) {
        convertedConsts[[pname]] = tuningConst
    }
    return( convertedConsts )
}
