library(tensorflow)

# Define declareDynamics generic, defined for each SGMCMC method in their respective modules
# @param sgmcmc a stochastic gradient mcmc object, as defined in the respective modules sgld.r etc.
declareDynamics = function( sgmcmc ) UseMethod("declareDynamics")

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

# Get gradient estimates for control variate methods
getGradients.sgmcmcCV = function( sgmcmcCV ) {
    estLogPostGrads = list()
    for ( pname in names( sgmcmcCV$params ) ) {
        gradCurr = tf$gradients( sgmcmcCV$estLogPost, sgmcmcCV$params[[pname]] )[[1]]
        optGradCurr = tf$gradients( sgmcmcCV$estLogPostOpt, sgmcmcCV$paramsOpt[[pname]] )[[1]]
        optGradFull = sgmcmcCV$logPostOptGrad[[pname]]
        estLogPostGrads[[pname]] = optGradFull - optGradCurr + gradCurr
    }
    return( estLogPostGrads )
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

# Redeclare parameters as tensorflow variables, keep starting values
setupParams = function( params ) {
    tfParams = list()
    for ( pname in names( params ) ) {
        tfParams[[pname]] = tf$Variable( params[[pname]], dtype = tf$float32 )
    }
    return( tfParams )
}

# Declare tensorflow placeholders for each dataset based on data list and minibatch size n
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
setupEstLogPost = function( logLik, logPrior, params, placeholders, N, n, gibbsParams ) {
    correction = tf$constant( N / n, dtype = tf$float32 )
    # If parameters are declared that will updated using Gibbs then the passed logPrior & logLik 
    # functions will also take gibbsParams as arguments
    if ( is.null( gibbsParams ) ) {
        estLogPost = logPrior( params, placeholders ) + 
                correction * logLik( params, placeholders )
    } else {
        estLogPost = logPrior( params, placeholders, gibbsParams ) + 
                correction * logLik( params, placeholders, gibbsParams )
    }
    return( estLogPost )
}

# Get the rank of each of the parameter objects, needed for sghmc
# @assume paramsRaw object is the original list of numeric R arrays, not tensorflow tensors
getRanks = function( paramsRaw ) {
    ranks = list()
    for ( pname in names( paramsRaw ) ) {
        param = paramsRaw[[pname]]
        ranks[[pname]] = length( dim( param ) )
    }
    return( ranks )
}
