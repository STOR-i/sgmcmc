library(tensorflow)

setupParams = function( params ) {
    # Redeclare parameters as tensorflow variables
    param_names = names( params )
    tfParams = list()
    for ( pname in param_names ) {
        tfParams[[pname]] = tf$Variable( params[[pname]], dtype = tf$float32 )
    }
    return( tfParams )
}

setupPlaceholders = function( data, minibatch_size ) {
    # Declare placeholders for each dataset
    data_names = names( data )
    tfPlaceholders = list()
    for ( dname in data_names ) {
        current_size = dim( data[[dname]] )
        current_size[1] = minibatch_size
        tfPlaceholders[[dname]] = tf$placeholder( tf$float32, current_size )
    }
    return( tfPlaceholders )
}

setupEstLogPost = function( logLik, logPrior, params, placeholders, N, n, gibbsParams ) {
    # Declare log posterior estimate
    correction = tf$constant( N / n, dtype = tf$float32 )
    if ( is.null( gibbsParams ) ) {
        estLogPost = logPrior( params, placeholders ) + 
                correction * logLik( params, placeholders )
    } else {
        estLogPost = logPrior( params, placeholders, gibbsParams ) + 
                correction * logLik( params, placeholders, gibbsParams )
    }
    return( estLogPost )
}

setupGrads = function( params ) {
    # Create placeholders to hold current gradients, used for storage
    tfGrads = list()
    for ( pname in names( params ) ) {
        tfGrads[[pname]] = tf$Variable( params[[pname]], dtype = tf$float32 )
    }
    return( tfGrads )
}

getRanks = function( paramsRaw ) {
    # Get the rank of each of the parameter objects
    ranks = list()
    for ( pname in names( paramsRaw ) ) {
        param = paramsRaw[[pname]]
        ranks[[pname]] = length( dim( param ) )
    }
    return( ranks )
}
