library(tensorflow)

data_feed = function( data, placeholders, minibatch_size ) {
    # Creates the data drip feed to the algorithm.
    feed_dict = dict()
    N = dim( data[[1]] )[1]
    selection = sample( N, minibatch_size )
    input_names = names( placeholders )
    for ( input in input_names ) {
        feed_dict[[ placeholders[[input]] ]] = dataSelect( data[[input]], selection )
    }
    return( feed_dict )
}

dataSelect = function( data, selection ) {
    # Subset data based on selection across general dimension containers
    dataDim = dim( data )
    d = length( dataDim )
    # Handle the vector and 1d matrix case
    if ( d < 2 ) {
        return( data[selection] )
    }
    # Create do.call expression for `[` slice operator, providing required dimensionality
    argList = list( data, selection )
    for ( i in 2:d ) {
        argList[[i+1]] = 1:dataDim[i]
    }
    argList = c( argList, list( drop = FALSE ) )
    return( do.call( `[`, argList ) )
}

feedFullDataset = function( data, placeholders ) {
    # Feeds the full dataset to the current operation
    feed_dict = dict()
    for ( input in names( placeholders ) ) {
        feed_dict[[ placeholders[[input]] ]] = data[[input]]
    }
    return( feed_dict )
}

initSess = function() { 
    # Initialise tensorflow session
    sess = tf$Session()
    init = tf$global_variables_initializer()
    sess$run(init)
    return(sess)
}

checkDivergence = function( sess, sgmcmc, iter, verbose ) {
    # check divergence of chain and print progress if verbose == TRUE
    currentEstimate = sess$run( sgmcmc$estLogPost, feed_dict = data_feed( 
            sgmcmc$data, sgmcmc$placeholders, sgmcmc$n ) )
    if ( is.nan( currentEstimate ) ) {
        stop("Chain diverged")
    }
    if ( verbose ) {
        writeLines( paste0( "Iteration: ", iter, "\t\tLog posterior estimate: ", currentEstimate ) )
    }
}

getParams = function( sess, sgmcmc ) {
    return( sess$run( sgmcmc$params ) )
}
