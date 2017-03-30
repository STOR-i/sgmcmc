setup = function( data, params, minibatch_size ) {
    data_names = names( data )
    N = dim( data[[1]] )[1]
    if ( is.null(N) ) {
        N = length( data )
    }
    for ( data_name in data_names ) {
        if ( dim( data[[data_name]] )[1] != N || length( data[[data_name]] ) != N ) {
            stop("Number of observations do not match across datasets")
