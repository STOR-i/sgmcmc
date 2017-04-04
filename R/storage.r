initStorage = function( paramsRaw, n_iters ) {
    paramStorage = list()
    for ( pname in names( paramsRaw ) ) {
        shapeCurrent = dim( paramsRaw[[pname]] )
        shapeCurrent = c( n_iters, shapeCurrent )
        paramStorage[[pname]] = array( 0, dim = shapeCurrent )
    }
    return( paramStorage )
}

storeState = function( sess, iter, params, paramStorage ) {
    for ( pname in names( params ) ) {
        paramCurrent = params[[pname]]$eval( sess )
        paramStorage[[pname]] = updateStorage( paramStorage[[pname]], iter, paramCurrent )
    }
    return( paramStorage )
}

updateStorage = function( storage, index, params ) {
    d = length( dim( storage ) )
    if ( d < 2 ) {
        storage[index] = params
    } else {
        storageDims = dim( storage )
        argList = list(index)
        for ( i in 2:d ) {
            argList[[i]] = 1:storageDims[i]
        }
        selection = as.matrix(do.call( "expand.grid", argList ))
        storage[selection] = params
    }
    return( storage )
}
