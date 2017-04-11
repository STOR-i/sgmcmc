# Initialise storage array, this will hold the MCMC chain and is the output returned by e.g. runSGLD
initStorage = function( paramsRaw, n_iters ) {
    paramStorage = list()
    for ( pname in names( paramsRaw ) ) {
        shapeCurrent = dim( paramsRaw[[pname]] )
        # Add new dimension to parameter's current shape which holds each iteration
        shapeCurrent = c( n_iters, shapeCurrent )
        paramStorage[[pname]] = array( 0, dim = shapeCurrent )
    }
    return( paramStorage )
}

# Store the current parameter values
storeState = function( sess, iter, sgld, storage ) {
    for ( pname in names( sgld$params ) ) {
        paramCurrent = sgld$params[[pname]]$eval( sess )
        storage[[pname]] = updateStorage( storage[[pname]], iter, paramCurrent )
    }
    return( storage )
}

# Given a current parameter value, update storage at the current iteration
# Enables slicing along the first dimension of a storage array of general dimension at index.
updateStorage = function( storage, index, params ) {
    d = length( dim( storage ) )
    # Catch edge cases of rank 1 storage array
    if ( d < 2 ) {
        storage[index] = params
    } else {
        # Use matrix indexing to specify array slice for rank > 1
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
