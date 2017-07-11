# Perform a single MCMC step using the dynamics defined in the sgmcmc object
# 
# This function performs a single update of the sgmcmc dynamics declared in the sgmcmc object,
# and updates the sgmcmc$params objects accordingly.
mcmcStep = function( sgmcmc, sess ) UseMethod("mcmcStep")

# Perform one step of SGLD
mcmcStep.sgld = function( sgld, sess ) {
    # Sample minibatch of data
    feedCurr = dataFeed( sgld$data, sgld$placeholders, sgld$n )
    for ( step in sgld$dynamics ) {
        sess$run( step, feed_dict = feedCurr )
    }
}

# Perform one step of SGHMC
mcmcStep.sghmc = function( sghmc, sess ) {
    # Refresh momentum
    for ( step in sghmc$dynamics$refresh ) {
        sess$run( step )
    }
    for ( l in 1:sghmc$L ) {
        # Sample minibatch of data
        feedCurr = dataFeed( sghmc$data, sghmc$placeholders, sghmc$n )
        for ( pname in names( sghmc$params ) ) {
            sess$run( sghmc$dynamics$nu[[pname]], feed_dict = feedCurr )
            sess$run( sghmc$dynamics$theta[[pname]], feed_dict = feedCurr )
        }
    }
}

# Perform one step of SGNHT
mcmcStep.sgnht = function( sgnht, sess ) {
    # Refresh momentum
    feedCurr = dataFeed( sgnht$data, sgnht$placeholders, sgnht$n )
    for ( step in sgnht$dynamics$u ) {
        sess$run( step, feed_dict = feedCurr )
    }
    for ( step in sgnht$dynamics$theta ) {
        sess$run( step, feed_dict = feedCurr )
    }
    for ( step in sgnht$dynamics$alpha ) {
        sess$run( step, feed_dict = feedCurr )
    }
}

# Run stochastic gradient MCMC for declared dynamics
# @param paramsRaw is the original list of numeric arrays used to define parameter starting points.
#   This is needed to calculate the dimensions of the storage arrays.
runSGMCMC = function( sgmcmc, paramsRaw, options ) UseMethod("runSGMCMC")

# SGMCMC run when control variates are not used
runSGMCMC.sgmcmc = function( sgmcmc, paramsRaw, options ) {
    # Initialize storage
    paramStorage = initStorage( paramsRaw, options$nIters )
    # Initalize TensorFlow session
    sess = initSess( options$verbose )
    # Perform SGMCMC for desired iterations, storing parameters at each step
    for ( i in 1:options$nIters ) {
        mcmcStep( sgmcmc, sess )
        paramStorage = storeState( sess, i, sgmcmc, paramStorage )
        if ( i %% 100 == 0 ) {
            # If chain has diverged throw an error, this function also prints progress
            checkDivergence( sess, sgmcmc, i, options$verbose )
        }
    }
    return( paramStorage )
}

# Run stochastic gradient MCMC with Control Variates for declared dynamics
runSGMCMC.sgmcmcCV = function( sgmcmcCV, paramsRaw, options ) {
    # Initialize storage
    paramStorage = initStorage( paramsRaw, options$nIters )
    # Initalize TensorFlow session
    sess = initSess( options$verbose )
    # Run initial optimization to find mode of parameters
    getMode( sess, sgmcmcCV, options$nItersOpt, options$verbose )
    # Perform SGMCMCCV for desired iterations, storing parameters at each step
    if ( options$verbose ) {
        writeLines( "\nSampling using SGMCMC..." )
    }
    for ( i in 1:options$nIters ) {
        mcmcStep( sgmcmcCV, sess )
        paramStorage = storeState( sess, i, sgmcmcCV, paramStorage )
        if ( i %% 100 == 0 ) {
            # If chain has diverged throw an error, this function also prints progress
            checkDivergence( sess, sgmcmcCV, i, options$verbose )
        }
    }
    return( paramStorage )
}

# Creates the feed_dict for each TensorFlow placeholder to feed the minibatch of data
dataFeed = function( data, placeholders, n ) {
    feed_dict = dict()
    # Get dataset size
    N = getDatasetSize( data )
    # Get indices of subsample
    selection = sample( N, n )
    for ( input in names( placeholders ) ) {
        # Slice datasets at selection
        feed_dict[[ placeholders[[input]] ]] = dataSelect( data[[input]], selection )
    }
    return( feed_dict )
}

# Subset data based on selection across general dimension containers
dataSelect = function( data, selection ) {
    dataDim = dim( data )
    d = length( dataDim )
    # Handle the vector and 1d matrix edge case
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

# Initialise TensorFlow session and all global variables
initSess = function( verbose ) { 
    sess = tf$Session()
    init = tf$global_variables_initializer()
    sess$run(init)
    return(sess)
}

# Check for divergence of chain and print progress if verbose == TRUE
checkDivergence = function( sess, sgmcmc, iter, verbose ) {
    currentEstimate = sess$run( sgmcmc$estLogPost, feed_dict = dataFeed( 
            sgmcmc$data, sgmcmc$placeholders, sgmcmc$n ) )
    # If chain diverged throw an error
    if ( is.nan( currentEstimate ) ) {
        stop("Chain diverged, try decreasing stepsize")
    }
    if ( verbose ) {
        writeLines( paste0( "Iteration: ", iter, "\t\tLog posterior estimate: ", currentEstimate ) )
    }
}
