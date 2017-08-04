#' Single step of sgmcmc
#' 
#' Update parameters by performing a single sgmcmc step with dynamics as defined in the sgmcmc 
#'  object. This can be used to perform sgmcmc steps inside a loop as in standard
#'  TensorFlow optimization procedures.
#'  This is useful when high dimensional chains cannot fit into memory.
#' 
#' @param sgmcmc a stochastic gradient MCMC object returned by *Setup such as 
#'  \code{\link{sgldSetup}}, \code{\link{sgldcvSetup}} etc.
#' @param sess a TensorFlow session created using \code{\link{initSess}}
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Simulate from a Normal Distribution, unknown location and known scale with uninformative prior
#' # Run sgmcmc step by step and calculate estimate of location on the fly to reduce storage
#' dataset = list("x" = rnorm(1000))
#' params = list("theta" = 0)
#' logLik = function(params, dataset) {
#'     distn = tf$contrib$distributions$Normal(params$theta, 1)
#'     return(tf$reduce_sum(distn$log_prob(dataset$x)))
#' }
#' stepsize = list("theta" = 1e-4)
#' sgld = sgldSetup(logLik, dataset, params, stepsize)
#' nIters = 10^4L
#' # Initialize location estimate
#' locEstimate = 0
#' # Initialise TensorFlow session
#' sess = initSess(sgld)
#' for ( i in 1:nIters ) {
#'     sgmcmcStep(sgld, sess)
#'     locEstimate = locEstimate + 1 / nIters * getParams(sgld, sess)$theta
#' }
#' # For more examples see vignettes
#' }
sgmcmcStep = function( sgmcmc, sess ) UseMethod("sgmcmcStep")
# Method for sgld or sgldcv objects
#' @export
sgmcmcStep.sgld = function( sgmcmc, sess ) {
    # Sample minibatch of data
    feedCurr = dataFeed( sgmcmc$data, sgmcmc$placeholders, sgmcmc$n )
    for ( step in sgmcmc$dynamics ) {
        sess$run( step, feed_dict = feedCurr )
    }
}
# Method for sghmc or sghmccv objects
#' @export
sgmcmcStep.sghmc = function( sgmcmc, sess ) {
    # Refresh momentum
    for ( step in sgmcmc$dynamics$refresh ) {
        sess$run( step )
    }
    for ( l in 1:sgmcmc$L ) {
        # Sample minibatch of data
        feedCurr = dataFeed( sgmcmc$data, sgmcmc$placeholders, sgmcmc$n )
        for ( pname in names( sgmcmc$params ) ) {
            sess$run( sgmcmc$dynamics$nu[[pname]], feed_dict = feedCurr )
            sess$run( sgmcmc$dynamics$theta[[pname]], feed_dict = feedCurr )
        }
    }
}
# Method for sgnht or sgnhtcv objects
#' @export
sgmcmcStep.sgnht = function( sgmcmc, sess ) {
    # Refresh momentum
    feedCurr = dataFeed( sgmcmc$data, sgmcmc$placeholders, sgmcmc$n )
    for ( step in sgmcmc$dynamics$u ) {
        sess$run( step, feed_dict = feedCurr )
    }
    for ( step in sgmcmc$dynamics$theta ) {
        sess$run( step, feed_dict = feedCurr )
    }
    for ( step in sgmcmc$dynamics$alpha ) {
        sess$run( step, feed_dict = feedCurr )
    }
}


#' Initialise TensorFlow session and sgmcmc algorithm
#'
#' Initalise the TensorFlow session and the sgmcmc algorithm. For algorithms with control variates
#'  this will find the MAP estimates of the log posterior and calculate the full log posterior
#'  gradient at this point. For algorithms without control variates this will simply initialise a
#'  TensorFlow session.
#'
#' @param sgmcmc an sgmcmc object created using *Setup e.g. \code{\link{sgldSetup}}, 
#'  \code{\link{sgldcvSetup}}
#' @param verbose optional. Default TRUE. Boolean specifying whether to print progress.
#' 
#' @return sess a TensorFlow session, see the 
#'  \href{https://tensorflow.rstudio.com/}{TensorFlow for R website} for more details.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Simulate from a Normal Distribution, unknown location and known scale with uninformative prior
#' # Run sgmcmc step by step and calculate estimate of location on the fly to reduce storage
#' dataset = list("x" = rnorm(1000))
#' params = list("theta" = 0)
#' logLik = function(params, dataset) {
#'     distn = tf$contrib$distributions$Normal(params$theta, 1)
#'     return(tf$reduce_sum(distn$log_prob(dataset$x)))
#' }
#' stepsize = list("theta" = 1e-4)
#' sgld = sgldSetup(logLik, dataset, params, stepsize)
#' nIters = 10^4L
#' # Initialize location estimate
#' locEstimate = 0
#' # Initialise TensorFlow session
#' sess = initSess(sgld)
#' for ( i in 1:nIters ) {
#'     sgmcmcStep(sgld, sess)
#'     locEstimate = locEstimate + 1 / nIters * getParams(sgld, sess)$theta
#' }
#' # For more examples see vignettes
#' }
initSess = function( sgmcmc, verbose = TRUE ) UseMethod("initSess") 
# Method for standard sgmcmc objects: sgld, sghmc, sgnht
#' @export
initSess.sgmcmc = function( sgmcmc, verbose = TRUE ) {
    sess = tf$Session()
    init = tf$global_variables_initializer()
    sess$run(init)
    return(sess)
}
# Method for sgmcmc objects with control variates: sgldcv, sghmccv, sgnhtcv
#' @export
initSess.sgmcmccv = function( sgmcmc, verbose = TRUE ) {
    sess = tf$Session()
    init = tf$global_variables_initializer()
    sess$run(init)
    getMode(sgmcmc, sess, verbose)
    return(sess)
}

# Run stochastic gradient MCMC for declared dynamics
# @param paramsRaw is the original list of numeric arrays used to define parameter starting points.
#   This is needed to calculate the dimensions of the storage arrays.
runSGMCMC = function( sgmcmc, paramsRaw, options ) {
    # Initialize TensorFlow session and sgmcmc algorithm
    sess = initSess(sgmcmc, options$verbose)
    # Initialize storage
    paramStorage = initStorage( paramsRaw, options$nIters )
    if (options$verbose) {
        message("\nSimulating from SGMCMC algorithm...")
    }
    # Perform SGMCMC for desired iterations, storing parameters at each step
    for ( i in 1:options$nIters ) {
        sgmcmcStep( sgmcmc, sess )
        paramStorage = storeState( i, sess, sgmcmc, paramStorage )
        if ( i %% 100 == 0 ) {
            # If chain has diverged throw an error, this function also prints progress
            checkDivergence( sess, sgmcmc, i, options$verbose )
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

# Check for divergence of chain and print progress if verbose == TRUE
checkDivergence = function( sess, sgmcmc, iter, verbose ) {
    currentEstimate = sess$run( sgmcmc$estLogPost, feed_dict = dataFeed( 
            sgmcmc$data, sgmcmc$placeholders, sgmcmc$n ) )
    # If chain diverged throw an error
    if ( is.nan( currentEstimate ) ) {
        stop("Chain diverged, try decreasing stepsize")
    }
    if ( verbose ) {
        message( paste0( "Iteration: ", iter, "\t\tLog posterior estimate: ", currentEstimate ) )
    }
}
