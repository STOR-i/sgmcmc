#' Get current parameter values
#' 
#' Return the current parameter values as a list of R arrays (converted from the tensorflow tensors).
#' 
#' @param sgmcmc a stochastic gradient MCMC object returned by *Setup such as 
#'  \code{\link{sgldSetup}}, \code{\link{sgldcvSetup}} etc.
#' @param sess a TensorFlow session created using \code{\link{initSess}}
#'
#' @return Returns a list with the same names as \code{params}, with \code{R} arrays of the current
#'  parameter values
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
getParams = function( sgmcmc, sess ) {
    return( sess$run( sgmcmc$params ) )
}

# Initialise storage array, this will hold the MCMC chain and is the output returned by e.g. runSGLD
initStorage = function( paramsRaw, n_iters ) {
    paramStorage = list()
    for ( pname in names( paramsRaw ) ) {
        shapeCurrent = getShape( paramsRaw[[pname]] )
        # Add new dimension to parameter's current shape which holds each iteration
        shapeCurrent = c( n_iters, shapeCurrent )
        paramStorage[[pname]] = array( 0, dim = shapeCurrent )
    }
    return( paramStorage )
}

# Store the current parameter values
storeState = function( iter, sess, sgmcmc, storage ) {
    for ( pname in names( sgmcmc$params ) ) {
        paramCurrent = sgmcmc$params[[pname]]$eval( session = sess )
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
