#' Stochastic Gradient Nose Hoover Thermostat
#' 
#' Simulates from the posterior defined by the functions logLik and logPrior using
#'  stochastic gradient Nose Hoover Thermostat.
#'  The thermostat step needs a dot product to be calculated between two vectors.
#'  So when the algorithm uses parameters that are higher order than vectors 
#'  (e.g. matrices and tensors), the thermostat step uses a tensor contraction.
#'  Tensor contraction is otherwise known as the inner product between two tensors.
#'
#' @references \itemize{\item \href{http://people.ee.duke.edu/~lcarin/sgnht-4.pdf}{
#'  Ding, N., Fang, Y., Babbush, R., Chen, C., Skeel, R. D., and Neven, H. (2014). 
#'  Bayesian sampling using stochastic gradient thermostats. NIPS (pp. 3203-3211).}}
#'
#' @inheritParams sgld
#' @param a optional. Default 0.01. List of numeric values corresponding to SGNHT diffusion factors
#'  (see Algorithm 2 of the original paper). One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#'  Alternatively specify a single float to specify that value for all parameters.
#'
#' @return Returns list of arrays for each parameter containing the MCMC chain.
#'  Dimension of the form (nIters,paramDim1,paramDim2,...)
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Simulate from a Normal Distribution with uninformative, improper prior
#' dataset = list("x" = rnorm(1000))
#' params = list("theta" = 0)
#' logLik = function(params, dataset) { 
#'     distn = tf$distributions$Normal(params$theta, 1)
#'     return(tf$reduce_sum(distn$log_prob(dataset$x)))
#' }
#' stepsize = list("theta" = 5e-6)
#' output = sgnht(logLik, dataset, params, stepsize)
#' # For more examples see vignettes
#' }
sgnht = function( logLik, dataset, params, stepsize, logPrior = NULL, minibatchSize = 0.01, 
            a = 0.01, nIters = 10^4L, verbose = TRUE, seed = NULL ) {
    # Declare SGNHT object
    sgnht = sgnhtSetup( logLik, dataset, params, stepsize, logPrior, minibatchSize, a, seed )
    options = list( "nIters" = nIters, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgnht, params, options )
    return( paramStorage )
}

#' Create an sgnht object
#' 
#' Creates an sgnht (stochastic gradient Nose Hoover Thermostat) object which can be passed to 
#'  \code{\link{sgmcmcStep}} to simulate from 1 step of SGNHT for the posterior defined by 
#'  logLik and logPrior. This allows the user to code the loop themselves, as in many standard 
#'  TensorFlow procedures (such as optimization). Which means they do not need to store 
#'  the chain at each iteration. This is useful when the full chain needs a lot of memory.
#' 
#' @inheritParams sgnht
#'
#' @return The function returns an 'sgnht' object, which is used to pass the required information
#'  about the current model to the \code{\link{sgmcmcStep}} function. The function 
#'  \code{\link{sgmcmcStep}} runs one step of sgnht. The sgnht object has the following attributes:
#' \describe{
#' \item{params}{list of tf$Variables with the same names as the params list passed to
#'  \code{\link{sgnhtSetup}}. This is the object passed to the logLik and logPrior functions you
#'  declared to calculate the log posterior gradient estimate.}
#' \item{estLogPost}{a tensor that estimates the log posterior given the current 
#'  placeholders and params.}
#' \item{N}{dataset size.}
#' \item{data}{dataset as passed to \code{\link{sgnhtSetup}}.}
#' \item{n}{minibatchSize as passed to \code{\link{sgnhtSetup}}.}
#' \item{placeholders}{list of tf$placeholder objects with the same names as dataset
#'  used to feed minibatches of data to \code{\link{sgmcmcStep}}. This object
#'  gets fed to the dataset argument of the logLik and logPrior functions you declared.}
#' \item{stepsize}{list of stepsizes as passed to \code{\link{sgnhtSetup}}.}
#' \item{a}{list of a tuning parameters as passed to \code{\link{sgnhtSetup}}.}
#' \item{dynamics}{a list of TensorFlow steps that are evaluated by \code{\link{sgmcmcStep}}.}}
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
#'     distn = tf$distributions$Normal(params$theta, 1)
#'     return(tf$reduce_sum(distn$log_prob(dataset$x)))
#' }
#' stepsize = list("theta" = 1e-4)
#' sgnht = sgnhtSetup(logLik, dataset, params, stepsize)
#' nIters = 10^4L
#' # Initialize location estimate
#' locEstimate = 0
#' # Initialise TensorFlow session
#' sess = initSess(sgnht)
#' for ( i in 1:nIters ) {
#'     sgmcmcStep(sgnht, sess)
#'     locEstimate = locEstimate + 1 / nIters * getParams(sgnht, sess)$theta
#' }
#' # For more examples see vignettes
#' }
sgnhtSetup = function( logLik, dataset, params, stepsize, logPrior = NULL, minibatchSize = 0.01, 
            a = 0.01, seed = NULL ) {
    # Create generic sgmcmc object
    sgnht = createSGMCMC( logLik, logPrior, dataset, params, stepsize, minibatchSize, seed )
    # Get ranks for each parameter tensor, required for sgnht dynamics
    sgnht$ranks = getRanks( params )
    # Declare sgnht specific tuning constants, checking they're in list format
    sgnht$a = convertList( a, sgnht$params )
    # Declare object types
    class(sgnht) = c( "sgnht", "sgmcmc" )
    # Declare SGNHT dynamics
    sgnht$dynamics = declareDynamics( sgnht, seed )
    return( sgnht )
}
