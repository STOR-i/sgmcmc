#' Stochastic Gradient Langevin Dynamics
#' 
#' Simulates from the posterior defined by the functions logLik and logPrior using
#'  stochastic gradient Langevin Dynamics. The function uses TensorFlow, so needs
#'  TensorFlow for python installed.
#'
#' @references \itemize{\item \href{http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf}{
#'  Welling, M., and Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. 
#'  ICML (pp. 681-688).}}
#'
#' @param logLik function which takes parameters and dataset 
#'  (list of TensorFlow variables and placeholders respectively) as input. 
#'  It should return a TensorFlow expression which defines the log likelihood of the model.
#' @param dataset list of numeric R arrays which defines the datasets for the problem.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param params list of numeric R arrays which define the starting point of each parameter.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param stepsize list of numeric values corresponding to the SGLD stepsizes for each parameter
#'  The names in the list should correspond to those in params.
#'  Alternatively specify a single numeric value to use that stepsize for all parameters.
#' @param logPrior optional. Default uninformative improper prior.
#'  Function which takes parameters (list of TensorFlow variables) as input.
#'  The function should return a TensorFlow tensor which defines the log prior of the model.
#' @param minibatchSize optional. Default 0.01.
#'  Numeric or integer value that specifies amount of dataset to use at each iteration 
#'  either as proportion of dataset size (if between 0 and 1) or actual magnitude (if an integer).
#' @param nIters optional. Default 10^4L. Integer specifying number of iterations to perform.
#' @param verbose optional. Default TRUE. Boolean specifying whether to print algorithm progress
#' @param seed optional. Default NULL. Numeric seed for random number generation. The default
#'  does not declare a seed for the TensorFlow session.
#'
#' @return Returns list of arrays for each parameter containing the MCMC chain.
#'  Dimension of the form (nIters,paramDim1,paramDim2,...)
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Simulate from a Normal Distribution with uninformative prior
#' dataset = list("x" = rnorm(1000))
#' params = list("theta" = 0)
#' logLik = function(params, dataset) { 
#'     distn = tf$contrib$distributions$Normal(params$theta, 1)
#'     return(tf$reduce_sum(distn$log_prob(dataset$x)))
#' }
#' stepsize = list("theta" = 1e-4)
#' output = sgld(logLik, dataset, params, stepsize)
#' # For more examples see vignettes
#' }
sgld = function( logLik, dataset, params, stepsize, logPrior = NULL, minibatchSize = 0.01, 
            nIters = 10^4L, verbose = TRUE, seed = NULL ) {
    # Create SGLD object
    sgld = sgldSetup( logLik, dataset, params, stepsize, logPrior, minibatchSize, seed )
    options = list( "nIters" = nIters, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgld, params, options )
    return( paramStorage )
}


#' Create an sgld object
#' 
#' Creates an sgld (stochastic gradient Langevin dynamics) object which can be passed to 
#'  \code{\link{sgmcmcStep}} to simulate from 1 step of SGLD for the posterior defined by logLik 
#'  and logPrior. This allows the user to code the loop themselves, as in many standard 
#'  TensorFlow procedures (such as optimization). Which means they do not need to store 
#'  the chain at each iteration. This is useful when the full chain needs a lot of memory.
#' 
#' @inheritParams sgld
#'
#' @return The function returns an 'sgld' object, which is used to pass the required information
#'  about the current model to the \code{\link{sgmcmcStep}} function. The function 
#'  \code{\link{sgmcmcStep}} runs one step of sgld. The sgld object has the following attributes:
#' \describe{
#' \item{params}{list of tf$Variables with the same names as the params list passed to
#'  \code{\link{sgldSetup}}. This is the object passed to the logLik and logPrior functions you
#'  declared to calculate the log posterior gradient estimate.}
#' \item{estLogPost}{a tensor that estimates the log posterior given the current 
#'  placeholders and params (the placeholders holds the minibatches of data).}
#' \item{N}{dataset size.}
#' \item{data}{dataset as passed to \code{\link{sgldSetup}}.}
#' \item{n}{minibatchSize as passed to \code{\link{sgldSetup}}.}
#' \item{placeholders}{list of tf$placeholder objects with the same names as dataset
#'  used to feed minibatches of data to \code{\link{sgmcmcStep}}. These are the objects
#'  that get fed to the dataset argument of the logLik and logPrior functions you declared.}
#' \item{stepsize}{list of stepsizes as passed to \code{\link{sgldSetup}}.}
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
sgldSetup = function( logLik, dataset, params, stepsize, logPrior = NULL, minibatchSize = 0.01,
            seed = NULL ) {
    # Create generic sgmcmc object, no extra tuning constants need to be added for sgld
    sgld = createSGMCMC( logLik, logPrior, dataset, params, stepsize, minibatchSize, seed )
    # Declare object type
    class( sgld ) = c( "sgld", "sgmcmc" )
    # Declare SGLD dynamics
    sgld$dynamics = declareDynamics( sgld, seed )
    return( sgld )
}
