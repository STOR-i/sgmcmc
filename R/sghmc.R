#' Stochastic Gradient Hamiltonian Monte Carlo
#' 
#' Simulates from the posterior defined by the functions logLik and logPrior using
#'  stochastic gradient Hamiltonian Monte Carlo. The function uses TensorFlow, so needs
#'  Tensorflow for python installed. Currently we use the approximation \eqn{\hat \beta = 0},
#'  as used in the simulations by the original reference. 
#'  This will be changed in future implementations.
#'
#' @references \itemize{\item \href{https://arxiv.org/pdf/1402.4102v2.pdf}{
#'  Chen, T., Fox, E. B., and Guestrin, C. (2014). Stochastic gradient Hamiltonian Monte Carlo. 
#'  In ICML (pp. 1683-1691).}}
#'
#' @inheritParams sgld
#' @param alpha optional. Default 0.01. 
#'  List of numeric values corresponding to the SGHMC momentum tuning constants
#'  (\eqn{\alpha} in the original paper). One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#'  Alternatively specify a single float to specify that value for all parameters.
#' @param L optional. Default 5L. Integer specifying the trajectory parameter of the simulation, 
#'  as defined in the main reference.
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
#'     distn = tf$contrib$distributions$Normal(params$theta, 1)
#'     return(tf$reduce_sum(distn$log_prob(dataset$x)))
#' }
#' stepsize = list("theta" = 1e-5)
#' output = sghmc(logLik, dataset, params, stepsize)
#' # For more examples see vignettes
#' }
sghmc = function( logLik, dataset, params, stepsize, logPrior = NULL, minibatchSize = 0.01, 
            alpha = 0.01, L = 5L, nIters = 10^4L, verbose = TRUE ) {
    # Setup SGHMC object
    sghmc = sghmcSetup( logLik, dataset, params, stepsize, logPrior, minibatchSize, alpha, L )
    options = list( "nIters" = nIters, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sghmc, params, options )
    return( paramStorage )
}

#' Create an sghmc object
#' 
#' Creates an sghmc (stochastic gradient Hamiltonian Monte Carlo) object which can be passed to 
#'  \code{\link{sgmcmcStep}} to simulate from 1 step of SGLD for the posterior defined by logLik
#'  and logPrior. This allows the user to code the loop themselves, as in many standard 
#'  TensorFlow procedures (such as optimization). Which means they do not need to store 
#'  the chain at each iteration. This is useful when the full chain needs a lot of memory.
#' 
#' @inheritParams sghmc
#'
#' @return The function returns an 'sghmc' object, which is used to pass the required information
#'  about the current model to the \code{\link{sgmcmcStep}} function. The function 
#'  \code{\link{sgmcmcStep}} runs one step of sghmc. The sghmc object has the following attributes:
#' \describe{
#' \item{params}{list of tf$Variables with the same names as the params list passed to
#'  \code{\link{sghmcSetup}}. This is the object passed to the logLik and logPrior functions you
#'  declared to calculate the log posterior gradient estimate.}
#' \item{estLogPost}{a tensor that estimates the log posterior given the current 
#'  placeholders and params.}
#' \item{N}{dataset size.}
#' \item{data}{dataset as passed to \code{\link{sghmcSetup}}.}
#' \item{n}{minibatchSize as passed to \code{\link{sghmcSetup}}.}
#' \item{placeholders}{list of tf$placeholder objects with the same names as dataset
#'  used to feed minibatches of data to \code{\link{sgmcmcStep}}. These are also the objects
#'  that gets fed to the dataset argument of the logLik and logPrior functions you declared.}
#' \item{stepsize}{list of stepsizes as passed to \code{\link{sghmcSetup}}.}
#' \item{alpha}{list of alpha tuning parameters as passed to \code{\link{sghmcSetup}}.}
#' \item{L}{integer trajectory parameter as passed to \code{\link{sghmcSetup}}.}
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
#' sghmc = sghmcSetup(logLik, dataset, params, stepsize)
#' nIters = 10^4L
#' # Initialize location estimate
#' locEstimate = 0
#' # Initialise TensorFlow session
#' sess = initSess(sghmc)
#' for ( i in 1:nIters ) {
#'     sgmcmcStep(sghmc, sess)
#'     locEstimate = locEstimate + 1 / nIters * getParams(sghmc, sess)$theta
#' }
#' # For more examples see vignettes
#' }
sghmcSetup = function( logLik, dataset, params, stepsize, logPrior = NULL, minibatchSize = 0.01, 
            alpha = 0.01, L = 5L ) {
    # Create generic sgmcmc object
    sghmc = createSGMCMC( logLik, logPrior, dataset, params, stepsize, minibatchSize )
    # Add SGHMC specific tuning constants and check they're in list format
    sghmc$alpha = convertList( alpha, sghmc$params )
    sghmc$L = L
    # Declare object type
    class( sghmc ) = c( "sghmc", "sgmcmc" )
    # Declare SGHMC dynamics
    sghmc$dynamics = declareDynamics( sghmc )
    return( sghmc )
}
