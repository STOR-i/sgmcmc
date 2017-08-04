#' Stochastic Gradient Hamiltonian Monte Carlo with Control Variates
#' 
#' Simulates from the posterior defined by the functions logLik and logPrior using 
#'  stochastic gradient Hamiltonian Monte Carlo with an improved gradient estimate 
#'  that is calculated using control variates. 
#'  Currently we use the approximation \eqn{\hat \beta = 0}, 
#'  as used in the simulations by the original reference. 
#'  This will be changed in future implementations.
#'
#' @references \itemize{
#'  \item \href{https://arxiv.org/pdf/1706.05439.pdf}{
#'  Baker, J., Fearnhead, P., Fox, E. B., and Nemeth, C. (2017).
#'  Control variates for stochastic gradient MCMC. ArXiv preprint arXiv:1706.05439.}
#'  \item \href{https://arxiv.org/pdf/1402.4102v2.pdf}{
#'  Chen, T., Fox, E. B., and Guestrin, C. (2014). Stochastic gradient Hamiltonian Monte Carlo. 
#'  In ICML (pp. 1683-1691).}}
#'
#' @inheritParams sghmc
#' @param optStepsize numeric value specifying the stepsize for the optimization 
#'  to find MAP estimates of parameters. The TensorFlow GradientDescentOptimizer is used.
#' @param nItersOpt optional. Default 10^4L. 
#'  Integer specifying number of iterations of initial optimization to perform.
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
#' stepsize = list("theta" = 1e-5)
#' optStepsize = 1e-1
#' output = sghmccv(logLik, dataset, params, stepsize, optStepsize)
#' }
sghmccv = function( logLik, dataset, params, stepsize, optStepsize, logPrior = NULL, 
            minibatchSize = 0.01, alpha = 0.01, L = 5L, nIters = 10^4L, nItersOpt = 10^4L, 
            verbose = TRUE, seed = NULL ) {
    # Setup SGHMCCV object
    sghmccv = sghmccvSetup( logLik, dataset, params, stepsize, optStepsize, logPrior, minibatchSize, 
            alpha, L, nItersOpt, verbose, seed )
    options = list( "nIters" = nIters, "nItersOpt" = nItersOpt, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sghmccv, params, options )
    return( paramStorage )
}

#' Create an sghmccv object
#' 
#' Creates an sghmccv (stochastic gradient Hamiltonian Monte Carlo with Control Variates) object 
#'  which can be passed to \code{\link{sgmcmcStep}} to simulate from 1 step of sghmc, using a 
#'  gradient estimate with control variates for the posterior defined by logLik and logPrior.
#'  This allows the user to code the loop themselves, as in many standard 
#'  TensorFlow procedures (such as optimization). Which means they do not need to store 
#'  the chain at each iteration. This is useful when the full chain needs a lot of memory.
#' 
#' @inheritParams sghmccv
#'
#' @return The function returns an 'sghmccv' object, a type of sgmcmc object. 
#'  Which is used to pass the required information about the current model to the 
#'  \code{\link{sgmcmcStep}} function. The function \code{\link{sgmcmcStep}} runs one 
#'  step of sghmc with a gradient estimate that uses control variates. 
#'  Attributes of the sghmccv object you'll probably find most useful are:
#' \describe{
#' \item{params}{list of tf$Variables with the same names as the params list passed to
#'  \code{\link{sghmccvSetup}}. This is the object passed to the logLik and logPrior functions you
#'  declared to calculate the log posterior gradient estimate.}
#' \item{paramsOpt}{list of tf$Variables with the same names as the \code{params} list passed to
#'  \code{\link{sghmccvSetup}}. These variables are used to initially find MAP estimates
#'  and then store these optimal parameter estimates.}
#' \item{estLogPost}{a tensor that estimates the log posterior given the current 
#'  placeholders and params.}
#' \item{logPostOptGrad}{list of \code{tf$Variables} with same names as \code{params}, this stores
#'  the full log posterior gradient at each MAP estimate after the initial optimization step.}}
#'  Other attributes of the object are as follows:
#' \describe{
#' \item{N}{dataset size.}
#' \item{data}{dataset as passed to \code{\link{sghmccvSetup}}.}
#' \item{n}{minibatchSize as passed to \code{\link{sghmccvSetup}}.}
#' \item{placeholders}{list of tf$placeholder objects with the same names as dataset
#'  used to feed minibatches of data to \code{\link{sgmcmcStep}}. These are also the objects
#'  that gets fed to the dataset argument of the logLik and logPrior functions you declared.}
#' \item{stepsize}{list of stepsizes as passed to \code{\link{sghmccvSetup}}}
#' \item{alpha}{list of alpha tuning parameters as passed to \code{\link{sghmcSetup}}.}
#' \item{L}{integer trajectory parameter as passed to \code{\link{sghmcSetup}}.}
#' \item{dynamics}{a list of TensorFlow steps that are evaluated by \code{\link{sgmcmcStep}}.}
#' \item{estLogPostOpt}{a TensorFlow tensor relying on \code{paramsOpt} and \code{placeholders} which
#'  estimates the log posterior at the optimal parameters. Used in the initial optimization step.}
#' \item{fullLogPostOpt}{a TensorFlow tensor used in the calculation of the full log posterior
#'  gradient at the MAP estimates.}
#' \item{optimizer}{a TensorFlow optimizer object used to find the initial MAP estimates.}}
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
#' optStepsize = 1e-1
#' sghmccv = sghmccvSetup(logLik, dataset, params, stepsize, optStepsize)
#' nIters = 10^4L
#' # Initialize location estimate
#' locEstimate = 0
#' # Initialise TensorFlow session
#' sess = initSess(sghmccv)
#' for ( i in 1:nIters ) {
#'     sgmcmcStep(sghmccv, sess)
#'     locEstimate = locEstimate + 1 / nIters * getParams(sghmccv, sess)$theta
#' }
#' # For more examples see vignettes
#' }
sghmccvSetup = function( logLik, dataset, params, stepsize, optStepsize, logPrior = NULL, 
            minibatchSize = 0.01, alpha = 0.01, L = 5L, nItersOpt = 10^4L, 
            verbose = TRUE, seed = NULL ) {
    # Create generic sgmcmcCV object
    sghmccv = createSGMCMCCV( logLik, logPrior, dataset, params, stepsize, optStepsize, 
            minibatchSize, nItersOpt, seed )
    # Declare SGHMC specific tuning constants and check they're in list format
    sghmccv$alpha = convertList( alpha, sghmccv$params )
    sghmccv$L = L
    # Declare object type
    class(sghmccv) = c( "sghmc", "sgmcmccv", "sgmcmc" )
    # Declare SGHMC dynamics
    sghmccv$dynamics = declareDynamics( sghmccv, seed )
    return( sghmccv )
}
