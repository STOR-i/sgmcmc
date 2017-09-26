#' Stochastic Gradient Langevin Dynamics with Control Variates
#' 
#' Simulates from the posterior defined by the functions logLik and logPrior using
#'  stochastic gradient Langevin Dynamics with an improved gradient estimate using Control Variates.
#'  The function uses TensorFlow, so needs TensorFlow for python installed.
#'
#' @references \itemize{
#'  \item \href{https://arxiv.org/pdf/1706.05439.pdf}{
#'  Baker, J., Fearnhead, P., Fox, E. B., and Nemeth, C. (2017).
#'  Control variates for stochastic gradient MCMC. ArXiv preprint arXiv:1706.05439.}
#'  \item \href{http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf}{
#'  Welling, M., and Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. 
#'  ICML (pp. 681-688).} }
#'
#' @inheritParams sgld
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
#' stepsize = list("theta" = 1e-4)
#' optStepsize = 1e-1
#' output = sgldcv(logLik, dataset, params, stepsize, optStepsize)
#' }
sgldcv = function( logLik, dataset, params, stepsize, optStepsize, logPrior = NULL,
        minibatchSize = 0.01, nIters = 10^4L, nItersOpt = 10^4L, verbose = TRUE, seed = NULL ) {
    # Setup SGLDCV object
    sgldcv = sgldcvSetup( logLik, dataset, params, stepsize, optStepsize, logPrior, minibatchSize,
            nItersOpt, verbose, seed )
    options = list( "nIters" = nIters, "nItersOpt" = nItersOpt, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgldcv, params, options )
    return( paramStorage )
}


#' Create an sgldcv object
#' 
#' Creates an sgldcv (stochastic gradient Langevin Dynamics with Control Variates) object 
#'  which can be passed to \code{\link{sgmcmcStep}} to simulate from 1 step of sgld, using a 
#'  gradient estimate with control variates for the posterior defined by logLik and logPrior.
#'  This allows the user to code the loop themselves, as in many standard 
#'  TensorFlow procedures (such as optimization). Which means they do not need to store 
#'  the chain at each iteration. This is useful when the full chain needs a lot of memory.
#' 
#' @inheritParams sgldcv
#'
#' @return The function returns an 'sgldcv' object, a type of sgmcmc object. 
#'  Which is used to pass the required information about the current model to the 
#'  \code{\link{sgmcmcStep}} function. The function \code{\link{sgmcmcStep}} runs one 
#'  step of sgld with a gradient estimate that uses control variates. 
#'  Attributes of the sgldcv object you'll probably find most useful are:
#' \describe{
#' \item{params}{list of tf$Variables with the same names as the params list passed to
#'  \code{\link{sgldcvSetup}}. This is the object passed to the logLik and logPrior functions you
#'  declared to calculate the log posterior gradient estimate.}
#' \item{paramsOpt}{list of tf$Variables with the same names as the \code{params} list passed to
#'  \code{\link{sgldcvSetup}}. These variables are used to initially find MAP estimates
#'  and then store these optimal parameter estimates.}
#' \item{estLogPost}{a tensor that estimates the log posterior given the current 
#'  placeholders and params.}
#' \item{logPostOptGrad}{list of \code{tf$Variables} with same names as \code{params}, this stores
#'  the full log posterior gradient at each MAP estimate after the initial optimization step.}}
#'  Other attributes of the object are as follows:
#' \describe{
#' \item{N}{dataset size.}
#' \item{data}{dataset as passed to \code{\link{sgldcvSetup}}.}
#' \item{n}{minibatchSize as passed to \code{\link{sgldcvSetup}}.}
#' \item{placeholders}{list of tf$placeholder objects with the same names as dataset
#'  used to feed minibatches of data to \code{\link{sgmcmcStep}}. These are also the objects
#'  that gets fed to the dataset argument of the logLik and logPrior functions you declared.}
#' \item{stepsize}{list of stepsizes as passed to \code{\link{sgldcvSetup}}}
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
#' sgldcv = sgldcvSetup(logLik, dataset, params, stepsize, optStepsize)
#' nIters = 10^4L
#' # Initialize location estimate
#' locEstimate = 0
#' # Initialise TensorFlow session
#' sess = initSess(sgldcv)
#' for ( i in 1:nIters ) {
#'     sgmcmcStep(sgldcv, sess)
#'     locEstimate = locEstimate + 1 / nIters * getParams(sgldcv, sess)$theta
#' }
#' # For more examples see vignettes
#' }
sgldcvSetup = function( logLik, dataset, params, stepsize, optStepsize, logPrior = NULL, 
            minibatchSize = 0.01, nItersOpt = 10^4L, verbose = TRUE, seed = NULL ) {
    # Create generic sgmcmccv object, no extra tuning constants need to be added for sgld
    sgldcv = createSGMCMCCV( logLik, logPrior, dataset, params, stepsize, optStepsize, 
            minibatchSize, nItersOpt, seed )
    class(sgldcv) = c( "sgld", "sgmcmccv", "sgmcmc" )
    # Declare SGLD dynamics
    sgldcv$dynamics = declareDynamics( sgldcv, seed )
    return( sgldcv )
}
