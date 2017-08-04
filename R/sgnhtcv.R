#' Stochastic Gradient Nose Hoover Thermostat with Control Variates
#' 
#' Simulates from the posterior defined by the functions logLik and logPrior using
#'  stochastic gradient Nose Hoover Thermostat with an improved gradient estimate 
#'  that is calculated using control variates.
#'  The thermostat step needs a dot product to be calculated between two vectors.
#'  So when the algorithm uses parameters that are higher order than vectors 
#'  (e.g. matrices and tensors), the thermostat step uses a tensor contraction.
#'  Tensor contraction is otherwise known as the inner product between two tensors.
#'
#' @references \itemize{
#'  \item \href{https://arxiv.org/pdf/1706.05439.pdf}{
#'  Baker, J., Fearnhead, P., Fox, E. B., and Nemeth, C. (2017).
#'  Control variates for stochastic gradient MCMC. ArXiv preprint arXiv:1706.05439.}
#'  \item \href{http://people.ee.duke.edu/~lcarin/sgnht-4.pdf}{
#'  Ding, N., Fang, Y., Babbush, R., Chen, C., Skeel, R. D., and Neven, H. (2014). 
#'  Bayesian sampling using stochastic gradient thermostats. NIPS (pp. 3203-3211).}}
#'
#' @inheritParams sgnht
#' @param optStepsize numeric value specifying the stepsize for the optimization 
#'  to find MAP estimates of parameters. The TensorFlow GradientDescentOptimizer is used.
#' @param nItersOpt optional. Default 10^4L. 
#'  Integer specifying number of iterations of initial optimization to perform.
#'
#' @return Returns list of arrays for each parameter containing the MCMC chain.
#'  Dimension of the form (nIters,paramDim1,paramDim2,...). Names are the same as the params list.
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
#' output = sgnhtcv(logLik, dataset, params, stepsize, optStepsize)
#' }
sgnhtcv = function( logLik, dataset, params, stepsize, optStepsize, logPrior = NULL, 
            minibatchSize = 0.01, a = 0.01, nIters = 10^4L, nItersOpt = 10^4L, 
            verbose = TRUE, seed = NULL ) {
    # Declare SGNHTCV object
    sgnhtcv = sgnhtcvSetup( logLik, dataset, params, stepsize, optStepsize, logPrior, minibatchSize, 
            a, nItersOpt, verbose, seed )
    options = list( "nIters" = nIters, "nItersOpt" = nItersOpt, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgnhtcv, params, options )
    return( paramStorage )
}

#' Create an sgnhtcv object
#' 
#' Creates an sgnhtcv (stochastic gradient Nose Hoover thermostat with Control Variates) object 
#'  which can be passed to \code{\link{sgmcmcStep}} to simulate from 1 step of sgnht, using a 
#'  gradient estimate with control variates for the posterior defined by logLik and logPrior.
#'  This allows the user to code the loop themselves, as in many standard 
#'  TensorFlow procedures (such as optimization). Which means they do not need to store 
#'  the chain at each iteration. This is useful when the full chain needs a lot of memory.
#' 
#' @inheritParams sgnhtcv
#'
#' @return The function returns an 'sgnhtcv' object, a type of sgmcmc object. 
#'  Which is used to pass the required information about the current model to the 
#'  \code{\link{sgmcmcStep}} function. The function \code{\link{sgmcmcStep}} runs one 
#'  step of sgnht with a gradient estimate that uses control variates. 
#'  Attributes of the sgnhtcv object you'll probably find most useful are:
#' \describe{
#' \item{params}{list of tf$Variables with the same names as the params list passed to
#'  \code{\link{sgnhtcvSetup}}. This is the object passed to the logLik and logPrior functions you
#'  declared to calculate the log posterior gradient estimate.}
#' \item{paramsOpt}{list of tf$Variables with the same names as the \code{params} list passed to
#'  \code{\link{sgnhtcvSetup}}. These variables are used to initially find MAP estimates
#'  and then store these optimal parameter estimates.}
#' \item{estLogPost}{a tensor relying on \code{params} and \code{placeholders}. 
#'  This tensor estimates the log posterior given the current placeholders and params.}
#' \item{logPostOptGrad}{list of \code{tf$Variables} with same names as \code{params}, this stores
#'  the full log posterior gradient at each MAP estimate after the initial optimization step.}}
#'  Other attributes of the object are as follows:
#' \describe{
#' \item{N}{dataset size.}
#' \item{data}{dataset as passed to \code{\link{sgnhtcvSetup}}.}
#' \item{n}{minibatchSize as passed to \code{\link{sgnhtcvSetup}}.}
#' \item{placeholders}{list of tf$placeholder objects with the same names as dataset
#'  used to feed minibatches of data to \code{\link{sgmcmcStep}}. These are also the objects
#'  that gets fed to the dataset argument of the logLik and logPrior functions you declared.}
#' \item{stepsize}{list of stepsizes as passed to \code{\link{sgnhtcvSetup}}}
#' \item{alpha}{list of alpha tuning parameters as passed to \code{\link{sgnhtSetup}}.}
#' \item{L}{integer trajectory parameter as passed to \code{\link{sgnhtSetup}}.}
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
#' sgnhtcv = sgnhtcvSetup(logLik, dataset, params, stepsize, optStepsize)
#' nIters = 10^4L
#' # Initialize location estimate
#' locEstimate = 0
#' # Initialise TensorFlow session
#' sess = initSess(sgnhtcv)
#' for ( i in 1:nIters ) {
#'     sgmcmcStep(sgnhtcv, sess)
#'     locEstimate = locEstimate + 1 / nIters * getParams(sgnhtcv, sess)$theta
#' }
#' # For more examples see vignettes
#' }
sgnhtcvSetup = function( logLik, dataset, params, stepsize, optStepsize, logPrior = NULL, 
            minibatchSize = 0.01, a = 0.01, nItersOpt = 10^4L, verbose = TRUE, seed = NULL ) {
    # Create generic sgmcmcCV object
    sgnhtcv = createSGMCMCCV( logLik, logPrior, dataset, params, stepsize, optStepsize, 
            minibatchSize, nItersOpt, seed )
    # Get ranks for each parameter tensor, required for sgnht dynamics
    sgnhtcv$ranks = getRanks( params )
    # Declare sgnht specific tuning constants, checking they're in list format
    sgnhtcv$a = convertList( a, sgnhtcv$params )
    # Declare object types
    class(sgnhtcv) = c( "sgnht", "sgmcmccv", "sgmcmc" )
    # Declare SGNHT dynamics
    sgnhtcv$dynamics = declareDynamics( sgnhtcv, seed )
    return( sgnhtcv )
}
