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
#' @param logLik function which takes parameters and dataset 
#'  (list of TensorFlow variables and placeholders respectively) as input. 
#'  It should return a TensorFlow expression which defines the log likelihood of the model.
#' @param dataset list of numeric R arrays which defines the datasets for the problem.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param params list of numeric R arrays which define the starting point of each parameter.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param stepsize list of numeric values corresponding to the SGHMC stepsizes for each parameter
#'  (\eqn{\eta} in the original paper). The names in the list should correspond to those in params.
#'  Alternatively specify a single numeric value to use that stepsize for all parameters.
#' @param logPrior optional. Default uninformative improper prior.
#'  Function which takes parameters (list of TensorFlow variables) as input.
#'  The function should return a TensorFlow tensor which defines the log prior of the model.
#' @param minibatchSize optional. Default 0.01.
#'  Numeric or integer value that specifies amount of dataset to use at each iteration 
#'  either as proportion of dataset size (if between 0 and 1) or actual magnitude (if an integer).
#' @param alpha optional. Default 0.01. 
#'  List of numeric values corresponding to the SGHMC momentum tuning constants
#'  (\eqn{\alpha} in the original paper). One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#'  Alternatively specify a single float to specify that value for all parameters.
#' @param L optional. Default 5L. Integer specifying the trajectory parameter of the simulation, 
#'  as defined in the main reference.
#' @param nIters optional. Default 10^4L. Integer specifying number of iterations to perform.
#' @param verbose optional. Default TRUE. Boolean specifying whether to print algorithm progress
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
#' }
sghmc = function( logLik, dataset, params, stepsize, logPrior = NULL, minibatchSize = 0.01, 
            alpha = 0.01, L = 5L, nIters = 10^4L, verbose = TRUE ) {
    # Setup SGHMC object
    sghmc = genSGHMC( logLik, logPrior, dataset, params, stepsize, alpha, L, minibatchSize )
    options = list( "nIters" = nIters, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sghmc, params, options )
    return( paramStorage )
}

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
#' @param logLik function which takes parameters and dataset 
#'  (list of TensorFlow variables and placeholders respectively) as input. 
#'  It should return a TensorFlow expression which defines the log likelihood of the model.
#' @param dataset list of numeric R arrays which defines the datasets for the problem.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param params list of numeric R arrays which define the starting point of each parameter.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param stepsize list of numeric values corresponding to the SGHMC stepsizes for each parameter
#'  (\eqn{\eta} in the original paper). The names in the list should correspond to those in params.
#'  Alternatively specify a single numeric value to use that stepsize for all parameters.
#' @param optStepsize numeric value specifying the stepsize for the optimization 
#'  to find MAP estimates of parameters. The TensorFlow AdamOptimizer is used.
#' @param logPrior optional. Default uninformative improper prior.
#'  Function which takes parameters (list of TensorFlow variables) as input.
#'  The function should return a TensorFlow tensor which defines the log prior of the model.
#' @param minibatchSize optional. Default 0.01.
#'  Numeric or integer value that specifies amount of dataset to use at each iteration 
#'  either as proportion of dataset size (if between 0 and 1) or actual magnitude (if an integer).
#' @param alpha optional. Default 0.01. 
#'  List of numeric values corresponding to the SGHMC momentum tuning constants
#'  (\eqn{\alpha} in the original paper). One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#'  Alternatively specify a single float to specify that value for all parameters.
#' @param L optional. Default 5L. Integer specifying the trajectory parameter of the simulation, 
#'  as defined in the main reference.
#' @param nIters optional. Default 10^4L. Integer specifying number of iterations to perform.
#' @param nItersOpt optional. Default 10^4L. 
#'  Integer specifying number of iterations of initial optimization to perform.
#' @param verbose optional. Default TRUE. Boolean specifying whether to print algorithm progress.
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
            verbose = TRUE ) {
    # Setup SGHMCCV object
    sghmccv = genSGHMCCV( logLik, logPrior, dataset, params, stepsize, alpha, L, optStepsize, minibatchSize )
    options = list( "nIters" = nIters, "nItersOpt" = nItersOpt, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sghmccv, params, options )
    return( paramStorage )
}

# Create Stochastic Gradient Hamiltonian Monte Carlo Object
# 
# Creates a stochastic gradient Langevin Dynamics (SGHMC) object which can be passed to mcmcStep
#  to simulate from 1 step of SGHMC for the posterior defined by logLik and logPrior.
genSGHMC = function( logLik, logPrior, dataset, params, stepsize, alpha, L, minibatchSize ) {
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

# Create Stochastic Gradient Hamiltonian Monte Carlo with Control Variates Object
# 
# Creates a stochastic gradient Hamiltonian Monte Carlo with Control Variates (SGHMCCV) object 
#  which can be passed to optUpdate and mcmcStep functions to simulate from SGHMCCV
#  for the posterior defined by logLik and logPrior.
genSGHMCCV = function( logLik, logPrior, dataset, params, stepsize, alpha, L, 
            optStepsize, minibatchSize ) {
    # Create generic sgmcmcCV object
    sghmccv = createSGMCMCCV( 
            logLik, logPrior, dataset, params, stepsize, optStepsize, minibatchSize )
    # Declare SGHMC specific tuning constants and check they're in list format
    sghmccv$alpha = convertList( alpha, sghmccv$params )
    sghmccv$L = L
    # Declare object type
    class(sghmccv) = c( "sghmc", "sgmcmccv" )
    # Declare SGHMC dynamics
    sghmccv$dynamics = declareDynamics( sghmccv )
    return( sghmccv )
}

# Declare the TensorFlow steps needed for one step of SGHMC, input SGHMC object
# @param is an sghmc object
declareDynamics.sghmc = function( sghmc) {
    dynamics = list( "theta" = list(), "nu" = list(), "refresh" = list(), "grad" = list() )
    # Get the correct gradient estimate given the sgld object (i.e. standard sgld or sgldcv) 
    estLogPostGrads = getGradients( sghmc )
    # Loop over each parameter in params
    for ( pname in names( sghmc$params ) ) {
        dynamics$grad[[pname]] = tf$gradients( sghmc$estLogPost, sghmc$params[[pname]] )[[1]]
        # Declare tuning constants
        stepsize = sghmc$stepsize[[pname]]
        alpha = sghmc$alpha[[pname]]
        # Declare parameters
        theta = sghmc$params[[pname]]
        nu = tf$Variable( sqrt( stepsize ) * tf$random_normal( theta$get_shape() ) )
        # Declare dynamics
        gradU = estLogPostGrads[[pname]]
        dynamics$refresh[[pname]] = nu$assign( sqrt( stepsize ) * tf$random_normal( 
                theta$get_shape() ) )
        dynamics$nu[[pname]] = nu$assign_add( stepsize*gradU - alpha*nu + 
                sqrt( 2 * stepsize * alpha ) * tf$random_normal( theta$get_shape() ) )
        dynamics$theta[[pname]] = theta$assign_add( nu )
    }
    return( dynamics )
}
