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
#'  (list of tensorflow variables and placeholders respectively) as input. 
#'  It should return a tensorflow expression which defines the log likelihood of the model.
#' @param dataset list of R arrays which defines the datasets for the problem.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param params list of R arrays which define the starting point of each parameter.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param stepsize list of numeric values corresponding to the SGHMC eta stepsize terms,
#'  as defined in the paper given in the references. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#'  Alternatively specify a single float to specify that value for all parameters.
#' @param logPrior function which takes parameters (list of tensorflow variables) as input.
#'  The function should return a tensorflow tensor which defines the log prior of the model.
#'  Optional. Default uninformative improper prior.
#' @param minibatchSize either as proportion of dataset size or actual size, assumed float or int.
#'  Optional. Default 0.01.
#' @param alpha list of numeric values corresponding to the SGHMC alpha momentum terms,
#'  as defined in the paper given in the references. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#'  Alternatively specify a single float to specify that value for all parameters.
#'  Optional. Default 0.01 for all parameters.
#' @param L integer specifying the trajectory of the simulation, as defined in the main reference.
#'  Optional. Default 3.
#' @param nIters number of iterations of SGLD to perform, optional, assumed integer, default 10^4.
#' @param verbose whether to print algorithm progress or not, assumed BOOLEAN, default TRUE.
#'
#' @return List of arrays for each parameter containing the MCMC chain.
#'  Dimension of the form (nIters,paramDim1,paramDim2,...)
#'
#' @export
#'
sghmc = function( logLik, dataset, params, stepsize, logPrior = NULL, minibatchSize = 500, 
        alpha = 0.01, L = 3, nIters = 10^4, verbose = TRUE ) {
    # Setup SGHMC object
    sgmcmc = genSGHMC( logLik, logPrior, dataset, params, stepsize, alpha, L, minibatchSize )
    options = list( "nIters" = nIters, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgmcmc, params, options )
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
#'  (list of tensorflow variables and placeholders respectively) as input. 
#'  It should return a tensorflow expression which defines the log likelihood of the model.
#' @param dataset list of R arrays which defines the datasets for the problem.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param params list of R arrays which define the starting point of each parameter.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param stepsize list of numeric values corresponding to the SGHMC eta stepsize terms,
#'  as defined in the paper given in the references. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#'  Alternatively specify a single float to specify that value for all parameters.
#' @param optStepsize numeric, stepsize for optimization step to find MAP estimates of parameters.
#' @param logPrior function which takes parameters (list of tensorflow variables) as input.
#'  The function should return a tensorflow tensor which defines the log prior of the model.
#'  Optional. Default uninformative improper prior.
#' @param minibatchSize either as proportion of dataset size or actual size, assumed float or int.
#'  Optional. Default 0.01.
#' @param alpha list of numeric values corresponding to the SGHMC alpha momentum terms,
#'  as defined in the paper given in the references. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#'  Alternatively specify a single float to specify that value for all parameters.
#'  Optional. Default 0.01 for all parameters.
#' @param L integer specifying the trajectory of the simulation, as defined in the main reference.
#'  Optional. Default 3.
#' @param nIters integer, number of iterations of SGLD to perform, optional, default 10^4.
#' @param nItersOpt integer, number of iterations of initial optimization to perform, 
#'  optional, default 10^4.
#' @param verbose Boolean, whether to print algorithm progress or not, default TRUE.
#'
#' @return List of arrays for each parameter containing the MCMC chain.
#'  Dimension of the form (nIters,paramDim1,paramDim2,...)
#'
#' @export
#'
sghmccv = function( logLik, dataset, params, stepsize, optStepsize, logPrior = NULL, 
        minibatchSize = 500, alpha = 0.01, L = 3, nIters = 10^4, nItersOpt = 10^4, 
        verbose = TRUE ) {
    # Setup SGHMCCV object
    sgmcmcCV = genSGHMCCV( logLik, logPrior, dataset, params, stepsize, alpha, L, optStepsize, minibatchSize )
    options = list( "nIters" = nIters, "nItersOpt" = nItersOpt, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgmcmcCV, params, options )
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
genSGHMCCV = function( logLik, logPrior, dataset, params, stepsize, alpha, L, optStepsize, minibatchSize ) {
    # Create generic sgmcmcCV object
    sghmcCV = createSGMCMCCV( 
        logLik, logPrior, dataset, params, stepsize, optStepsize, minibatchSize )
    # Declare SGHMC specific tuning constants and check they're in list format
    sghmcCV$alpha = convertList( alpha, sghmcCV$params )
    sghmcCV$L = L
    # Declare object type
    class(sghmcCV) = c( "sghmc", "sgmcmcCV" )
    # Declare SGHMC dynamics
    sghmcCV$dynamics = declareDynamics( sghmcCV )
    return( sghmcCV )
}

# Declare the tensorflow steps needed for one step of SGHMC, input SGHMC object
# @param is an sghmc object
declareDynamics.sghmc = function( sghmc) {
    dynamics = list( "theta" = list(), "nu" = list(), "refresh" = list(), "grad" = list() )
    # Get the correct gradient estimate given the sgld object (i.e. standard sgld or sgldcv) 
    estLogPostGrads = getGradients( sghmc )
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
        dynamics$refresh[[pname]] = nu$assign( sqrt( stepsize ) * tf$random_normal( theta$get_shape() ) )
        dynamics$nu[[pname]] = nu$assign_add( stepsize*gradU - alpha*nu + 
                sqrt( 2 * stepsize * alpha ) * tf$random_normal( theta$get_shape() ) )
        dynamics$theta[[pname]] = theta$assign_add( nu )
    }
    return( dynamics )
}
