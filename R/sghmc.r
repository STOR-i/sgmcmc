library(tensorflow)
source("setup.r")
source("update.r")
source("controlVariates.r")
source("storage.r")

#' Stochastic Gradient Hamiltonian Monte Carlo
#' 
#' Simulates from the posterior defined by the functions logLik & logPrior using
#'  stochastic gradient Hamiltonian Monte Carlo. The function uses TensorFlow, so needs
#'  Tensorflow for python installed.
#'
#'  @references Chen, T., Fox, E. B., & Guestrin, C. (2014). 
#'      stochastic gradient Hamiltonian Monte Carlo. In ICML (pp. 1683-1691).
#'
#' @param logLik function which takes parameters and data 
#'  (list of tensorflow variables and placeholders respectively) as input. 
#'  It should return a tensorflow expression which defines the log likelihood of the model.
#' @param logPrior function which takes parameters and data 
#'  (list of tensorflow variables and placeholders respectively) as input. 
#'  The function should return a tensorflow tensor which defines the log prior of the model.
#' @param data list of R arrays which defines the datasets for the problem.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param params list of R arrays which define the starting point of each parameter.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param eta list of numeric values corresponding to the SGHMC eta stepsize terms,
#'  as defined in the paper given in the references. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#' @param alpha list of numeric values corresponding to the SGHMC alpha momentum terms,
#'  as defined in the paper given in the references. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#' @param L integer specifying the trajectory of the simulation, as defined in the main reference.
#' @param n minibatch size, assumed integer.
#' @param nIters number of iterations of SGLD to perform, optional, assumed integer, default 10^4.
#' @param verbose whether to print algorithm progress or not, assumed BOOLEAN, default TRUE.
#'
#' @return List of arrays for each parameter containing the MCMC chain.
#'  Dimension of the form (nIters,paramDim1,paramDim2,...)
#'
#' @examples Tutorials available at [link to be added]
#'
runSGHMC = function( logLik, logPrior, data, params, eta, alpha, L, n, 
        nIters = 10^4, verbose = TRUE ) {
    # Setup SGHMC object
    sgmcmc = sghmc( logLik, logPrior, data, params, eta, alpha, L, n, NULL )
    options = list( "nIters" = nIters, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgmcmc, params, options )
    return( paramStorage )
}

#' Stochastic Gradient Hamiltonian Monte Carlo with Control Variates
#' 
#' Simulates from the posterior defined by the functions logLik & logPrior using 
#'  stochastic gradient Hamiltonian Monte Carlo with an improved gradient estimate 
#'  that is calculated using control variates.
#'
#'  @references Baker, J., Fearnhead, P., Fox, E. B., & Nemeth, C. (2017) 
#'      control variates for stochastic gradient Langevin dynamics. Preprint.
#'  @references Chen, T., Fox, E. B., & Guestrin, C. (2014). 
#'      stochastic gradient Hamiltonian Monte Carlo. In ICML (pp. 1683-1691).
#'
#' @param logLik function which takes parameters and data 
#'  (list of tensorflow variables and placeholders respectively) as input. 
#'  It should return a tensorflow expression which defines the log likelihood of the model.
#' @param logPrior function which takes parameters and data 
#'  (list of tensorflow variables and placeholders respectively) as input. 
#'  The function should return a tensorflow tensor which defines the log prior of the model.
#' @param data list of R arrays which defines the datasets for the problem.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param params list of R arrays which define the starting point of each parameter.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param eta list of numeric values corresponding to the SGHMC eta stepsize terms,
#'  as defined in the paper given in the references. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#' @param alpha list of numeric values corresponding to the SGHMC alpha momentum terms,
#'  as defined in the paper given in the references. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#' @param L integer specifying the trajectory of the simulation, as defined in the main reference.
#' @param optStepsize numeric, stepsize for optimization step to find MAP estimates of parameters.
#' @param n minibatch size, assumed integer.
#' @param nIters integer, number of iterations of SGLD to perform, optional, default 10^4.
#' @param nItersOpt integer, number of iterations of initial optimization to perform, 
#'  optional, default 10^4.
#' @param verbose Boolean, whether to print algorithm progress or not, default TRUE.
#'
#' @return List of arrays for each parameter containing the MCMC chain.
#'  Dimension of the form (nIters,paramDim1,paramDim2,...)
#'
#' @examples Tutorials available at [link to be added]
#'
runSGHMCCV = function( logLik, logPrior, data, params, eta, alpha, L, optStepsize, 
            n, nIters = 10^4, nItersOpt = 10^4, verbose = TRUE ) {
    # Setup SGHMCCV object
    sgmcmcCV = sghmcCV( logLik, logPrior, data, params, eta, alpha, L, optStepsize, n, NULL )
    options = list( "nIters" = nIters, "nItersOpt" = nItersOpt, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgmcmcCV, params, options )
    return( paramStorage )
}

#' Create Stochastic Gradient Hamiltonian Monte Carlo Object
#' 
#' Creates a stochastic gradient Langevin Dynamics (SGHMC) object which can be passed to mcmcStep
#'  to simulate from 1 step of SGHMC for the posterior defined by logLik and logPrior.
#'  This is for advanced use of the library when additional Gibbs steps are needed.
#'
#'  @references Chen, T., Fox, E. B., & Guestrin, C. (2014). 
#'      stochastic gradient Hamiltonian Monte Carlo. In ICML (pp. 1683-1691).
#'
#' @param logLik function which takes parameters and data 
#'  (list of tensorflow variables and placeholders respectively) as input. 
#'  It should return a tensorflow expression which defines the log likelihood of the model.
#' @param logPrior function which takes parameters and data 
#'  (list of tensorflow variables and placeholders respectively) as input. 
#'  The function should return a tensorflow tensor which defines the log prior of the model.
#' @param data list of R numeric arrays which defines the datasets for the problem.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param params list of numeric R arrays which define the starting point of each parameter.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param eta list of numeric values corresponding to the SGHMC eta stepsize terms,
#'  as defined in the paper given in the references. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#' @param alpha list of numeric values corresponding to the SGHMC alpha momentum terms,
#'  as defined in the paper given in the references. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#' @param L integer specifying the trajectory of the simulation, as defined in the main reference.
#' @param n integer corresponding to minibatch size
#' @param gibbsParams list of tensorflow variables, corresponding to parameters that will be updated
#'  manually using Gibbs steps
#'
#' @return List of arrays for each parameter containing the MCMC chain.
#'  Dimension of the form (nIters,paramDim1,paramDim2,...)
#'
#' @examples Tutorials available at [link to be added]
#'
sghmc = function( logLik, logPrior, data, params, eta, alpha, L, n, gibbsParams ) {
    # Get dataset size
    N = getDatasetSize( data )
    # Convert params and data to tensorflow variables and placeholders
    paramstf = setupParams( params )
    placeholders = setupPlaceholders( data, n )
    # Declare estimated log posterior tensor using declared variables and placeholders
    estLogPost = setupEstLogPost( logLik, logPrior, paramstf, placeholders, N, n, gibbsParams )
    # Declare SGLD dynamics
    sghmc = list( "data" = data, "n" = n, "placeholders" = placeholders, 
            "eta" = eta, "alpha" = alpha, "L" = L, "params" = paramstf, "estLogPost" = estLogPost )
    class( sghmc ) = c( "sghmc", "sgmcmc" )
    sghmc$dynamics = declareDynamics( sghmc )
    return( sghmc )
}

#' Create Stochastic Gradient Hamiltonian Monte Carlo with Control Variates Object
#' 
#' Creates a stochastic gradient Hamiltonian Monte Carlo with Control Variates (SGHMCCV) object 
#'  which can be passed to optUpdate and mcmcStep functions to simulate from SGHMCCV
#'  for the posterior defined by logLik and logPrior.
#'  This is for advanced use of the library when additional Gibbs steps are needed.
#'
#'  @references Baker, J., Fearnhead, P., Fox, E. B., & Nemeth, C. (2017) 
#'      control variates for stochastic gradient Langevin dynamics. Preprint.
#'  @references Chen, T., Fox, E. B., & Guestrin, C. (2014). 
#'      stochastic gradient Hamiltonian Monte Carlo. In ICML (pp. 1683-1691).
#'
#' @param logLik function which takes parameters and data 
#'  (list of tensorflow variables and placeholders respectively) as input. 
#'  It should return a tensorflow expression which defines the log likelihood of the model.
#' @param logPrior function which takes parameters and data 
#'  (list of tensorflow variables and placeholders respectively) as input. 
#'  The function should return a tensorflow tensor which defines the log prior of the model.
#' @param data list of R numeric arrays which defines the datasets for the problem.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param params list of numeric R arrays which define the starting point of each parameter.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param stepsize list of numeric stepsizes corresponding to the SGLD stepsizes for each parameter
#'  The names in the list should correspond to those in params.
#' @param optStepsize numeric, stepsize for optimization step to find MAP estimates of parameters.
#' @param n integer corresponding to minibatch size
#' @param gibbsParams list of tensorflow variables, corresponding to parameters that will be updated
#'  manually using Gibbs steps
#'
#' @return List of arrays for each parameter containing the MCMC chain.
#'  Dimension of the form (nIters,paramDim1,paramDim2,...)
#'
#' @examples Tutorials available at [link to be added]
#'
sghmcCV = function( logLik, logPrior, data, params, eta, alpha, L, optStepsize, n, gibbsParams ) {
    # Get dataset size
    N = getDatasetSize( data )
    # Convert params and data to tensorflow variables and placeholders
    paramstf = setupParams( params )
    placeholders = setupPlaceholders( data, n )
    # Declare tensorflow variables for initial optimizer
    paramsOpt = setupParams( params )
    placeholdersFull = setupFullPlaceholders( data )
    # Declare container for full gradients at mode
    logPostOptGrad = setupFullGradients( params )
    # Declare estimated log posterior tensor using declared variables and placeholders
    estLogPost = setupEstLogPost( logLik, logPrior, paramstf, placeholders, N, n, gibbsParams )
    # Declare estimated log posterior tensor for optimization
    estLogPostOpt = setupEstLogPost( logLik, logPrior, paramsOpt, placeholders, N, n, gibbsParams )
    # Declare full log posterior for calculation at MAP estimate
    fullLogPostOpt = setupFullLogPost( logLik, logPrior, paramsOpt, placeholdersFull, gibbsParams )
    # Declare optimizer
    optimizer = declareOptimizer( estLogPostOpt, fullLogPostOpt, paramsOpt, 
            paramstf, logPostOptGrad, optStepsize )
    # Create SGHMCCV object
    sghmcCV = list( "optimizer" = optimizer, "data" = data, "n" = n, "eta" = eta, "alpha" = alpha, 
            "L" = L, "placeholders" = placeholders, "placeholdersFull" = placeholdersFull, 
            "params" = paramstf, "paramsOpt" = paramsOpt, "estLogPost" = estLogPost, 
            "estLogPostOpt" = estLogPostOpt, "logPostOptGrad" = logPostOptGrad, 
            "fullLogPostOpt" = fullLogPostOpt )
    class(sghmcCV) = c( "sghmc", "sgmcmcCV" )
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
        eta = sghmc$eta[[pname]]
        alpha = sghmc$alpha[[pname]]
        # Declare parameters
        theta = sghmc$params[[pname]]
        nu = tf$Variable( sqrt( eta ) * tf$random_normal( theta$get_shape() ) )
        # Declare dynamics
        gradU = estLogPostGrads[[pname]]
        dynamics$refresh[[pname]] = nu$assign( sqrt( eta ) * tf$random_normal( theta$get_shape() ) )
        dynamics$nu[[pname]] = nu$assign_add( eta*gradU - alpha*nu + 
                sqrt( 2 * eta * alpha ) * tf$random_normal( theta$get_shape() ) )
        dynamics$theta[[pname]] = theta$assign_add( nu )
    }
    return( dynamics )
}
