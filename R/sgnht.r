library(tensorflow)
source("setup.r")
source("update.r")
source("storage.r")
source("controlVariates.r")

#' Stochastic Gradient Nose Hoover Thermostat
#' 
#' Simulates from the posterior defined by the functions logLik & logPrior using
#'  stochastic gradient Nose Hoover Thermostat.
#'  When used for parameters that are higher dimensional then vectors, the thermostat
#'  step is approximated by a tensor contraction.
#'
#'  @references Ding, N., Fang, Y., Babbush, R., Chen, C., Skeel, R. D., & Neven, H. (2014). 
#'      Bayesian sampling using stochastic gradient thermostats. NIPS (pp. 3203-3211).
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
#' @param eta list of numeric values corresponding to the SGNHT eta stepsize terms,
#'  as defined in the paper given in the references, Appendix F. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#' @param a list of numeric values corresponding to the SGNHT a diffusion factors,
#'  as defined in the paper given in the references, Appendix F. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#' @param n minibatch size, assumed integer.
#' @param nIters number of iterations of SGLD to perform, optional, assumed integer, default 10^4.
#' @param verbose whether to print algorithm progress or not, assumed BOOLEAN, default TRUE.
#'
#' @return List of arrays for each parameter containing the MCMC chain.
#'  Dimension of the form (nIters,paramDim1,paramDim2,...)
#'
#' @examples Tutorials available at [link to be added]
#'
runSGNHT = function( logLik, logPrior, data, params, eta, a, n, 
        nIters = 10^4, verbose = TRUE ) {
    # Declare SGNHT object
    sgmcmc = sgnht( logLik, logPrior, data, params, eta, a, n, NULL )
    options = list( "nIters" = nIters, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgmcmc, params, options )
    return( paramStorage )
}

#' Stochastic Gradient Nose Hoover Thermostat with Control Variates
#' 
#' Simulates from the posterior defined by the functions logLik & logPrior using
#'  stochastic gradient Nose Hoover Thermostat with an improved gradient estimate 
#'  that is calculated using control variates.
#'  When used for parameters that are higher dimensional then vectors, the thermostat
#'  step is approximated by a tensor contraction.
#'
#' @references Baker, J., Fearnhead, P., Fox, E. B., & Nemeth, C. (2017) 
#'  control variates for stochastic gradient Langevin dynamics. Preprint.
#' @references Ding, N., Fang, Y., Babbush, R., Chen, C., Skeel, R. D., & Neven, H. (2014). 
#'  Bayesian sampling using stochastic gradient thermostats. NIPS (pp. 3203-3211).
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
#' @param eta list of numeric values corresponding to the SGNHT eta stepsize terms,
#'  as defined in the paper given in the references, Appendix F. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#' @param a list of numeric values corresponding to the SGNHT a diffusion factors,
#'  as defined in the paper given in the references, Appendix F. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
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
runSGNHTCV = function( logLik, logPrior, data, params, eta, a, optStepsize, n, 
            nIters = 10^4, nItersOpt = 10^4, verbose = TRUE ) {
    # Declare SGNHTCV object
    sgmcmcCV = sgnhtCV( logLik, logPrior, data, params, eta, a, optStepsize, n, NULL )
    options = list( "nIters" = nIters, "nItersOpt" = nItersOpt, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgmcmcCV, params, options )
    return( paramStorage )
}

#' Create Stochastic Gradient Nose Hoover Thermostat Object
#' 
#' Creates a stochastic gradient Nose Hoover Thermostat (SGNHT) object which can be passed 
#'  to mcmcStep to simulate from 1 step of SGHMC for the posterior defined by logLik and logPrior.
#'  This is for advanced use of the library when additional Gibbs steps are needed.
#'
#'  @references Ding, N., Fang, Y., Babbush, R., Chen, C., Skeel, R. D., & Neven, H. (2014). 
#'      Bayesian sampling using stochastic gradient thermostats. NIPS (pp. 3203-3211).
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
#' @param eta list of numeric values corresponding to the SGNHT eta stepsize terms,
#'  as defined in the paper given in the references, Appendix F. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#' @param a list of numeric values corresponding to the SGNHT a diffusion factors,
#'  as defined in the paper given in the references, Appendix F. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#' @param n minibatch size, assumed integer.
#' @param gibbsParams list of tensorflow variables, corresponding to parameters that will be updated
#'  manually using Gibbs steps
#'
#' @return List of arrays for each parameter containing the MCMC chain.
#'  Dimension of the form (nIters,paramDim1,paramDim2,...)
#'
#' @examples Tutorials available at [link to be added]
#'
sgnht = function( logLik, logPrior, data, params, eta, a, n, gibbsParams ) {
    # 
    # Get dataset size
    N = dim( data[[1]] )[1]
    # Convert params and data to tensorflow variables and placeholders
    paramstf = setupParams( params )
    placeholders = setupPlaceholders( data, n )
    # Declare estimated log posterior tensor using declared variables and placeholders
    estLogPost = setupEstLogPost( logLik, logPrior, paramstf, placeholders, N, n, gibbsParams )
    # Get ranks for each parameter tensor, required for dynamics
    ranks = getRanks( params )
    # Create SGNHT object
    sgnht = list( "data" = data, "n" = n, "eta" = eta, "a" = a, "ranks" = ranks, 
            "placeholders" = placeholders, "params" = paramstf, "estLogPost" = estLogPost )
    class(sgnht) = c( "sgnht", "sgmcmc" )
    # Declare SGNHT dynamics
    sgnht$dynamics = declareDynamics( sgnht )
    return( sgnht )
}

#' Create Stochastic Gradient Nose Hoover Thermostat with Control Variates Object
#' 
#' Creates a stochastic gradient Nose Hoover Thermostat with Control Variates (SGHMCCV) object 
#'  which can be passed to optUpdate and mcmcStep functions to simulate from SGNHTCV
#'  for the posterior defined by logLik and logPrior.
#'  This is for advanced use of the library when additional Gibbs steps are needed.
#'
#' @references Baker, J., Fearnhead, P., Fox, E. B., & Nemeth, C. (2017) 
#'  control variates for stochastic gradient Langevin dynamics. Preprint.
#' @references Ding, N., Fang, Y., Babbush, R., Chen, C., Skeel, R. D., & Neven, H. (2014). 
#'  Bayesian sampling using stochastic gradient thermostats. NIPS (pp. 3203-3211).
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
#' @param eta list of numeric values corresponding to the SGNHT eta stepsize terms,
#'  as defined in the paper given in the references, Appendix F. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#' @param a list of numeric values corresponding to the SGNHT a diffusion factors,
#'  as defined in the paper given in the references, Appendix F. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#' @param optStepsize numeric, stepsize for optimization step to find MAP estimates of parameters.
#' @param n minibatch size, assumed integer.
#' @param gibbsParams list of tensorflow variables, corresponding to parameters that will be updated
#'  manually using Gibbs steps
#'
#' @return List of arrays for each parameter containing the MCMC chain.
#'  Dimension of the form (nIters,paramDim1,paramDim2,...)
#'
#' @examples Tutorials available at [link to be added]
#'
sgnhtCV = function( logLik, logPrior, data, params, eta, a, 
            optStepsize, n, gibbsParams, n_iters = 10^4 ) {
    # Get key sizes and declare correction term for log posterior estimate
    N = dim( data[[1]] )[1]
    correction = tf$constant( N / n, dtype = tf$float32 )
    # Convert params and data to tensorflow variables and placeholders
    paramstf = setupParams( params )
    placeholders = setupPlaceholders( data, minibatch_size )
    # Declare tensorflow variables for initial optimizer
    paramsOpt = setupParams( params )
    placeholdersFull = setupFullPlaceholders( data )
    # Get ranks for each parameter tensor, required for dynamics
    ranks = getRanks( params )
    # Declare container for full gradient
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
    # Declare SGNHTCV object
    sgnhtCV = list( "optimizer" = optimizer, "data" = data, "n" = n, 
            "eta" = eta, "a" = a, "ranks" = ranks, "placeholders" = placeholders, 
            "placeholdersFull" = placeholdersFull, "params" = paramstf, "paramsOpt" = paramsOpt, 
            "estLogPost" = estLogPost, "estLogPostOpt" = estLogPostOpt, 
            "logPostOptGrad" = logPostOptGrad )
    class(sgnhtCV) = c( "sgnht", "sgmcmcCV" )
    # Declare SGNHT dynamics
    sgnht$dynamics = declareDynamics( sgnht )
    return( sgnhtCV )
}

# Declare the tensorflow steps needed for one step of SGNHT
# @param sgnht is an sgnht object
declareDynamics.sgnht = function( sgnht ) {
    dynamics = list( "theta" = list(), "u" = list(), "alpha" = list() )
    estLogPostGrads = getGradients( sgnht )
    for ( pname in names(sgnht$params) ) {
        # Get constants for this parameter
        eta = sgnht$eta[[pname]]
        a = sgnht$a[[pname]]
        rankTheta = sgnht$ranks[[pname]]
        # Declare momentum params
        theta = sgnht$params[[pname]]
        u = tf$Variable( sqrt(eta) * tf$random_normal( theta$get_shape() ) )
        alpha = tf$Variable( a, dtype = tf$float32 )
        # Declare dynamics
        gradU = estLogPostGrads[[pname]]
        dynamics$u[[pname]] = u$assign_sub( u * alpha - eta * gradU - 
                sqrt( 2*a*eta ) * tf$random_normal( u$get_shape() ) )
        dynamics$theta[[pname]] = theta$assign_add( u )
        # tensordot throws error if rank is 0 so catch this edge case
        if ( rankTheta == 0 ) {
            dynamics$alpha[[pname]] = alpha$assign_add( u * u - eta )
        } else {
            dynamics$alpha[[pname]] = alpha$assign_add( 
                    tf$tensordot( u, u, rankTheta ) / tf$size( u, out_type = tf$float32 ) - eta )
        }
    }
    return( dynamics )
}
