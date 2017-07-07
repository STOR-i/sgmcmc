#' Stochastic Gradient Nose Hoover Thermostat
#' 
#' Simulates from the posterior defined by the functions logLik and logPrior using
#'  stochastic gradient Nose Hoover Thermostat.
#'  When used for parameters that are higher dimensional then vectors, the thermostat
#'  step is approximated by a tensor contraction. Currently the implementation will only
#'  handle vector or scalar parameters, this will be updated in future implementations.
#'
#' @references \itemize{\item \href{http://people.ee.duke.edu/~lcarin/sgnht-4.pdf}{
#'  Ding, N., Fang, Y., Babbush, R., Chen, C., Skeel, R. D., and Neven, H. (2014). 
#'  Bayesian sampling using stochastic gradient thermostats. NIPS (pp. 3203-3211).}}
#'
#' @param logLik function which takes parameters and dataset 
#'  (list of tensorflow variables and placeholders respectively) as input. 
#'  It should return a tensorflow expression which defines the log likelihood of the model.
#' @param dataset list of R arrays which defines the datasets for the problem.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param params list of R arrays which define the starting point of each parameter.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param stepsize list of numeric values corresponding to the SGNHT eta stepsize terms,
#'  as defined in the paper given in the references, Appendix F. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#'  Alternatively specify a single float to use that stepsize for all parameters.
#' @param logPrior function which takes parameters (list of tensorflow variables) as input.
#'  The function should return a tensorflow tensor which defines the log prior of the model.
#'  Optional. Default uninformative improper prior.
#' @param minibatchSize either as proportion of dataset size or actual size, assumed float or int.
#'  Optional. Default 0.01.
#' @param a list of numeric values corresponding to the SGNHT a diffusion factors,
#'  as defined in the paper given in the references, Appendix F. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#'  Alternatively specify a single float to specify that value for all parameters.
#'  Optional. Default 0.01 for all parameters.
#' @param nIters number of iterations of SGLD to perform, optional, assumed integer, default 10^4.
#' @param verbose whether to print algorithm progress or not, assumed BOOLEAN, default TRUE.
#'
#' @return List of arrays for each parameter containing the MCMC chain.
#'  Dimension of the form (nIters,paramDim1,paramDim2,...)
#'
#' @export
#'
sgnht = function( logLik, dataset, params, stepsize, logPrior = NULL, minibatchSize = 500, 
        a = 0.01, nIters = 10^4, verbose = TRUE ) {
    # Declare SGNHT object
    sgmcmc = genSGNHT( logLik, logPrior, dataset, params, stepsize, a, minibatchSize )
    options = list( "nIters" = nIters, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgmcmc, params, options )
    return( paramStorage )
}

#' Stochastic Gradient Nose Hoover Thermostat with Control Variates
#' 
#' Simulates from the posterior defined by the functions logLik and logPrior using
#'  stochastic gradient Nose Hoover Thermostat with an improved gradient estimate 
#'  that is calculated using control variates.
#'  When used for parameters that are higher dimensional then vectors, the thermostat
#'  step is approximated by a tensor contraction. Currently the implementation will only
#'  handle vector or scalar parameters, this will be updated in future implementations.
#'
#' @references \itemize{
#'  \item \href{https://arxiv.org/pdf/1706.05439.pdf}{
#'  Baker, J., Fearnhead, P., Fox, E. B., and Nemeth, C. (2017).
#'  Control variates for stochastic gradient MCMC. ArXiv preprint arXiv:1706.05439.}
#'  \item \href{http://people.ee.duke.edu/~lcarin/sgnht-4.pdf}{
#'  Ding, N., Fang, Y., Babbush, R., Chen, C., Skeel, R. D., and Neven, H. (2014). 
#'  Bayesian sampling using stochastic gradient thermostats. NIPS (pp. 3203-3211).}}
#'
#' @param logLik function which takes parameters and dataset 
#'  (list of tensorflow variables and placeholders respectively) as input. 
#'  It should return a tensorflow expression which defines the log likelihood of the model.
#' @param dataset list of R arrays which defines the datasets for the problem.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param params list of R arrays which define the starting point of each parameter.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param stepsize list of numeric values corresponding to the SGNHT eta stepsize terms,
#'  as defined in the paper given in the references, Appendix F. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#'  Alternatively specify a single float to use that stepsize for all parameters.
#' @param optStepsize numeric, stepsize for optimization step to find MAP estimates of parameters.
#' @param logPrior function which takes parameters (list of tensorflow variables) as input.
#'  The function should return a tensorflow tensor which defines the log prior of the model.
#'  Optional. Default uninformative improper prior.
#' @param minibatchSize either as proportion of dataset size or actual size, assumed float or int.
#'  Optional. Default 0.01.
#' @param a list of numeric values corresponding to the SGNHT a diffusion factors,
#'  as defined in the paper given in the references, Appendix F. One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#'  Alternatively specify a single float to specify that value for all parameters.
#'  Optional. Default 0.01 for all parameters.
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
sgnhtcv = function( logLik, dataset, params, stepsize, optStepsize, logPrior = NULL, 
        minibatchSize = 500, a = 0.01, nIters = 10^4, nItersOpt = 10^4, verbose = TRUE ) {
    # Declare SGNHTCV object
    sgmcmcCV = genSGNHTCV( logLik, logPrior, dataset, params, stepsize, a, optStepsize, minibatchSize)
    options = list( "nIters" = nIters, "nItersOpt" = nItersOpt, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgmcmcCV, params, options )
    return( paramStorage )
}

# Create Stochastic Gradient Nose Hoover Thermostat Object
# 
# Creates a stochastic gradient Nose Hoover Thermostat (SGNHT) object which can be passed 
#  to mcmcStep to simulate from 1 step of SGHMC for the posterior defined by logLik and logPrior.
genSGNHT = function( logLik, logPrior, dataset, params, stepsize, a, minibatchSize ) {
    # Create generic sgmcmc object
    sgnht = createSGMCMC( logLik, logPrior, dataset, params, stepsize, minibatchSize )
    # Get ranks for each parameter tensor, required for sgnht dynamics
    sgnht$ranks = getRanks( params )
    # Declare sgnht specific tuning constants, checking they're in list format
    sgnht$a = convertList( a, sgnht$params )
    # Declare object types
    class(sgnht) = c( "sgnht", "sgmcmc" )
    # Declare SGNHT dynamics
    sgnht$dynamics = declareDynamics( sgnht )
    return( sgnht )
}

# Create Stochastic Gradient Nose Hoover Thermostat with Control Variates Object
# 
# Creates a stochastic gradient Nose Hoover Thermostat with Control Variates (SGHMCCV) object 
#  which can be passed to optUpdate and mcmcStep functions to simulate from SGNHTCV
#  for the posterior defined by logLik and logPrior.
genSGNHTCV = function( logLik, logPrior, dataset, params, stepsize, a, 
            optStepsize, minibatchSize ) {
    # Create generic sgmcmcCV object
    sgnhtCV = createSGMCMCCV( 
        logLik, logPrior, dataset, params, stepsize, optStepsize, minibatchSize )
    # Get ranks for each parameter tensor, required for sgnht dynamics
    sgnhtCV$ranks = getRanks( params )
    # Declare sgnht specific tuning constants, checking they're in list format
    sgnhtCV$a = convertList( a, sgnhtCV$params )
    # Declare object types
    class(sgnhtCV) = c( "sgnht", "sgmcmcCV" )
    # Declare SGNHT dynamics
    sgnhtCV$dynamics = declareDynamics( sgnhtCV )
    return( sgnhtCV )
}

# Declare the tensorflow steps needed for one step of SGNHT
# @param sgnht is an sgnht object
declareDynamics.sgnht = function( sgnht ) {
    dynamics = list( "theta" = list(), "u" = list(), "alpha" = list() )
    estLogPostGrads = getGradients( sgnht )
    for ( pname in names(sgnht$params) ) {
        # Get constants for this parameter
        stepsize = sgnht$stepsize[[pname]]
        a = sgnht$a[[pname]]
        rankTheta = sgnht$ranks[[pname]]
        # Declare momentum params
        theta = sgnht$params[[pname]]
        u = tf$Variable( sqrt(stepsize) * tf$random_normal( theta$get_shape() ) )
        alpha = tf$Variable( a, dtype = tf$float32 )
        # Declare dynamics
        gradU = estLogPostGrads[[pname]]
        dynamics$u[[pname]] = u$assign_sub( u * alpha - stepsize * gradU - 
                sqrt( 2*a*stepsize ) * tf$random_normal( u$get_shape() ) )
        dynamics$theta[[pname]] = theta$assign_add( u )
        # tensordot throws error if rank is 0 so catch this edge case
        if ( rankTheta == 0 ) {
            dynamics$alpha[[pname]] = alpha$assign_add( u * u - stepsize )
        } else if( rankTheta >= 1 ) {
            axes = matrix( rep( 0:( rankTheta - 1 ), each = 2 ), nrow = 2 )
            axes = tf$constant( axes, dtype = tf$int32 )
            dynamics$alpha[[pname]] = alpha$assign_add( 
                    tf$tensordot( u, u, axes ) / tf$size( u, out_type = tf$float32 ) - stepsize )
        } 
    }
    return( dynamics )
}
