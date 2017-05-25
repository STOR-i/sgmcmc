#' Stochastic Gradient Langevin Dynamics
#' 
#' Simulates from the posterior defined by the functions logLik & logPrior using
#'  stochastic gradient Langevin Dynamics. The function uses TensorFlow, so needs
#'  Tensorflow for python installed.
#'
#'  @references Welling, M., & Teh, Y. W. (2011). 
#'      Bayesian learning via stochastic gradient Langevin dynamics. ICML (pp. 681-688).
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
#' @param stepsize list of stepsizes corresponding to the SGLD stepsizes for each parameter
#'  The names in the list should correspond to those in params.
#' @param n minibatch size, assumed integer.
#' @param nIters number of iterations of SGLD to perform, optional, assumed integer, default 10^4.
#' @param verbose whether to print algorithm progress or not, assumed BOOLEAN, default TRUE.
#'
#' @return List of arrays for each parameter containing the MCMC chain.
#'  Dimension of the form (nIters,paramDim1,paramDim2,...)
#'
#' @import tensorflow
#'
sgld = function( logLik, logPrior, data, params, stepsize, n, nIters = 10^4, verbose = TRUE ) {
    # Create SGLD object
    sgmcmc = genSGLD( logLik, logPrior, data, params, stepsize, n, NULL )
    options = list( "nIters" = nIters, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgmcmc, params, options )
    return( paramStorage )
}


#' Stochastic Gradient Langevin Dynamics with Control Variates
#' 
#' Simulates from the posterior defined by the functions logLik & logPrior using
#'  stochastic gradient Langevin Dynamics with an improved gradient estimate using Control Variates.
#'  The function uses TensorFlow, so needs Tensorflow for python installed.
#'
#'  @references Baker, J., Fearnhead, P., Fox, E. B., & Nemeth, C. (2017) 
#'      control variates for stochastic gradient Langevin dynamics. Preprint.
#'  @references Welling, M., & Teh, Y. W. (2011). 
#'      Bayesian learning via stochastic gradient Langevin dynamics. ICML (pp. 681-688).
#'
#' @param logLik function which takes parameters and data 
#'  (list of tensorflow variables and placeholders respectively) as input. 
#'  It should return a tensorflow expression which defines the log likelihood of the model.
#' @param logPrior function which takes parameters and data 
#'  (list of tensorflow variables and placeholders respectively) as input. 
#'  The function should return a tensorflow tensor which defines the log prior of the model.
#' @param data list of numeric R arrays which defines the datasets for the problem.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param params list of numeric R arrays which define the starting point of each parameter.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param stepsize list of stepsizes corresponding to the SGLD stepsizes for each parameter
#'  The names in the list should correspond to those in params.
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
#' @import tensorflow
#'
sgldcv = function( logLik, logPrior, data, paramsRaw, stepsize, optStepsize, 
            n, nIters = 10^4, nItersOpt = 10^4, verbose = TRUE ) {
    # Setup SGLDCV object
    sgmcmcCV = genSGLDCV( logLik, logPrior, data, paramsRaw, stepsize, optStepsize, n, NULL )
    options = list( "nIters" = nIters, "nItersOpt" = nItersOpt, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgmcmcCV, paramsRaw, options )
    return( paramStorage )
}

# Create Stochastic Gradient Langevin Dynamics Object
# 
# Creates a stochastic gradient Langevin Dynamics (SGLD) object which can be passed to mcmcStep
#  to simulate from 1 step of SGLD for the posterior defined by logLik and logPrior.
genSGLD = function( logLik, logPrior, data, params, stepsize, n, gibbsParams ) {
    # Get dataset size
    N = getDatasetSize( data )
    # Convert params and data to tensorflow variables and placeholders
    paramstf = setupParams( params )
    placeholders = setupPlaceholders( data, n )
    # Declare estimated log posterior tensor using declared variables and placeholders
    estLogPost = setupEstLogPost( logLik, logPrior, paramstf, placeholders, N, n, gibbsParams )
    # Declare SGLD object
    sgld = list( "data" = data, "n" = n, "placeholders" = placeholders, "stepsize" = stepsize,
            "params" = paramstf, "estLogPost" = estLogPost )
    class( sgld ) = c( "sgld", "sgmcmc" )
    # Declare SGLD dynamics
    sgld$dynamics = declareDynamics( sgld )
    return( sgld )
}

# Create Stochastic Gradient Langevin Dynamics with Control Variates Object
# 
# Creates a stochastic gradient Langevin Dynamics with Control Variates (SGLDCV) object 
#  which can be passed to optUpdate and mcmcStep to simulate from 1 step of SGLD 
#  for the posterior defined by logLik and logPrior.
genSGLDCV = function( logLik, logPrior, data, params, stepsize, optStepsize, n, gibbsParams ) {
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
    # Declare SGLDCV object
    sgldCV = list( "optimizer" = optimizer, "data" = data, "n" = n, "stepsize" = stepsize,
            "placeholders" = placeholders, "placeholdersFull" = placeholdersFull, 
            "params" = paramstf, "paramsOpt" = paramsOpt, "estLogPost" = estLogPost, 
            "estLogPostOpt" = estLogPostOpt, "logPostOptGrad" = logPostOptGrad )
    class(sgldCV) = c( "sgld", "sgmcmcCV" )
    # Declare SGLD dynamics
    sgldCV$dynamics = declareDynamics( sgldCV )
    return( sgldCV )
}

# Declare the tensorflow steps needed for one step of SGLD
# @param sgld is an sgld object
declareDynamics.sgld = function( sgld ) {
    # dynamics is returned, contains list of tensorflow steps for SGLD
    dynamics = list()
    # Get the correct gradient estimate given the sgld object (i.e. standard sgld or sgldcv) 
    estLogPostGrads = getGradients( sgld )
    for ( pname in names( sgld$params ) ) {
        # Declare simulation parameters
        theta = sgld$params[[pname]]
        epsilon = sgld$stepsize[[pname]]
        grad = estLogPostGrads[[pname]]
        # Declare form of one step of SGLD
        dynamics[[pname]] = theta$assign_add( 0.5 * epsilon * grad + 
                sqrt( epsilon ) * tf$random_normal( theta$get_shape() ) )
    }
    return( dynamics )
}
