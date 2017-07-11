#' Stochastic Gradient Langevin Dynamics
#' 
#' Simulates from the posterior defined by the functions logLik and logPrior using
#'  stochastic gradient Langevin Dynamics. The function uses TensorFlow, so needs
#'  Tensorflow for python installed.
#'
#' @references \itemize{\item \href{http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf}{
#'  Welling, M., and Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. 
#'  ICML (pp. 681-688).}}
#'
#' @param logLik function which takes parameters and dataset 
#'  (list of TensorFlow variables and placeholders respectively) as input. 
#'  It should return a TensorFlow expression which defines the log likelihood of the model.
#' @param dataset list of numeric R arrays which defines the datasets for the problem.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param params list of numeric R arrays which define the starting point of each parameter.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param stepsize list of numeric values corresponding to the SGLD stepsizes for each parameter
#'  The names in the list should correspond to those in params.
#'  Alternatively specify a single numeric value to use that stepsize for all parameters.
#' @param logPrior optional. Default uninformative improper prior.
#'  Function which takes parameters (list of TensorFlow variables) as input.
#'  The function should return a TensorFlow tensor which defines the log prior of the model.
#' @param minibatchSize optional. Default 0.01.
#'  Numeric or integer value that specifies amount of dataset to use at each iteration 
#'  either as proportion of dataset size (if between 0 and 1) or actual magnitude (if an integer).
#' @param nIters optional. Default 10^4L. Integer specifying number of iterations to perform.
#' @param verbose optional. Default TRUE. Boolean specifying whether to print algorithm progress
#'
#' @return Returns list of arrays for each parameter containing the MCMC chain.
#'  Dimension of the form (nIters,paramDim1,paramDim2,...)
#'
#' @export
#'
sgld = function( logLik, dataset, params, stepsize, logPrior = NULL, minibatchSize = 0.01, 
        nIters = 10^4L, verbose = TRUE ) {
    # Create SGLD object
    sgmcmc = genSGLD( logLik, logPrior, dataset, params, stepsize, minibatchSize )
    options = list( "nIters" = nIters, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgmcmc, params, options )
    return( paramStorage )
}


#' Stochastic Gradient Langevin Dynamics with Control Variates
#' 
#' Simulates from the posterior defined by the functions logLik and logPrior using
#'  stochastic gradient Langevin Dynamics with an improved gradient estimate using Control Variates.
#'  The function uses TensorFlow, so needs Tensorflow for python installed.
#'
#' @references \itemize{
#'  \item \href{https://arxiv.org/pdf/1706.05439.pdf}{
#'  Baker, J., Fearnhead, P., Fox, E. B., and Nemeth, C. (2017).
#'  Control variates for stochastic gradient MCMC. ArXiv preprint arXiv:1706.05439.}
#'  \item \href{http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf}{
#'  Welling, M., and Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. 
#'  ICML (pp. 681-688).} }
#'
#' @param logLik function which takes parameters and dataset 
#'  (list of TensorFlow variables and placeholders respectively) as input. 
#'  It should return a TensorFlow expression which defines the log likelihood of the model.
#' @param dataset list of numeric R arrays which defines the datasets for the problem.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param params list of numeric R arrays which define the starting point of each parameter.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param stepsize list of numeric values corresponding to the SGLD stepsizes for each parameter
#'  The names in the list should correspond to those in params.
#'  Alternatively specify a single numeric value to use that stepsize for all parameters.
#' @param optStepsize numeric value specifying the stepsize for the optimization 
#'  to find MAP estimates of parameters. The TensorFlow AdamOptimizer is used.
#' @param logPrior optional. Default uninformative improper prior.
#'  Function which takes parameters (list of TensorFlow variables) as input.
#'  The function should return a TensorFlow tensor which defines the log prior of the model.
#' @param minibatchSize optional. Default 0.01.
#'  Numeric or integer value that specifies amount of dataset to use at each iteration 
#'  either as proportion of dataset size (if between 0 and 1) or actual magnitude (if an integer).
#' @param nIters optional. Default 10^4L. Integer specifying number of iterations to perform.
#' @param nItersOpt optional. Default 10^4L. 
#'  Integer specifying number of iterations of initial optimization to perform.
#' @param verbose optional. Default TRUE. Boolean specifying whether to print algorithm progress
#'
#' @return Returns list of arrays for each parameter containing the MCMC chain.
#'  Dimension of the form (nIters,paramDim1,paramDim2,...)
#'
#' @export
#'
sgldcv = function( logLik, dataset, params, stepsize, optStepsize, logPrior = NULL,
        minibatchSize = 0.01, nIters = 10^4L, nItersOpt = 10^4L, verbose = TRUE ) {
    # Setup SGLDCV object
    sgmcmcCV = genSGLDCV( logLik, logPrior, dataset, params, stepsize, optStepsize, minibatchSize )
    options = list( "nIters" = nIters, "nItersOpt" = nItersOpt, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgmcmcCV, params, options )
    return( paramStorage )
}

# Create Stochastic Gradient Langevin Dynamics Object
# 
# Creates a stochastic gradient Langevin Dynamics (SGLD) object which can be passed to mcmcStep
#  to simulate from 1 step of SGLD for the posterior defined by logLik and logPrior.
genSGLD = function( logLik, logPrior, dataset, params, stepsize, minibatchSize ) {
    # Create generic sgmcmc object
    sgld = createSGMCMC( logLik, logPrior, dataset, params, stepsize, minibatchSize )
    # Declare object type
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
genSGLDCV = function( logLik, logPrior, dataset, params, stepsize, optStepsize, minibatchSize ) {
    # Create generic sgmcmcCV object
    sgldCV = createSGMCMCCV( 
        logLik, logPrior, dataset, params, stepsize, optStepsize, minibatchSize )
    class(sgldCV) = c( "sgld", "sgmcmcCV" )
    # Declare SGLD dynamics
    sgldCV$dynamics = declareDynamics( sgldCV )
    return( sgldCV )
}

# Declare the TensorFlow steps needed for one step of SGLD
# @param sgld is an sgld object
declareDynamics.sgld = function( sgld ) {
    # dynamics is returned, contains list of TensorFlow steps for SGLD
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
