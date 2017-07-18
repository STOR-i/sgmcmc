#' Stochastic Gradient Nose Hoover Thermostat
#' 
#' Simulates from the posterior defined by the functions logLik and logPrior using
#'  stochastic gradient Nose Hoover Thermostat.
#'  The thermostat step needs a dot product to be calculated between two vectors.
#'  So when the algorithm uses parameters that are higher order than vectors 
#'  (e.g. matrices and tensors), the thermostat step uses a tensor contraction.
#'  Tensor contraction is otherwise known as the inner product between two tensors.
#'
#' @references \itemize{\item \href{http://people.ee.duke.edu/~lcarin/sgnht-4.pdf}{
#'  Ding, N., Fang, Y., Babbush, R., Chen, C., Skeel, R. D., and Neven, H. (2014). 
#'  Bayesian sampling using stochastic gradient thermostats. NIPS (pp. 3203-3211).}}
#'
#' @param logLik function which takes parameters and dataset 
#'  (list of TensorFlow variables and placeholders respectively) as input. 
#'  It should return a TensorFlow expression which defines the log likelihood of the model.
#' @param dataset list of numeric R arrays which defines the datasets for the problem.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param params list of numeric R arrays which define the starting point of each parameter.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param stepsize list of numeric values corresponding to the SGNHT stepsizes for each parameter
#'  (\eqn{\eta} in the original paper, Algorithm 2). 
#'  The names in the list should correspond to those in params.
#'  Alternatively specify a single numeric value to use that stepsize for all parameters.
#' @param logPrior optional. Default uninformative improper prior.
#'  Function which takes parameters (list of TensorFlow variables) as input.
#'  The function should return a TensorFlow tensor which defines the log prior of the model.
#' @param minibatchSize optional. Default 0.01.
#'  Numeric or integer value that specifies amount of dataset to use at each iteration 
#'  either as proportion of dataset size (if between 0 and 1) or actual magnitude (if an integer).
#' @param a optional. Default 0.01. List of numeric values corresponding to SGNHT diffusion factors
#'  (see Algorithm 2 of the original paper). One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#'  Alternatively specify a single float to specify that value for all parameters.
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
#' stepsize = list("theta" = 5e-6)
#' output = sgnht(logLik, dataset, params, stepsize)
#' }
sgnht = function( logLik, dataset, params, stepsize, logPrior = NULL, minibatchSize = 0.01, 
            a = 0.01, nIters = 10^4L, verbose = TRUE ) {
    # Declare SGNHT object
    sgnht = genSGNHT( logLik, logPrior, dataset, params, stepsize, a, minibatchSize )
    options = list( "nIters" = nIters, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgnht, params, options )
    return( paramStorage )
}

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
#' @param logLik function which takes parameters and dataset 
#'  (list of TensorFlow variables and placeholders respectively) as input. 
#'  It should return a TensorFlow expression which defines the log likelihood of the model.
#' @param dataset list of numeric R arrays which defines the datasets for the problem.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param params list of numeric R arrays which define the starting point of each parameter.
#'  The names in the list should correspond to those referred to in the logLik and logPrior functions
#' @param stepsize list of numeric values corresponding to the SGNHT stepsizes for each parameter
#'  (\eqn{\eta} in the original paper, Algorithm 2). 
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
#' @param a optional. Default 0.01. List of numeric values corresponding to SGNHT diffusion factors
#'  (see Algorithm 2 of the original paper). One value should be given 
#'  for each parameter in params, the names should correspond to those in params.
#'  Alternatively specify a single float to specify that value for all parameters.
#' @param nIters optional. Default 10^4L. Integer specifying number of iterations to perform.
#' @param nItersOpt optional. Default 10^4L. 
#'  Integer specifying number of iterations of initial optimization to perform.
#' @param verbose optional. Default TRUE. Boolean specifying whether to print algorithm progress.
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
            minibatchSize = 0.01, a = 0.01, nIters = 10^4L, nItersOpt = 10^4L, verbose = TRUE ) {
    # Declare SGNHTCV object
    sgnhtcv = genSGNHTCV( logLik, logPrior, dataset, params, stepsize, a, optStepsize, minibatchSize)
    options = list( "nIters" = nIters, "nItersOpt" = nItersOpt, "verbose" = verbose )
    # Run MCMC for declared object
    paramStorage = runSGMCMC( sgnhtcv, params, options )
    return( paramStorage )
}

# Create Stochastic Gradient Nose Hoover Thermostat Object
# 
# Creates a stochastic gradient Nose Hoover Thermostat (SGNHT) object which can be passed 
#  to mcmcStep to simulate from 1 step of SGNHT for the posterior defined by logLik and logPrior.
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
# Creates a stochastic gradient Nose Hoover Thermostat with Control Variates (SGNHTCV) object 
#  which can be passed to optUpdate and mcmcStep functions to simulate from SGNHTCV
#  for the posterior defined by logLik and logPrior.
genSGNHTCV = function( logLik, logPrior, dataset, params, stepsize, a, 
            optStepsize, minibatchSize ) {
    # Create generic sgmcmcCV object
    sgnhtcv = createSGMCMCCV( 
            logLik, logPrior, dataset, params, stepsize, optStepsize, minibatchSize )
    # Get ranks for each parameter tensor, required for sgnht dynamics
    sgnhtcv$ranks = getRanks( params )
    # Declare sgnht specific tuning constants, checking they're in list format
    sgnhtcv$a = convertList( a, sgnhtcv$params )
    # Declare object types
    class(sgnhtcv) = c( "sgnht", "sgmcmccv" )
    # Declare SGNHT dynamics
    sgnhtcv$dynamics = declareDynamics( sgnhtcv )
    return( sgnhtcv )
}

# Declare the TensorFlow steps needed for one step of SGNHT
# @param sgnht is an sgnht object
declareDynamics.sgnht = function( sgnht ) {
    dynamics = list( "theta" = list(), "u" = list(), "alpha" = list() )
    estLogPostGrads = getGradients( sgnht )
    # Loop over each parameter in params
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
        dynamics$u[[pname]] = u$assign_add( stepsize * gradU - u * alpha +  
                sqrt( 2 * a * stepsize ) * tf$random_normal( u$get_shape() ) )
        dynamics$theta[[pname]] = theta$assign_add( u )
        # Tensordot throws error if rank is 0 so catch this edge case
        # For parameters of higher order than vectors we use tensor contraction
        # to calculate the inner product for the thermostat.
        if ( rankTheta == 0 ) {
            dynamics$alpha[[pname]] = alpha$assign_add( u * u - stepsize )
        } else if( rankTheta >= 1 ) {
            # Declare axes for tensor contraction
            axes = matrix( rep( 0:( rankTheta - 1 ), each = 2 ), nrow = 2 )
            axes = tf$constant( axes, dtype = tf$int32 )
            dynamics$alpha[[pname]] = alpha$assign_add( 
                    tf$tensordot( u, u, axes ) / tf$size( u, out_type = tf$float32 ) - stepsize )
        }
    }
    return( dynamics )
}
