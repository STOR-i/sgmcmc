# Methods for implementing Stochastic Gradient Langevin Dynamics (SGLD) using Tensorflow.
# Gradients are automatically calculated. The main function is sgld, which implements a full
# SGLD procedure for a given model, including gradient calculation.
#
# References:
#   1. Welling, M. and Teh, Y. W. (2011). 
#       Bayesian learning via stochastic gradient Langevin dynamics. 
#       In Proceedings of the 28th International Conference on Machine Learning (ICML-11), 
#       pages 681–688.

library(tensorflow)
source("setup.r")
source("update.r")
source("storage.r")

declareDynamics = function( lpost, params, stepsize ) {
    # Initialize SGLD tensorflow by declaring Langevin Dynamics
    #
    param_names = names( params )
    # List contains operations consisting of one SGLD update across all parameters
    step_list = list()
    # Declare SGLD dynamics using tensorflow autodiff
    for ( param_name in param_names ) {
        param_current = params[[param_name]]
        grad = tf$gradients( lpost, param_current )[[1]]
        step_list$dynamics[[param_name]] = param_current$assign_add( 0.5 * stepsize[[param_name]] * grad + sqrt( stepsize[[param_name]] ) * tf$random_normal( param_current$get_shape() ) )
    }
    return( step_list )
}

# Add option to include a summary measure??
sgld = function( calcLogLik, calcLogPrior, data, paramsRaw, stepsize, minibatch_size, 
        n_iters = 10^4 ) {
    # 
    # Get key sizes and declare correction term for log posterior estimate
    n = getMinibatchSize( minibatch_size )
    N = dim( data[[1]] )[1]
    correction = tf$constant( N / n, dtype = tf$float32 )
    # Convert params and data to tensorflow variables and placeholders
    params = setupParams( paramsRaw )
    placeholders = setupPlaceholders( data, minibatch_size )
    paramStorage = initStorage( paramsRaw, n_iters )
    # Declare estimated log posterior tensor using declared variables and placeholders
    logLik = calcLogLik( params, placeholders )
    logPrior = calcLogPrior( params, placeholders )
    estLogPost = logPrior + correction * logLik
    # Declare SGLD dynamics
    dynamics = declareDynamics( estLogPost, params, stepsize )
    # Initalize tensorflowsession
    sess = initSess()
    # Run Langevin dynamics on each parameter for n_iters
    for ( i in 1:n_iters ) {
        updateSGLD( sess, dynamics, data, placeholders, minibatch_size )
        paramStorage = storeState( sess, i, params, paramStorage )
        if ( i %% 100 == 0 ) {
            printProgress( sess, estLogPost, data, placeholders, i, minibatch_size, params )
        }
    }
    return( paramStorage )
}
