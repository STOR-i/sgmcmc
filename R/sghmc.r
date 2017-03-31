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

declareDynamics = function( lpost, params, eta, alpha ) {
    # Initialize SGLD tensorflow by declaring Langevin Dynamics
    #
    param_names = names( params )
    step_list = list( "dynamics" = list(), "momentum" = list(), "refresh" = list() )
    vs = list()
    for ( param_name in param_names ) {
        # Declare momentum params and reparameterize
        param_current = params[[param_name]]
        vs[[param_name]] = tf$Variable( tf$random_normal( param_current$get_shape() ) )
        v_current = vs[[param_name]]
        step_list$refresh[[param_name]] = v_current$assign( sqrt( eta[[param_name]] ) * tf$random_normal( param_current$get_shape() ) )
        grad = tf$gradients( lpost, param_current )[[1]]
        step_list$momentum[[param_name]] = v_current$assign( eta[[param_name]]*grad + alpha[[param_name]]*v_current + sqrt( eta[[param_name]] * alpha[[param_name]] / 4 ) * tf$random_normal( param_current$get_shape() ) )
        step_list$dynamics[[param_name]] = param_current$assign_add( v_current )
    }
    return( step_list )
}


# Add option to include a summary measure??
sghmc = function( calcLogLik, calcLogPrior, data, paramsRaw, eta, alpha, L, minibatch_size, 
        n_iters = 10^4 ) {
    # 
    # Get key sizes and declare correction term for log posterior estimate
    n = getMinibatchSize( minibatch_size )
    N = dim( data[[1]] )[1]
    correction = tf$constant( N / minibatch_size, dtype = tf$float32 )
    # Convert params and data to tensorflow variables and placeholders
    params = setupParams( paramsRaw )
    placeholders = setupPlaceholders( data, minibatch_size )
    # Declare estimated log posterior tensor using declared variables and placeholders
    logLik = calcLogLik( params, placeholders )
    logPrior = calcLogPrior( params, placeholders )
    estLogPost = logPrior + correction * logLik
    # Declare SGLD dynamics
    dynamics = declareDynamics( estLogPost, params, eta, alpha )
    # Initalize tensorflowsession
    sess = initSess()
    # Run Langevin dynamics on each parameter for n_iters
    for ( i in 1:n_iters ) {
        updateSGHMC( sess, dynamics, data, placeholders, minibatch_size, L )
        if ( i %% 100 == 0 ) {
            printProgress( sess, estLogPost, data, placeholders, i, minibatch_size, params )
        }

    }
}
