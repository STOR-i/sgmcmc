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

# WILL ADAM WORK OKAY??
sgld_initial_descent = function( lpost, stepsize ) {
    optimizer = tf$train$AdamOptimizer( stepsize )
    find_mode = optimizer$maximize( lpost )
    return( find_mode )
}

sgld_init = function( lpost, params, param_opt, params_sgld, optlogpost, stepsize ) {
    # Initialize SGLD tensorflow by declaring Langevin Dynamics
    #
    # For each parameter in param, the gradient of the approximate log posterior lpost 
    # is calculated using tensorflows gradients function
    # These gradients are then used to build the dynamics of a single SGLD update, as in reference 1,
    # for each parameter in param.
    # 
    # Parameters:
    #   lpost - Tensorflow tensor, unbiased estimate of the log posterior, as in reference 1.
    #   params - an R list object, the name of this list corresponds to the name of each parameter,
    #           the value is the corresponding tensorflow tensor for that variable.
    #   stepsize - an R list object, the name of this list corresponds to the name of each parameter,
    #           the value is the corresponding stepsize for the SGLD run for that variable.
    param_names = names( params )
    # Calculate improved gradient estimate
    grads = gradEstCalc( lpost, params, param_opt, params_sgld, optlogpost )
    step_list = list()
    for ( param_name in param_names ) {
        param_current = params[[param_name]]
        grad = tf$gradients( lpost, param_current )
        step_list[[param_name]] = param_current$assign_add( 0.5 * stepsize[[param_name]] * grad[[1]] + sqrt( stepsize[[param_name]] ) * tf$random_normal( param_current$get_shape() ) )
    }
    return( step_list )
}

# Extend to multidimensional arrays??
data_feed = function( data, placeholders ) {
    # Creates the drip feed to the algorithm. Each 
    feed_dict = dict()
    # Parse minibatch size from Python tuple object
    minibatch_size = as.numeric( unlist( strsplit( gsub( "[() ]", "", as.character( 
            placeholders[[1]]$get_shape() ) ), "," ) ) )[1] 
    N = dim( data[[1]] )[1]
    selection = sample( N, minibatch_size )
    input_names = names( placeholders )
    for ( input in input_names ) {
        feed_dict[[ placeholders[[input]] ]] = data[[input]][selection,]
    }
    return( feed_dict )
}

# Perform one step of SGLD
sgld_step = function( sess, step_list, data, placeholders ) {
    for ( step in step_list ) {
        sess$run( step, feed_dict = data_feed( data, placeholders ) )
    }
}

# Currently assuming stepsize is a dictionary
# Add option to include a summary measure??
sgldCV = function( calcLogLik, calcLogPrior, data, params, placeholders, stepsize, n_iters = 10^4 ) {

    # Parse data & minibatch sizes from Python objects
    minibatch_size = as.numeric( unlist( strsplit( gsub( "[() ]", "", as.character( 
            placeholders[[1]]$get_shape() ) ), "," ) ) )[1] 
    N = dim( data[[1]] )[1]
    correction = tf$constant( N / minibatch_size, dtype = tf$float32 )

    # Declare extra variables for initial optimizer
    param_names = names( params )
    param_opt = list()
    optlogpost = list()
    for ( pname in param_names ) {
        param_opt[[pname]] = tf$Variable( params[[pname]]$initialized_value )
        optlogpost[[pname]] = tf$Variable( params[[pname]]$initialized_value )
    }

    # Create new placeholders for feeding in full dataset
    fullPlaceholders = list()
    data_names = names( data )
    for ( dname in data_names ) {
        fullPlaceholders[[dname]] = tf$placeholder( 
                placeholders[[dname]]$dtype, dim( data[[dname]] ) )
    }

    # Declare log posterior operations for SGLD and optimizer
    estLPostOpt = calcLogPrior( param_opt, placeholders ) + correction*calcLogLik( 
            param_opt, placeholders )
    estLPostSGLD = calcLogPrior( params, placeholders ) + correction*calcLogLik( 
            params, placeholders )
    
    ### Temporarily assign opt_step
    opt_step = 1e-5
    sgld_initial_descent( lpost, opt_step )
    
    # Define SGLDCV dynamics
    step_list = sgld_init( estlpost, params, params_opt, params_sgld, optlogpost, stepsize )

    # Initialise tensorflow session
    sess = tf$Session()
    init = tf$global_variables_initializer()
    sess$run(init)

    # Run Langevin dynamics on each parameter for n_iters
    for ( i in 1:n_iters ) {
        sgld_step( sess, step_list, data, placeholders )
        if ( i %% 10 == 0 ) {
            lpostest = sess$run( estlpost, feed_dict = data_feed( data, placeholders ) )
            writeLines( paste0( "Iteration: ", i, "\t\tLog posterior estimate: ", lpostest ) )
        }
    }
}
