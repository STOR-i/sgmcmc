# Methods for implementing Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) using Tensorflow.
# Gradients are automatically calculated. The main function is sghmc, which implements a full
# SGHMC procedure for a given model, including gradient calculation.
#
# References:
#   1. T. Chen, E.B. Fox, and C. Guestrin.  
#           Stochastic gradient Hamiltonian Monte Carlo.  
#           In Proceeding of 31st International Conference on Machine Learning (ICML’14), 2014.


library(tensorflow)

sghmc_init = function( lpost, params, eta, alpha ) {
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
sghmc_step = function( sess, step_list, data, placeholders, L ) {
    for ( step in step_list$refresh ) {
        sess$run( step )
    }
    for ( l in 1:L ) {
        for ( step in step_list$momentum ) {
            sess$run( step, feed_dict = data_feed( data, placeholders ) )
        }
        for ( step in step_list$dynamics ) {
            sess$run( step, feed_dict = data_feed( data, placeholders ) )
        }
    }
}

# Add verbose argument?
sghmc = function( lpost, data, params, placeholders, eta, alpha, L, n_iters = 10^4 ) {
    step_list = sghmc_init( lpost, params, eta, alpha )

    # Initialise tensorflow session
    sess = tf$Session()
    init = tf$global_variables_initializer()
    sess$run(init)

    # Run Langevin dynamics on each parameter for n_iters
    for ( i in 1:n_iters ) {
        sghmc_step( sess, step_list, data, placeholders, L )
        if ( i %% 10 == 0 ) {
            lpostest = sess$run( lpost, feed_dict = data_feed( data, placeholders ) )
            writeLines( paste0( "Iteration: ", i, "\t\tLog posterior estimate: ", lpostest ) )
        }
    }
}
