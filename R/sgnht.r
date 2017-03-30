# Methods for implementing Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) using Tensorflow.
# Gradients are automatically calculated. The main function is sghmc, which implements a full
# SGHMC procedure for a given model, including gradient calculation.
#
# References:
#   1. T. Chen, E.B. Fox, and C. Guestrin.  
#           Stochastic gradient Hamiltonian Monte Carlo.  
#           In Proceeding of 31st International Conference on Machine Learning (ICML’14), 2014.


library(tensorflow)

sgnht_init = function( lpost, params, stepsizes ) {
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
    step_list = list( "dynamics" = list(), "momentum" = list(), "thermostat" = list() )
    vs = list()
    xis = list()
    # ADD TENSOR DOT PRODUCT!!!!!!
    for ( param_name in param_names ) {
        # Declare momentum params and reparameterize
        param_current = params[[param_name]]
        vs[[param_name]] = tf$Variable( tf$random_normal( param_current$get_shape() ) )
        v_current = vs[[param_name]]
        # Parse momentum shape from Python tuple, needed for thermostat 
        shape_curr = as.numeric( unlist( strsplit( gsub( "[() ]", "", as.character( 
                v_current$get_shape() ) ), "," ) ) )
        rank = length(shape_curr)
        xis[[param_name]] = tf$Variable( 1.0, dtype = tf$float32 )
        xi_current = xis[[param_name]]
    
        grad = tf$gradients( lpost, param_current )[[1]]
        step_list$momentum[[param_name]] = v_current$assign_add( - stepsizes[[param_name]]*( v_current * xi_current + grad ) + sqrt( 2*stepsizes[[param_name]] ) * tf$random_normal( v_current$get_shape() ) )
        step_list$dynamics[[param_name]] = param_current$assign_add( stepsizes[[param_name]] * v_current )
        if ( rank == 0 ) {
            step_list$thermostat[[param_name]] = xi_current$assign_add( v_current * v_current / tf$size(v_current, out_type=tf$float32) - 1 )
        } else {
            axes_current = tf$constant( matrix( rep( 0:(rank-1), each=2 ), nrow = 2 ), dtype = tf$int32 )
            step_list$thermostat[[param_name]] = xi_current$assign_add( stepsizes[[param_name]] * ( tf$tensordot( v_current, v_current, axes_current ) / tf$size( v_current, out_type = tf$float32 ) - 1 ) )
        }
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
sgnht_step = function( sess, step_list, data, placeholders, L ) {
    for ( step in step_list$momentum ) {
        sess$run( step, feed_dict = data_feed( data, placeholders ) )
    }
    for ( step in step_list$dynamics ) {
        sess$run( step, feed_dict = data_feed( data, placeholders ) )
    }
    for ( step in step_list$thermostat ) {
        sess$run( step, feed_dict = data_feed( data, placeholders ) )
    }
}

# Add verbose argument?
sgnht = function( lpost, data, params, placeholders, stepsizes, n_iters = 10^4 ) {
    step_list = sgnht_init( lpost, params, stepsizes )

    # Initialise tensorflow session
    sess = tf$Session()
    init = tf$global_variables_initializer()
    sess$run(init)

    # Run Langevin dynamics on each parameter for n_iters
    for ( i in 1:n_iters ) {
        sgnht_step( sess, step_list, data, placeholders )
        if ( i %% 10 == 0 ) {
            lpostest = sess$run( lpost, feed_dict = data_feed( data, placeholders ) )
            writeLines( paste0( "Iteration: ", i, "\t\tLog posterior estimate: ", lpostest ) )
        }
    }
}
