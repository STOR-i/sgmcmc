# Currently have only tested this using that standard 'include' import, 
# not the more general library import
library(tensorflow)

sgld_init = function( lpost, params, stepsize ) {
    param_names = names( params )
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
sgld = function( lpost, data, params, placeholders, stepsize, n_iters = 10^4 ) {
    step_list = sgld_init( lpost, params, stepsize )

    # Initialise tensorflow session
    sess = tf$Session()
    init = tf$global_variables_initializer()
    sess$run(init)

    # Run Langevin dynamics on each parameter for n_iters
    for ( i in 1:n_iters ) {
        sgld_step( sess, step_list, data, placeholders )
        if ( i %% 10 == 0 ) {
            lpostest = sess$run( lpost, feed_dict = data_feed( data, placeholders ) )
            writeLines( paste0( "Iteration: ", i, "\t\tLog posterior estimate: ", lpostest ) )
        }
    }
}
