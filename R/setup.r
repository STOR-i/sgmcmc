library(tensorflow)

getMinibatchSize = function( minibatch_size ) {
    # Helper function to parse minibatch size if minibatch_size is list of tf$placeholders
    #
    # If minibatch_size is not a list, nothing needs to be done
    if ( typeof( minibatch_size ) != "list" ) {
        return( minibatch_size )
    }
    # Parse minibatch_size from string of Python tuple object
    batchSize = as.numeric( unlist( strsplit( gsub( "[() ]", "", as.character( 
            placeholders[[1]]$get_shape() ) ), "," ) ) )[1] 
    return( batchSize )
}

setupParams = function( params ) {
    # If needs be convert params to tensorflow variables
    #
    # If params are already tf$Variables, do nothing
    if ( typeof( params[[1]] ) == "environment" ) {
        return( params )
    }
    # Redeclare parameters as tensorflow variables
    param_names = names( params )
    tfParams = list()
    for ( pname in param_names ) {
        tfParams[[pname]] = tf$Variable( params[[pname]], dtype = tf$float32 )
    }
    return( tfParams )
}

setupPlaceholders = function( data, minibatch_size ) {
    # If needs be create placeholders
    #
    # If minibatch_size is already a list, assume the list contains placeholders and do nothing
    if ( typeof( minibatch_size ) == "list" ) {
        return( minibatch_size )
    }
    # Declare placeholders for each dataset
    data_names = names( data )
    tfPlaceholders = list()
    for ( dname in data_names ) {
        current_size = dim( data[[dname]] )
        current_size[1] = minibatch_size
        tfPlaceholders[[dname]] = tf$placeholder( tf$float32, current_size )
    }
    return( tfPlaceholders )
}

setupFullPlaceholders = function( data ) {
    # Create placeholders to hold full dataset for full log posterior calculation
    #
    # Declare placeholders for each dataset
    data_names = names( data )
    tfPlaceholders = list()
    for ( dname in data_names ) {
        current_size = dim( data[[dname]] )
        tfPlaceholders[[dname]] = tf$placeholder( tf$float32, current_size )
    }
    return( tfPlaceholders )
}

setupFullGradients = function( params ) {
    # Create container for the full gradient value after optimization
    #
    # Declare containers as tensorflow variables. Gradients will be same shape as parameters
    param_names = names( params )
    gradientContainer = list()
    for ( pname in param_names ) {
        gradientContainer[[pname]] = tf$Variable( params[[pname]], dtype = tf$float32 )
    }
    return( gradientContainer )
}

initSess = function() { 
    # Initialise tensorflow session
    sess = tf$Session()
    init = tf$global_variables_initializer()
    sess$run(init)
    return(sess)
}
