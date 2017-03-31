library(tensorflow)

# Extend to multidimensional arrays??
data_feed = function( data, placeholders, minibatch_size ) {
    # Creates the data drip feed to the algorithm.
    feed_dict = dict()
    N = dim( data[[1]] )[1]
    selection = sample( N, minibatch_size )
    input_names = names( placeholders )
    for ( input in input_names ) {
        if ( is.null( dim( data[[input]] ) ) ) {
            feed_dict[[ placeholders[[input]] ]] = data[[input]][selection]
        } else {
            feed_dict[[ placeholders[[input]] ]] = data[[input]][selection,,drop=FALSE]
        }
    }
    return( feed_dict )
}

feedFullDataset = function( data, placeholders ) {
    # Feeds the full dataset to the current operation
    feed_dict = dict()
    for ( input in names( placeholders ) ) {
        feed_dict[[ placeholders[[input]] ]] = data[[input]]
    }
    return( feed_dict )
}

update = function( sess, dynamics, data, placeholders, minibatch_size ) {
    # Perform one step of the declared dynamics
    feedCurr = data_feed( data, placeholders, minibatch_size )
    for ( step in dynamics ) {
        sess$run( step, feed_dict = feedCurr )
    }
}

printProgress = function( sess, estLogPost, data, placeholders, iter, minibatch_size, params ) {
    # Print progress of algorithm
    currentEstimate = sess$run( estLogPost, feed_dict = data_feed( 
            data, placeholders, minibatch_size ) )
    writeLines( paste0( "Iteration: ", iter, "\t\tLog posterior estimate: ", currentEstimate ) )
}

optUpdate = function( sess, optSteps, data, placeholders, minibatch_size ) {
    # Perform one optimization step
    sess$run(optSteps$update, feed_dict = data_feed( data, placeholders, minibatch_size ) )
}

calcFullGrads = function( sess, optSteps, data, placeholders ) {
    # Calculate full gradient information at MAP estimate
    for ( pname in names( optSteps$fullCalc ) ) {
        sess$run( optSteps$fullCalc[[pname]], feed_dict = feedFullDataset( data, placeholders ) )
        sess$run( optSteps$reassign[[pname]] )
    }
}
