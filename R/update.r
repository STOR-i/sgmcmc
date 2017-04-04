library(tensorflow)

data_feed = function( data, placeholders, minibatch_size ) {
    # Creates the data drip feed to the algorithm.
    feed_dict = dict()
    N = dim( data[[1]] )[1]
    selection = sample( N, minibatch_size )
    input_names = names( placeholders )
    for ( input in input_names ) {
        feed_dict[[ placeholders[[input]] ]] = dataSelect( data[[input]], selection )
    }
    return( feed_dict )
}

dataSelect = function( data, selection ) {
    # Subset data based on selection across general dimension containers
    dataDim = dim( data )
    d = length( dataDim )
    # Handle the vector and 1d matrix case
    if ( d < 2 ) {
        return( data[selection] )
    }
    # Create do.call expression for `[` slice operator, providing required dimensionality
    argList = list( data, selection )
    for ( i in 2:d ) {
        argList[[i+1]] = 1:dataDim[i]
    }
    argList = c( argList, list( drop = FALSE ) )
    return( do.call( `[`, argList ) )
}

feedFullDataset = function( data, placeholders ) {
    # Feeds the full dataset to the current operation
    feed_dict = dict()
    for ( input in names( placeholders ) ) {
        feed_dict[[ placeholders[[input]] ]] = data[[input]]
    }
    return( feed_dict )
}

updateSGLD = function( sess, stepList, data, placeholders, minibatch_size ) {
    # Perform one step of the declared dynamics
    feedCurr = data_feed( data, placeholders, minibatch_size )
    for ( step in stepList ) {
        sess$run( step, feed_dict = feedCurr )
    }
}

updateSGHMC = function( sess, dynamics, data, placeholders, minibatch_size, L ) {
    # Perform one step of SGHMC
    for ( pname in dynamics$momentum ) {
        sess$run( dynamics$refresh )
    }
    for ( l in 1:L ) {
        feedCurr = data_feed( data, placeholders, minibatch_size )
        for ( pname in names( dynamics$momentum ) ) {
            sess$run( dynamics$momentum[[pname]], feed_dict = feedCurr )
            sess$run( dynamics$dynamics[[pname]], feed_dict = feedCurr )
        }
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
