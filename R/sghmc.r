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

declareDynamics = function( estLogPost, params, etaList, alphaList ) {
    # Initialize SGHMC for tensorflow by declaring Hamiltonian Dynamics
    #
    dynamics = list( "theta" = list(), "nu" = list(), "refresh" = list() )
    for ( pname in names( params ) ) {
        # Declare constants
        eta = etaList[[pname]]
        alpha = alphaList[[pname]]
        # Declare parameters
        theta = params[[pname]]
        nu = tf$Variable( tf$random_normal( theta$get_shape() ) )
        # Declare dynamics
        gradU = tf$gradients( estLogPost, theta )[[1]]
        dynamics$refresh[[pname]] = nu$assign( sqrt( eta ) * tf$random_normal( theta$get_shape() ) )
        dynamics$nu[[pname]] = nu$assign( eta*gradU + alpha*nu + 
                sqrt( eta * alpha / 4 ) * tf$random_normal( theta$get_shape() ) )
        dynamics$theta[[pname]] = theta$assign_add( nu )
    }
    return( dynamics )
}

updateSGHMC = function( sess, sghmc ) {
    # Perform one step of SGHMC
    for ( step in sghmc$dynamics$refresh ) {
        sess$run( step )
    }
    for ( l in 1:sghmc$L ) {
        feedCurr = data_feed( sghmc$data, sghmc$placeholders, sghmc$n )
        for ( pname in names( sghmc$params ) ) {
            sess$run( sghmc$dynamics$nu[[pname]], feed_dict = feedCurr )
            sess$run( sghmc$dynamics$theta[[pname]], feed_dict = feedCurr )
        }
    }
}

setupSGHMC = function( logLik, logPrior, data, paramsRaw, eta, alpha, L, n, gibbsParams ) {
    # 
    # Get dataset size
    N = dim( data[[1]] )[1]
    # Convert params and data to tensorflow variables and placeholders
    params = setupParams( paramsRaw )
    placeholders = setupPlaceholders( data, n )
    # Declare estimated log posterior tensor using declared variables and placeholders
    estLogPost = setupEstLogPost( logLik, logPrior, params, placeholders, N, n, gibbsParams )
    # Declare SGLD dynamics
    dynamics = declareDynamics( estLogPost, params, eta, alpha )
    sghmc = list( "dynamics" = dynamics, "data" = data, "n" = n, "placeholders" = placeholders, 
            "params" = params, "estLogPost" = estLogPost, "L" = L )
    return( sghmc )
}

runSGHMC = function( logLik, logPrior, data, paramsRaw, eta, alpha, L, n, 
        nIters = 10^4, verbose = TRUE ) {
    # Setup SGHMC dynamics
    sghmc = setupSGHMC( logLik, logPrior, data, paramsRaw, eta, alpha, L, n, NULL )
    # Initialize storage
    paramStorage = initStorage( paramsRaw, nIters )
    # Initalize tensorflowsession
    sess = initSess()
    # Perform SGHMC, storing parameters at each step
    for ( i in 1:nIters ) {
        updateSGHMC( sess, sghmc )
        paramStorage = storeState( sess, i, sghmc, paramStorage )
        if ( i %% 100 == 0 ) {
            checkDivergence( sess, sghmc, i, verbose )
        }
    }
    return( paramStorage )
}
