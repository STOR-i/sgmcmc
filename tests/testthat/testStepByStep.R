# Declare constants and data for simulation from 1d gaussian
declareConsts = function() {
    testData = list()
    # Simulate data
    testData$N = 10^4
    testData$d = 3
    testData$mu = rep(0, 3)
    testData$Sigma = matrix( c( 1, 0.3, 0.5, 0.3, 1, 0.2, 0.5, 0.2, 1 ), ncol = 3 )
    testData$X = MASS::mvrnorm( testData$N, testData$mu, testData$Sigma )
    testData$n = 100
    testData$data = list( "X" = testData$X )
    testData$params = list( "theta" = rnorm( 3, mean = 0, sd = 1 ) )
    testData$optStepsize = 1e-1
    testData$nIters = 1100
    testData$nItersOpt = 1000
    testData$burnIn = 100
    return( testData )
}

logLik = function( params, data ) {
    Sigma = matrix( c( 1, 0.3, 0.5, 0.3, 1, 0.2, 0.5, 0.2, 1 ), ncol = 3 )
    Sigma = tf$constant( Sigma, dtype = tf$float32 )
    baseDist = tf$contrib$distributions$MultivariateNormalFullCovariance( params$theta, Sigma )
    return( tf$reduce_sum( baseDist$log_prob( data$X ) ) )
}

logPrior = function( params ) {
    baseDist = tf$contrib$distributions$Normal( 0, 5 )
    return( tf$reduce_sum( baseDist$log_prob( params$theta ) ) )
}

sgldTest = function( testData ) {
    stepsize = list( "theta" = 1e-4 )
    sgmcmc = sgldSetup( logLik, testData$data, testData$params, stepsize, testData$n, logPrior = logPrior )
    output = array( dim = c( testData$nIters, testData$d ) )
    sess = initSess( sgmcmc, FALSE )
    for ( i in 1:testData$nIters ) {
        sgmcmcStep( sgmcmc, sess )
        output[i,] = getParams( sgmcmc, sess )$theta
    }
    # Remove burn in
    output = output[-c(1:testData$burnIn),]
    return( output )
}

sgldcvTest = function( testData ) {
    stepsize = list( "theta" = 1e-4 )
    sgmcmc = sgldcvSetup( logLik, testData$data, testData$params, stepsize, testData$optStepsize, logPrior = logPrior, minibatchSize = testData$n, nItersOpt = testData$nItersOpt, verbose = FALSE )
    output = array( dim = c( testData$nIters, testData$d ) )
    sess = initSess( sgmcmc, FALSE )
    for ( i in 1:testData$nIters ) {
        sgmcmcStep( sgmcmc, sess )
        output[i,] = getParams( sgmcmc, sess )$theta
    }
    # Remove burn in
    output = output[-c(1:testData$burnIn),]
    return( output )
}

sghmcTest = function( testData ) {
    eta = list( "theta" = 1e-5 )
    alpha = list( "theta" = 1e-1 )
    L = 3
    sgmcmc = sghmcSetup( logLik, testData$data, testData$params, eta, logPrior = logPrior, minibatchSize = testData$n, alpha = alpha, L = L )
    output = array( dim = c( testData$nIters, testData$d ) )
    sess = initSess( sgmcmc, FALSE )
    for ( i in 1:testData$nIters ) {
        sgmcmcStep( sgmcmc, sess )
        output[i,] = getParams( sgmcmc, sess )$theta
    }
    # Remove burn in
    output = output[-c(1:testData$burnIn),]
    return( output )
}

sghmccvTest = function( testData ) {
    eta = list( "theta" = 5e-5 )
    alpha = list( "theta" = 1e-1 )
    L = 3
    sgmcmc = sghmccvSetup( logLik, testData$data, testData$params, eta, testData$optStepsize, logPrior = logPrior, minibatchSize = testData$n, alpha = alpha, L = L, nItersOpt = testData$nItersOpt, verbose = FALSE )
    output = array( dim = c( testData$nIters, testData$d ) )
    sess = initSess( sgmcmc, FALSE )
    for ( i in 1:testData$nIters ) {
        sgmcmcStep( sgmcmc, sess )
        output[i,] = getParams( sgmcmc, sess )$theta
    }
    # Remove burn in
    output = output[-c(1:testData$burnIn),]
    return( output )
}

sgnhtTest = function( testData ) {
    eta = list( "theta" = 5e-6 )
    a = list( "theta" = 1e-2 )
    # SGNHT tends to need a good starting point to work well
    sgnht = list( "theta" = testData$mu )
    sgmcmc = sgnhtSetup( logLik, testData$data, testData$params, eta, logPrior = logPrior, minibatchSize = testData$n, a = a )
    output = array( dim = c( testData$nIters, testData$d ) )
    sess = initSess( sgmcmc, FALSE )
    for ( i in 1:testData$nIters ) {
        sgmcmcStep( sgmcmc, sess )
        output[i,] = getParams( sgmcmc, sess )$theta
    }
    # Remove burn in
    output = output[-c(1:testData$burnIn),]
    return( output )
}

sgnhtcvTest = function( testData ) {
    eta = list( "theta" = 1e-4 )
    a = list( "theta" = 1e-2 )
    sgmcmc = sgnhtcvSetup( logLik, testData$data, testData$params, eta, testData$optStepsize, logPrior = logPrior, minibatchSize = testData$n, a = a, nItersOpt = testData$nItersOpt, verbose = FALSE )
    output = array( dim = c( testData$nIters, testData$d ) )
    sess = initSess( sgmcmc, FALSE )
    for ( i in 1:testData$nIters ) {
        sgmcmcStep( sgmcmc, sess )
        output[i,] = getParams( sgmcmc, sess )$theta
    }
    # Remove burn in
    output = output[-c(1:testData$burnIn),]
    return( output )
}

test_that( "Check SGLD chain step by step for 3d Gaussian", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sgldTest( testData )
    # Check true theta contained in chain
    for ( d in 1:testData$d ) {
        expect_gte(min(thetaOut[,d]), -1)
        expect_lte(max(thetaOut[,d]), 1)
    }
} )


test_that( "Check SGLDCV chain step by step for 3d Gaussian", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sgldcvTest( testData )
    # Check true theta contained in chain
    for ( d in 1:testData$d ) {
        expect_gte(min(thetaOut[,d]), -1)
        expect_lte(max(thetaOut[,d]), 1)
    }
} )

test_that( "Check SGHMC chain step by step for 3d Gaussian", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sghmcTest( testData )
    # Check true theta contained in chain
    for ( d in 1:testData$d ) {
        expect_gte(min(thetaOut[,d]), -1)
        expect_lte(max(thetaOut[,d]), 1)
    }
} )

test_that( "Check SGHMCCV chain step by step for 3d Gaussian", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sghmccvTest( testData )
    # Check true theta contained in chain
    for ( d in 1:testData$d ) {
        expect_gte(min(thetaOut[,d]), -1)
        expect_lte(max(thetaOut[,d]), 1)
    }
} )

test_that( "Check SGNHT chain step by step for 3d Gaussian", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sgnhtTest( testData )
    # Check true theta contained in chain
    for ( d in 1:testData$d ) {
        expect_gte(min(thetaOut[,d]), -1)
        expect_lte(max(thetaOut[,d]), 1)
    }
} )

test_that( "Check SGNHTCV chain step by step for 3d Gaussian", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sgnhtcvTest( testData )
    # Check true theta contained in chain
    for ( d in 1:testData$d ) {
        expect_gte(min(thetaOut[,d]), -1)
        expect_lte(max(thetaOut[,d]), 1)
    }
} )
