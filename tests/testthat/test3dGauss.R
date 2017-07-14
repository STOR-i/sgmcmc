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
    storage = sgld( logLik, testData$data, testData$params, stepsize, testData$n, logPrior = logPrior, nIters = testData$nIters, verbose = FALSE )
    # Remove burn in
    thetaOut = storage$theta[-c(1:testData$burnIn),]
    return( thetaOut )
}

sgldcvTest = function( testData ) {
    stepsize = list( "theta" = 1e-4 )
    storage = sgldcv( logLik, testData$data, testData$params, stepsize, testData$optStepsize, logPrior = logPrior, minibatchSize = testData$n, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    # Remove burn in
    thetaOut = storage$theta[-c(1:testData$burnIn),]
    return( thetaOut )
}

sghmcTest = function( testData ) {
    eta = list( "theta" = 1e-5 )
    alpha = list( "theta" = 1e-1 )
    L = 3
    storage = sghmc( logLik, testData$data, testData$params, eta, logPrior = logPrior, minibatchSize = testData$n, alpha = alpha, L = L, nIters = testData$nIters, verbose = FALSE )
    # Remove burn in
    thetaOut = storage$theta[-c(1:testData$burnIn),]
    return( thetaOut )
}

sghmccvTest = function( testData ) {
    eta = list( "theta" = 5e-5 )
    alpha = list( "theta" = 1e-1 )
    L = 3
    storage = sghmccv( logLik, testData$data, testData$params, eta, testData$optStepsize, logPrior = logPrior, minibatchSize = testData$n, alpha = alpha, L = L, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    # Remove burn in
    thetaOut = storage$theta[-c(1:testData$burnIn),]
    return( thetaOut )
}

sgnhtTest = function( testData ) {
    eta = list( "theta" = 5e-6 )
    a = list( "theta" = 1e-2 )
    # SGNHT tends to need a good starting point to work well
    sgnht = list( "theta" = testData$mu )
    storage = sgnht( logLik, testData$data, testData$params, eta, logPrior = logPrior, minibatchSize = testData$n, a = a, nIters = testData$nIters, verbose = FALSE )
    # Remove burn in
    thetaOut = storage$theta[-c(1:testData$burnIn),]
    return( thetaOut )
}

sgnhtcvTest = function( testData ) {
    eta = list( "theta" = 1e-4 )
    a = list( "theta" = 1e-2 )
    storage = sgnhtcv( logLik, testData$data, testData$params, eta, testData$optStepsize, logPrior = logPrior, minibatchSize = testData$n, a = a, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    # Remove burn in
    thetaOut = storage$theta[-c(1:testData$burnIn),]
    return( thetaOut )
}

test_that( "Check SGLD chain reasonable for 3d Gaussian", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sgldTest( testData )
    # Check true theta contained in chain
    for ( d in 1:testData$d ) {
        expect_gte(min(thetaOut[,d]), -1)
        expect_lte(max(thetaOut[,d]), 1)
    }
} )


test_that( "Check SGLDCV chain reasonable for 3d Gaussian", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sgldcvTest( testData )
    # Check true theta contained in chain
    for ( d in 1:testData$d ) {
        expect_gte(min(thetaOut[,d]), -1)
        expect_lte(max(thetaOut[,d]), 1)
    }
} )

test_that( "Check SGHMC chain reasonable for 3d Gaussian", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sghmcTest( testData )
    # Check true theta contained in chain
    for ( d in 1:testData$d ) {
        expect_gte(min(thetaOut[,d]), -1)
        expect_lte(max(thetaOut[,d]), 1)
    }
} )

test_that( "Check SGHMCCV chain reasonable for 3d Gaussian", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sghmccvTest( testData )
    # Check true theta contained in chain
    for ( d in 1:testData$d ) {
        expect_gte(min(thetaOut[,d]), -1)
        expect_lte(max(thetaOut[,d]), 1)
    }
} )

test_that( "Check SGNHT chain reasonable for 3d Gaussian", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sgnhtTest( testData )
    # Check true theta contained in chain
    for ( d in 1:testData$d ) {
        expect_gte(min(thetaOut[,d]), -1)
        expect_lte(max(thetaOut[,d]), 1)
    }
} )

test_that( "Check SGNHTCV chain reasonable for 3d Gaussian", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sgnhtcvTest( testData )
    # Check true theta contained in chain
    for ( d in 1:testData$d ) {
        expect_gte(min(thetaOut[,d]), -1)
        expect_lte(max(thetaOut[,d]), 1)
    }
} )
