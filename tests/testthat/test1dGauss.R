# Declare constants and data for simulation from 1d gaussian
declareConsts = function() {
    testData = list()
    # Simulate data
    testData$N = 10^4
    testData$mu = 0
    testData$sigma = 1
    testData$X = rnorm( testData$N, testData$mu, testData$sigma )
    testData$n = 100
    testData$data = list( "X" = testData$X )
    testData$params = list( "theta" = rnorm( 1, mean = 0, sd = 1 ) )
    testData$optStepsize = 1e-1
    testData$nIters = 1100
    testData$nItersOpt = 1000
    testData$burnIn = 100
    return( testData )
}

logLik = function( params, data ) {
    baseDist = tf$contrib$distributions$Normal( params$theta, 1 )
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
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

sgldcvTest = function( testData ) {
    stepsize = list( "theta" = 1e-4 )
    storage = sgldcv( logLik, testData$data, testData$params, stepsize, testData$optStepsize, logPrior = logPrior, minibatchSize = testData$n, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    # Remove burn in
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

sghmcTest = function( testData ) {
    eta = list( "theta" = 1e-5 )
    alpha = list( "theta" = 1e-1 )
    L = 3
    storage = sghmc( logLik, testData$data, testData$params, eta, logPrior = logPrior, minibatchSize = testData$n, alpha = alpha, L = L, nIters = testData$nIters, verbose = FALSE )
    # Remove burn in
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

sghmccvTest = function( testData ) {
    eta = list( "theta" = 5e-5 )
    alpha = list( "theta" = 1e-1 )
    L = 3
    storage = sghmccv( logLik, testData$data, testData$params, eta, testData$optStepsize, logPrior = logPrior, minibatchSize = testData$n, alpha = alpha, L = L, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    # Remove burn in
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

sgnhtTest = function( testData ) {
    eta = list( "theta" = 5e-6 )
    a = list( "theta" = 1e-2 )
    # SGNHT tends to need a good starting point to work well
    sgnht = list( "theta" = testData$mu )
    storage = sgnht( logLik, testData$data, testData$params, eta, logPrior = logPrior, minibatchSize = testData$n, a = a, nIters = testData$nIters, verbose = FALSE )
    # Remove burn in
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

sgnhtcvTest = function( testData ) {
    eta = list( "theta" = 1e-4 )
    a = list( "theta" = 1e-2 )
    storage = sgnhtcv( logLik, testData$data, testData$params, eta, testData$optStepsize, logPrior = logPrior, minibatchSize = testData$n, a = a, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    # Remove burn in
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

test_that( "Check SGLD chain reasonable for 1d Gaussian", {
    testData = declareConsts()
    thetaOut = sgldTest( testData )
    # Check sample is reasonable
    expect_lte(max(thetaOut), 1)
    expect_gte(min(thetaOut), -1)
} )


test_that( "Check SGLDCV chain reasonable for 1d Gaussian", {
    testData = declareConsts()
    thetaOut = sgldcvTest( testData )
    # Check sample is reasonable
    expect_lte(max(thetaOut), 1)
    expect_gte(min(thetaOut), -1)
} )

test_that( "Check SGHMC chain reasonable for 1d Gaussian", {
    testData = declareConsts()
    thetaOut = sghmcTest( testData )
    # Check sample is reasonable
    expect_lte(max(thetaOut), 1)
    expect_gte(min(thetaOut), -1)
} )

test_that( "Check SGHMCCV chain reasonable for 1d Gaussian", {
    testData = declareConsts()
    thetaOut = sghmccvTest( testData )
    # Check sample is reasonable
    expect_lte(max(thetaOut), 1)
    expect_gte(min(thetaOut), -1)
} )

test_that( "Check SGNHT chain reasonable for 1d Gaussian", {
    testData = declareConsts()
    thetaOut = sgnhtTest( testData )
    # Check sample is reasonable
    expect_lte(max(thetaOut), 1)
    expect_gte(min(thetaOut), -1)
} )

test_that( "Check SGNHTCV chain reasonable for 1d Gaussian", {
    testData = declareConsts()
    thetaOut = sgnhtcvTest( testData )
    # Check sample is reasonable
    expect_lte(max(thetaOut), 1)
    expect_gte(min(thetaOut), -1)
} )
