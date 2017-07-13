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

sgldTest = function( testData ) {
    stepsize = 1e-4
    storage = sgld( logLik, testData$data, testData$params, stepsize, nIters = testData$nIters, verbose = FALSE )
    # Remove burn in
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

sgldcvTest = function( testData ) {
    stepsize = 1e-4
    storage = sgldcv( logLik, testData$data, testData$params, stepsize, testData$optStepsize, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    # Remove burn in
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

sghmcTest = function( testData ) {
    stepsize = 1e-4
    storage = sghmc( logLik, testData$data, testData$params, stepsize, nIters = testData$nIters, verbose = FALSE )
    # Remove burn in
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

sghmccvTest = function( testData ) {
    stepsize = 1e-4
    storage = sghmccv( logLik, testData$data, testData$params, stepsize, testData$optStepsize, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    # Remove burn in
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

sgnhtTest = function( testData ) {
    stepsize = 1e-5
    storage = sgnht( logLik, testData$data, testData$params, stepsize, nIters = testData$nIters, verbose = FALSE )
    # Remove burn in
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

sgnhtcvTest = function( testData ) {
    stepsize = 1e-4
    storage = sgnhtcv( logLik, testData$data, testData$params, stepsize, testData$optStepsize, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    # Remove burn in
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

test_that( "Check SGLD optional parameters", {
    testData = declareConsts()
    thetaOut = sgldTest( testData )
    # Check true theta contained in chain
    expect_gte(max(thetaOut), 0)
    expect_lte(min(thetaOut), 0)
} )


test_that( "Check SGLDCV optional parameters", {
    testData = declareConsts()
    thetaOut = sgldcvTest( testData )
    # Check true theta contained in chain
    expect_gte(max(thetaOut), 0)
    expect_lte(min(thetaOut), 0)
} )

test_that( "Check SGHMC optional parameters", {
    testData = declareConsts()
    thetaOut = sghmcTest( testData )
    # Check true theta contained in chain
    expect_gte(max(thetaOut), 0)
    expect_lte(min(thetaOut), 0)
} )

test_that( "Check SGHMCCV optional parameters", {
    testData = declareConsts()
    thetaOut = sghmccvTest( testData )
    # Check true theta contained in chain
    expect_gte(max(thetaOut), 0)
    expect_lte(min(thetaOut), 0)
} )

test_that( "Check SGNHT optional parameters", {
    testData = declareConsts()
    thetaOut = sgnhtTest( testData )
    # Check true theta contained in chain
    expect_gte(max(thetaOut), 0)
    expect_lte(min(thetaOut), 0)
} )

test_that( "Check SGNHTCV optional parameters", {
    testData = declareConsts()
    thetaOut = sgnhtcvTest( testData )
    # Check true theta contained in chain
    expect_gte(max(thetaOut), 0)
    expect_lte(min(thetaOut), 0)
} )
