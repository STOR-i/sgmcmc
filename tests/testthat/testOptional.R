# Declare constants and data for simulation from 1d gaussian
declareConsts = function() {
    testData = list()
    # Simulate data
    testData$N = 10^4
    testData$x = rnorm( testData$N )
    testData$n = 100
    testData$data = list( "x" = testData$x )
    testData$params = list( "theta" = rnorm( 1, mean = 0, sd = 10 ) )
    testData$optStepsize = 1e-1
    testData$nIters = 1100
    testData$nItersOpt = 1000
    testData$burnIn = 100
    testData$alpha = 0.001
    testData$width = 1
    return( testData )
}

logLik = function( params, data ) {
    sigma = tf$constant(1, dtype = tf$float32)
    baseDist = tf$contrib$distributions$Normal(params$theta, sigma)
    return(tf$reduce_sum(baseDist$log_prob(data$x)))
}

sgldTest = function( testData ) {
    stepsize = 1e-4
    storage = sgld( logLik, testData$data, testData$params, stepsize, nIters = testData$nIters, verbose = FALSE )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

sgldcvTest = function( testData ) {
    stepsize = 1e-4
    storage = sgldcv( logLik, testData$data, testData$params, stepsize, testData$optStepsize, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    return( storage )
}

sghmcTest = function( testData ) {
    stepsize = 1e-4
    storage = sghmc( logLik, testData$data, testData$params, stepsize, nIters = testData$nIters, verbose = FALSE )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

sghmccvTest = function( testData ) {
    stepsize = 1e-4
    storage = sghmccv( logLik, testData$data, testData$params, stepsize, testData$optStepsize, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    return( storage )
}

sgnhtTest = function( testData ) {
    stepsize = 1e-6
    storage = sgnht( logLik, testData$data, testData$params, stepsize, nIters = testData$nIters, verbose = FALSE )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

sgnhtcvTest = function( testData ) {
    stepsize = 1e-4
    storage = sgnhtcv( logLik, testData$data, testData$params, stepsize, testData$optStepsize, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    return( storage )
}

test_that( "Check SGLD optional parameters", {
    testData = declareConsts()
    thetaOut = sgldTest( testData )
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    expect_lte(confInt[1], 0)
    expect_gte(confInt[2], 0)
} )

test_that( "Check SGLDCV optional parameters", {
    testData = declareConsts()
    storage = sgldcvTest( testData )
    # Check optimization found reasonable mode
    expect_lt( storage$theta[1], 1 )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    expect_lte(confInt[1], 0)
    expect_gte(confInt[2], 0)
} )

test_that( "Check SGHMC optional parameters", {
    testData = declareConsts()
    thetaOut = sghmcTest( testData )
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    expect_lte(confInt[1], 0)
    expect_gte(confInt[2], 0)
} )

test_that( "Check SGHMCCV optional parameters", {
    testData = declareConsts()
    storage = sghmccvTest( testData )
    # Check optimization found reasonable mode
    expect_lt( storage$theta[1], 1 )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    expect_lte(confInt[1], 0)
    expect_gte(confInt[2], 0)
} )

test_that( "Check SGNHT optional parameters", {
    testData = declareConsts()
    thetaOut = sgnhtTest( testData )
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    expect_lte(confInt[1], 0)
    expect_gte(confInt[2], 0)
} )

test_that( "Check SGNHTCV optional parameters", {
    testData = declareConsts()
    storage = sgnhtcvTest( testData )
    # Check optimization found reasonable mode
    expect_lt( storage$theta[1], 1 )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    expect_lte(confInt[1], 0)
    expect_gte(confInt[2], 0)
} )
