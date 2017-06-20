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
    testData$alpha = 0.05
    testData$width = 1
    return( testData )
}

logLik = function( params, data ) {
    sigma = tf$constant(1, dtype = tf$float32)
    baseDist = Normal(params$theta, sigma)
    return(tf$reduce_sum(baseDist$log_pdf(data$x)))
}

logPrior = function( params ) {
    return( 1 )
}

sgldTest = function( testData ) {
    stepsize = list( "theta" = 1e-4 )
    storage = sgld( logLik, logPrior, testData$data, testData$params, stepsize, testData$n, nIters = testData$nIters, verbose = FALSE )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

sgldcvTest = function( testData ) {
    stepsize = list( "theta" = 1e-4 )
    storage = sgldcv( logLik, logPrior, testData$data, testData$params, stepsize, testData$optStepsize, testData$n, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    return( storage )
}

sghmcTest = function( testData ) {
    eta = list( "theta" = 1e-5 )
    alpha = list( "theta" = 1e-1 )
    L = 3
    storage = sghmc( logLik, logPrior, testData$data, testData$params, eta, alpha, L, testData$n, nIters = testData$nIters, verbose = FALSE )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}


sghmccvTest = function( testData ) {
    eta = list( "theta" = 1e-4 )
    alpha = list( "theta" = 1e-1 )
    L = 3
    storage = sghmccv( logLik, logPrior, testData$data, testData$params, eta, alpha, L, testData$optStepsize, testData$n, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    return( storage )
}

sgnhtTest = function( testData ) {
    eta = list( "theta" = 1e-6 )
    a = list( "theta" = 1e-1 )
    storage = sgnht( logLik, logPrior, testData$data, testData$params, eta, a, testData$n, nIters = testData$nIters, verbose = FALSE )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

sgnhtcvTest = function( testData ) {
    eta = list( "theta" = 1e-5 )
    a = list( "theta" = 1e-1 )
    storage = sgnhtcv( logLik, logPrior, testData$data, testData$params, eta, a, testData$optStepsize, testData$n, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    return( storage )
}

test_that( "Check SGLD chain reasonable for 1d gauss", {
    testData = declareConsts()
    thetaOut = sgldTest( testData )
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    expect_lte(confInt[1], 0)
    expect_gte(confInt[2], 0)
    # Check width of the confidence interval is small
    expect_lte(abs( confInt[2] - confInt[1] ), testData$width)
} )

test_that( "Check SGLDCV chain reasonable for 1d gauss", {
    testData = declareConsts()
    storage = sgldcvTest( testData )
    # Check optimization found reasonable mode
    expect_lt( storage$theta[1], 1 )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    expect_lte(confInt[1], 0)
    expect_gte(confInt[2], 0)
    # Check width of the confidence interval is small
    expect_lte(abs( confInt[2] - confInt[1] ), testData$width)
} )

test_that( "Check SGHMC chain reasonable for 1d gauss", {
    testData = declareConsts()
    thetaOut = sghmcTest( testData )
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    expect_lte(confInt[1], 0)
    expect_gte(confInt[2], 0)
    # Check width of the confidence interval is small
    expect_lte(abs( confInt[2] - confInt[1] ), testData$width)
} )

test_that( "Check SGHMCCV chain reasonable for 1d gauss", {
    testData = declareConsts()
    storage = sghmccvTest( testData )
    # Check optimization found reasonable mode
    expect_lt( storage$theta[1], 1 )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    expect_lte(confInt[1], 0)
    expect_gte(confInt[2], 0)
    # Check width of the confidence interval is small
    expect_lte(abs( confInt[2] - confInt[1] ), testData$width)
} )

test_that( "Check SGNHT chain reasonable for 1d gauss", {
    testData = declareConsts()
    thetaOut = sgnhtTest( testData )
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    expect_lte(confInt[1], 0)
    expect_gte(confInt[2], 0)
    # Check width of the confidence interval is small
    expect_lte(abs( confInt[2] - confInt[1] ), testData$width)
} )

test_that( "Check SGNHTCV chain reasonable for 1d gauss", {
    testData = declareConsts()
    storage = sgnhtcvTest( testData )
    # Check optimization found reasonable mode
    expect_lt( storage$theta[1], 1 )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    expect_lte(confInt[1], 0)
    expect_gte(confInt[2], 0)
    # Check width of the confidence interval is small
    expect_lte(abs( confInt[2] - confInt[1] ), testData$width)
} )
