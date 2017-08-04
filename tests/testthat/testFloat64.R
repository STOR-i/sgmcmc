# Declare constants and data for simulation from 1d gaussian
declareConsts = function() {
    testData = list()
    # Simulate data
    testData$N = 10^4
    testData$x = rnorm( testData$N )
    testData$n = 100
    testData$data = list( "x" = testData$x )
    testData$params = list( "theta" = rnorm( 1, mean = 0, sd = 10 ) )
    testData$optStepsize = 1e-5
    testData$nIters = 1100
    testData$nItersOpt = 1000
    testData$burnIn = 100
    testData$alpha = 0.01
    testData$width = 1
    return( testData )
}

logLik = function( params, data ) {
    sigma = tf$constant( 1, dtype = tf$float64 )
    baseDist = tf$contrib$distributions$Normal(params$theta, sigma)
    return(tf$reduce_sum(baseDist$log_prob(data$x)))
}

logPrior = function( params ) {
    baseDist = tf$contrib$distributions$Normal(0, 5)
    return( baseDist$log_prob( params$theta ) )
}

sgldTest = function( testData ) {
    stepsize = list( "theta" = 1e-4 )
    storage = sgld( logLik, testData$data, testData$params, stepsize, logPrior = logPrior, minibatchSize = testData$n, nIters = testData$nIters, verbose = FALSE )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

sgldcvTest = function( testData ) {
    stepsize = list( "theta" = 1e-4 )
    storage = sgldcv( logLik, testData$data, testData$params, stepsize, testData$optStepsize, logPrior = logPrior, minibatchSize = testData$n, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    return( storage )
}

sghmcTest = function( testData ) {
    eta = list( "theta" = 1e-5 )
    alpha = list( "theta" = 1e-1 )
    L = 3
    storage = sghmc( logLik, testData$data, testData$params, eta, logPrior = logPrior, minibatchSize = testData$n, alpha = alpha, L = L, nIters = testData$nIters, verbose = FALSE )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}


sghmccvTest = function( testData ) {
    eta = list( "theta" = 1e-4 )
    alpha = list( "theta" = 1e-1 )
    L = 3
    storage = sghmccv( logLik, testData$data, testData$params, eta, testData$optStepsize, logPrior = logPrior, minibatchSize = testData$n, alpha = alpha, L = L, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    return( storage )
}

sgnhtTest = function( testData ) {
    eta = list( "theta" = 1e-6 )
    a = list( "theta" = 1e-2 )
    storage = sgnht( logLik, testData$data, testData$params, eta, logPrior = logPrior, minibatchSize = testData$n, a = a, nIters = testData$nIters, verbose = FALSE )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    return( thetaOut )
}

sgnhtcvTest = function( testData ) {
    eta = list( "theta" = 1e-5 )
    a = list( "theta" = 1e-2 )
    storage = sgnhtcv( logLik, testData$data, testData$params, eta, testData$optStepsize, logPrior = logPrior, minibatchSize = testData$n, a = a, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    return( storage )
}

test_that( "sgld: Check Error thrown for float64 input", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    expect_error( sgldTest( testData ) )
} )

test_that( "sgldcv: Check Error thrown for float64 input", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    expect_error( sgldcvTest( testData ) )
} )

test_that( "sghmc: Check Error thrown for float64 input", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    expect_error( sghmcTest( testData ) )
} )

test_that( "sghmccv: Check Error thrown for float64 input", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    expect_error( sghmccvTest( testData ) )
} )

test_that( "sgnht: Check Error thrown for float64 input", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    expect_error( sgnhtTest( testData ) )
} )

test_that( "sgnhtcv: Check Error thrown for float64 input", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    expect_error( sgnhtcvTest( testData ) )
} )
