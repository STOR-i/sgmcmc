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
    testData$optStepsize = 1e-5
    testData$nIters = 200
    testData$nItersOpt = 100
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

test_that( "Check SGLD with optional parameters runs okay", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sgldTest( testData )
} )


test_that( "Check SGLDCV with optional parameters run okay", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sgldcvTest( testData )
} )

test_that( "Check SGHMC with optional parameters runs okay", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sghmcTest( testData )
} )

test_that( "Check SGHMCCV with optional parameters runs okay", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sghmccvTest( testData )
} )

test_that( "Check SGNHT with optional parameters runs okay", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sgnhtTest( testData )
} )

test_that( "Check SGNHTCV with optional parameters runs okay", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    testData = declareConsts()
    thetaOut = sgnhtcvTest( testData )
} )
