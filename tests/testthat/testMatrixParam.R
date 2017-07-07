# Declare constants and data for simulation from 1d gaussian
declareConsts = function() {
    testData = list()
    # Simulate data
    testData$N = 10^4
    testData$Sigma = diag(2) + 0.5
    testData$X = MASS::mvrnorm( testData$N, c( 0, 0 ), testData$Sigma )
    testData$n = 100
    testData$data = list( "X" = testData$X )
    testData$params = list( "Sigma" = diag(2) )
    testData$optStepsize = 1e-1
    testData$nIters = 100
    testData$nItersOpt = 1000
    return( testData )
}

logLik = function( params, dataset ) {
    # Declare distribution of each observation
    baseDist = tf$contrib$distributions$MultivariateNormalFullCovariance( c(0, 0), params$Sigma )
    # Declare log likelihood function and return
    logLik = tf$reduce_sum( baseDist$log_prob( dataset$X ) )
    return( logLik )
}

logPrior = function( params ) {
    Sigma0 = tf$constant( diag(2), dtype = tf$float32 )
    baseDist = tf$contrib$distributions$WishartFull( 2, Sigma0 )
    logPrior = baseDist$log_prob( params$Sigma )
    return( logPrior )
}

sgldTest = function( testData ) {
    stepsize = 1e-4
    storage = sgld( logLik, testData$data, testData$params, stepsize, logPrior = logPrior, nIters = testData$nIters, verbose = FALSE )
    return( storage )
}

sgldcvTest = function( testData ) {
    stepsize = 1e-4
    storage = sgldcv( logLik, testData$data, testData$params, stepsize, testData$optStepsize, logPrior = logPrior, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    return( storage )
}

sghmcTest = function( testData ) {
    stepsize = 1e-4
    storage = sghmc( logLik, testData$data, testData$params, stepsize, logPrior = logPrior, nIters = testData$nIters, verbose = FALSE )
    return( storage )
}

sghmccvTest = function( testData ) {
    stepsize = 1e-4
    storage = sghmccv( logLik, testData$data, testData$params, stepsize, testData$optStepsize, logPrior = logPrior, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    return( storage )
}

sgnhtTest = function( testData ) {
    stepsize = 1e-6
    storage = sgnht( logLik, testData$data, testData$params, stepsize, logPrior = logPrior, nIters = testData$nIters, verbose = FALSE )
    return( storage )
}

sgnhtcvTest = function( testData ) {
    stepsize = 1e-4
    storage = sgnhtcv( logLik, testData$data, testData$params, stepsize, testData$optStepsize, logPrior = logPrior, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    return( storage )
}

test_that( "sgld: matrix parameters", {
    testData = declareConsts()
    i = sample( testData$nIters, 1 )
    output = sgldTest( testData )$Sigma[i,,]
    expect_equal( dim( output )[1], 2 )
    expect_equal( dim( output )[2], 2 )
} )

test_that( "sgldcv: matrix parameters", {
    testData = declareConsts()
    i = sample( testData$nIters, 1 )
    output = sgldcvTest( testData )$Sigma[i,,]
    expect_equal( dim( output )[1], 2 )
    expect_equal( dim( output )[2], 2 )
} )

test_that( "sghmc: matrix parameters", {
    testData = declareConsts()
    i = sample( testData$nIters, 1 )
    output = sghmcTest( testData )$Sigma[i,,]
    expect_equal( dim( output )[1], 2 )
    expect_equal( dim( output )[2], 2 )
} )

test_that( "sghmccv: matrix parameters", {
    testData = declareConsts()
    i = sample( testData$nIters, 1 )
    output = sghmccvTest( testData )$Sigma[i,,]
    expect_equal( dim( output )[1], 2 )
    expect_equal( dim( output )[2], 2 )
} )

test_that( "sgnht: matrix parameters", {
    testData = declareConsts()
    i = sample( testData$nIters, 1 )
    output = sgnhtTest( testData )$Sigma[i,,]
    expect_equal( dim( output )[1], 2 )
    expect_equal( dim( output )[2], 2 )
} )

test_that( "sgnhtcv: matrix parameters", {
    testData = declareConsts()
    i = sample( testData$nIters, 1 )
    output = sgnhtcvTest( testData )$Sigma[i,,]
    expect_equal( dim( output )[1], 2 )
    expect_equal( dim( output )[2], 2 )
} )
