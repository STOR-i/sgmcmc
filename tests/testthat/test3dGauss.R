# Declare constants and data for simulation from 1d gaussian
declareConsts = function() {
    testData = list()
    # Simulate data
    testData$N = 10^4
    testData$d = 3
    testData$mu = rnorm( testData$d, sd = 5 )
    testData$Sigma = matrix( c( 1, 0.3, 0.5, 0.3, 1, 0.2, 0.5, 0.2, 1 ), ncol = 3 )
    testData$X = MASS::mvrnorm( testData$N, testData$mu, testData$Sigma )
    testData$n = 100
    testData$data = list( "X" = testData$X )
    testData$params = list( "theta" = rnorm( 3, mean = 0, sd = 5 ) )
    testData$optStepsize = 1e-1
    testData$nIters = 1100
    testData$nItersOpt = 1000
    testData$burnIn = 100
    testData$alpha = 0.05
    testData$width = 1
    testData$modeDistance = 0.5
    return( testData )
}

logLik = function( params, data ) {
    Sigma = matrix( c( 1, 0.3, 0.5, 0.3, 1, 0.2, 0.5, 0.2, 1 ), ncol = 3 )
    Sigma = tf$constant( Sigma, dtype = tf$float32 )
    baseDist = tf$contrib$distributions$MultivariateNormalFull( params$theta, Sigma )
    return( tf$reduce_sum( baseDist$log_pdf( data$X ) ) )
}

logPrior = function( params, data ) {
    baseDist = tf$contrib$distributions$Normal( 0, 10 )
    return( tf$reduce_sum( baseDist$log_pdf( params$theta ) ) )
}

sgldTest = function( testData ) {
    stepsize = list( "theta" = 1e-4 )
    storage = sgld( logLik, logPrior, testData$data, testData$params, stepsize, testData$n, nIters = testData$nIters, verbose = FALSE )
    # Take difference between MCMC and true parameters
    paramDist = storage$theta - matrix( rep( testData$mu, each = testData$nIters ), ncol = 3 )
    # Remove burn in
    paramDist = paramDist[-c(1:testData$burnIn),]
    return( paramDist )
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
    # Take difference between MCMC and true parameters
    paramDist = storage$theta - matrix( rep( testData$mu, each = testData$nIters ), ncol = 3 )
    # Remove burn in
    paramDist = paramDist[-c(1:testData$burnIn),]
    return( paramDist )
}

sghmccvTest = function( testData ) {
    eta = list( "theta" = 5e-5 )
    alpha = list( "theta" = 1e-1 )
    L = 3
    storage = sghmccv( logLik, logPrior, testData$data, testData$params, eta, alpha, L, testData$optStepsize, testData$n, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    return( storage )
}

sgnhtTest = function( testData ) {
    eta = list( "theta" = 1e-5 )
    a = list( "theta" = 0.9 )
    # SGNHT tends to need a good starting point to work well
    sgnht = list( "theta" = testData$mu )
    storage = sgnht( logLik, logPrior, testData$data, paramsSGNHT, eta, a, testData$n, nIters = testData$nIters, verbose = FALSE )
    # Take difference between MCMC and true parameters
    paramDist = storage$theta - matrix( rep( testData$mu, each = testData$nIters ), ncol = 3 )
    # Remove burn in
    paramDist = paramDist[-c(1:testData$burnIn),]
    return( paramDist )
}

sgnhtcvTest = function( testData ) {
    eta = list( "theta" = 1e-5 )
    a = list( "theta" = 9e-1 )
    storage = sgnhtcv( logLik, logPrior, testData$data, testData$params, eta, a, testData$optStepsize, testData$n, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    return( storage )
}

test_that( "Check SGLD chain reasonable for 3d Gaussian", {
    testData = declareConsts()
    paramDist = sgldTest( testData )
    for ( d in 1:testData$d ) {
        # Check 0 contained within confidence interval
        confInt = quantile( paramDist[,d], c( testData$alpha, (1 - testData$alpha) ) )
        expect_lte(confInt[1], 0)
        expect_gte(confInt[2], 0)
        # Check width of the confidence interval is small
        expect_lte(abs( confInt[2] - confInt[1] ), testData$width)
    }
} )


test_that( "Check SGLDCV chain reasonable for 3d Gaussian", {
    testData = declareConsts()
    storage = sgldcvTest( testData )
    # Check optimization found reasonable mode
    expect_lte( sum( abs( storage$theta[1,] - testData$mu ) ), testData$modeDistance )
    # Take difference between MCMC and true parameters
    paramDist = storage$theta - matrix( rep( testData$mu, each = testData$nIters ), ncol = 3 )
    # Remove burn in
    paramDist = paramDist[-c(1:testData$burnIn),]
    for ( d in 1:testData$d ) {
        # Check 0 contained within confidence interval
        confInt = quantile( paramDist[,d], c( testData$alpha, (1 - testData$alpha) ) )
        expect_lte(confInt[1], 0)
        expect_gte(confInt[2], 0)
        # Check width of the confidence interval is small
        expect_lte(abs( confInt[2] - confInt[1] ), testData$width)
    }
} )

test_that( "Check SGHMC chain reasonable for 3d Gaussian", {
    testData = declareConsts()
    paramDist = sghmcTest( testData )
    for ( d in 1:testData$d ) {
        # Check 0 contained within confidence interval
        confInt = quantile( paramDist[,d], c( testData$alpha, (1 - testData$alpha) ) )
        expect_lte(confInt[1], 0)
        expect_gte(confInt[2], 0)
        # Check width of the confidence interval is small
        expect_lte(abs( confInt[2] - confInt[1] ), testData$width)
    }
} )

test_that( "Check SGHMCCV chain reasonable for 3d Gaussian", {
    testData = declareConsts()
    storage = sghmccvTest( testData )
    # Check optimization found reasonable mode
    expect_lte( sum( abs( storage$theta[1,] - testData$mu ) ), testData$modeDistance )
    # Take difference between MCMC and true parameters
    paramDist = storage$theta - matrix( rep( testData$mu, each = testData$nIters ), ncol = 3 )
    # Remove burn in
    paramDist = paramDist[-c(1:testData$burnIn),]
    for ( d in 1:testData$d ) {
        # Check 0 contained within confidence interval
        confInt = quantile( paramDist[,d], c( testData$alpha, (1 - testData$alpha) ) )
        expect_lte(confInt[1], 0)
        expect_gte(confInt[2], 0)
        # Check width of the confidence interval is small
        expect_lte(abs( confInt[2] - confInt[1] ), testData$width)
    }
} )

# test_that( "Check SGNHT chain reasonable for 3d Gaussian", {
#     testData = declareConsts()
#     paramDist = sgnhtTest( testData )
#     for ( d in 1:testData$d ) {
#         # Check 0 contained within confidence interval
#         confInt = quantile( paramDist[,d], c( testData$alpha, (1 - testData$alpha) ) )
#         expect_lte(confInt[1], 0)
#         expect_gte(confInt[2], 0)
#         # Check width of the confidence interval is small
#         expect_lte(abs( confInt[2] - confInt[1] ), testData$width)
#     }
# } )
# 
# test_that( "Check SGNHTCV chain reasonable for 3d Gaussian", {
#     testData = declareConsts()
#     storage = sgnhtcvTest( testData )
#     # Check optimization found reasonable mode
#     expect_lte( sum( abs( storage$theta[1,] - testData$mu ) ), testData$modeDistance )
#     # Take difference between MCMC and true parameters
#     paramDist = storage$theta - matrix( rep( testData$mu, each = testData$nIters ), ncol = 3 )
#     # Remove burn in
#     paramDist = paramDist[-c(1:testData$burnIn),]
#     for ( d in 1:testData$d ) {
#         # Check 0 contained within confidence interval
#         confInt = quantile( paramDist[,d], c( testData$alpha, (1 - testData$alpha) ) )
#         expect_lte(confInt[1], 0)
#         expect_gte(confInt[2], 0)
#         # Check width of the confidence interval is small
#         expect_lte(abs( confInt[2] - confInt[1] ), testData$width)
#     }
# } )
