library(tensorflow)
library(MASS)

setwd("../R/")
source("sgld.r")
source("sghmc.r")
source("sgnht.r")

# Declare constants and data for simulation from 1d gaussian
declareConsts = function() {
    testData = list()
    # Simulate data
    testData$N = 10^4
    testData$d = 3
    testData$mu = rnorm( testData$d, sd = 5 )
    testData$Sigma = matrix( c( 1, 0.3, 0.5, 0.3, 1, 0.2, 0.5, 0.2, 1 ), ncol = 3 )
    testData$X = mvrnorm( testData$N, testData$mu, testData$Sigma )
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

test.sgld = function() {
    testData = declareConsts()
    stepsize = list( "theta" = 1e-4 )
    storage = runSGLD( logLik, logPrior, testData$data, testData$params, stepsize, testData$n, nIters = testData$nIters, verbose = FALSE )
    # Take difference between MCMC and true parameters
    paramDist = storage$theta - matrix( rep( testData$mu, each = testData$nIters ), ncol = 3 )
    # Remove burn in
    paramDist = paramDist[-c(1:testData$burnIn),]
    for ( d in 1:testData$d ) {
        # Check 0 contained within confidence interval
        confInt = quantile( paramDist[,d], c( testData$alpha, (1 - testData$alpha) ) )
        checkTrue(0 >= confInt[1] & 0 <= confInt[2])
        # Check width of the confidence interval is small
        checkTrue(sum( abs( confInt ) ) <= testData$width)
    }
}

test.sgldCV = function() {
    testData = declareConsts()
    stepsize = list( "theta" = 1e-4 )
    storage = runSGLDCV( logLik, logPrior, testData$data, testData$params, stepsize, testData$optStepsize, testData$n, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    # Check optimization found reasonable mode
    checkTrue( sum( abs( storage$theta[1,] - testData$mu ) ) < testData$modeDistance )
    # Take difference between MCMC and true parameters
    paramDist = storage$theta - matrix( rep( testData$mu, each = testData$nIters ), ncol = 3 )
    # Remove burn in
    paramDist = paramDist[-c(1:testData$burnIn),]
    for ( d in 1:testData$d ) {
        # Check 0 contained within confidence interval
        confInt = quantile( paramDist[,d], c( testData$alpha, (1 - testData$alpha) ) )
        checkTrue(0 >= confInt[1] & 0 <= confInt[2])
        # Check width of the confidence interval is small
        checkTrue(sum( abs( confInt ) ) <= testData$width)
    }
}

test.sghmc = function() {
    testData = declareConsts()
    eta = list( "theta" = 1e-5 )
    alpha = list( "theta" = 1e-1 )
    L = 3
    storage = runSGHMC( logLik, logPrior, testData$data, testData$params, eta, alpha, L, testData$n, nIters = testData$nIters, verbose = FALSE )
    # Take difference between MCMC and true parameters
    paramDist = storage$theta - matrix( rep( testData$mu, each = testData$nIters ), ncol = 3 )
    # Remove burn in
    paramDist = paramDist[-c(1:testData$burnIn),]
    for ( d in 1:testData$d ) {
        # Check 0 contained within confidence interval
        confInt = quantile( paramDist[,d], c( testData$alpha, (1 - testData$alpha) ) )
        checkTrue(0 >= confInt[1] & 0 <= confInt[2])
        # Check width of the confidence interval is small
        checkTrue(sum( abs( confInt ) ) <= testData$width)
    }
}

test.sghmcCV = function() {
    testData = declareConsts()
    eta = list( "theta" = 5e-5 )
    alpha = list( "theta" = 1e-1 )
    L = 3
    storage = runSGHMCCV( logLik, logPrior, testData$data, testData$params, eta, alpha, L, testData$optStepsize, testData$n, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    # Check optimization found reasonable mode
    checkTrue( sum( abs( storage$theta[1,] - testData$mu ) ) < testData$modeDistance )
    # Take difference between MCMC and true parameters
    paramDist = storage$theta - matrix( rep( testData$mu, each = testData$nIters ), ncol = 3 )
    # Remove burn in
    paramDist = paramDist[-c(1:testData$burnIn),]
    for ( d in 1:testData$d ) {
        # Check 0 contained within confidence interval
        confInt = quantile( paramDist[,d], c( testData$alpha, (1 - testData$alpha) ) )
        checkTrue(0 >= confInt[1] & 0 <= confInt[2])
        # Check width of the confidence interval is small
        checkTrue(sum( abs( confInt ) ) <= testData$width)
    }
}

test.sgnht = function() {
    testData = declareConsts()
    eta = list( "theta" = 1e-5 )
    a = list( "theta" = 0.9 )
    # SGNHT tends to need a good starting point to work well
    paramsSGNHT = list( "theta" = testData$mu )
    storage = runSGNHT( logLik, logPrior, testData$data, paramsSGNHT, eta, a, testData$n, nIters = testData$nIters, verbose = FALSE )
    # Take difference between MCMC and true parameters
    paramDist = storage$theta - matrix( rep( testData$mu, each = testData$nIters ), ncol = 3 )
    # Remove burn in
    paramDist = paramDist[-c(1:testData$burnIn),]
    for ( d in 1:testData$d ) {
        # Check 0 contained within confidence interval
        confInt = quantile( paramDist[,d], c( testData$alpha, (1 - testData$alpha) ) )
        checkTrue(0 >= confInt[1] & 0 <= confInt[2])
        # Check width of the confidence interval is small
        checkTrue(sum( abs( confInt ) ) <= testData$width)
    }
}


test.sgnhtCV = function() {
    testData = declareConsts()
    eta = list( "theta" = 1e-5 )
    a = list( "theta" = 9e-1 )
    storage = runSGNHTCV( logLik, logPrior, testData$data, testData$params, eta, a, testData$optStepsize, testData$n, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    # Check optimization found reasonable mode
    checkTrue( sum( abs( storage$theta[1,] - testData$mu ) ) < testData$modeDistance )
    # Take difference between MCMC and true parameters
    paramDist = storage$theta - matrix( rep( testData$mu, each = testData$nIters ), ncol = 3 )
    # Remove burn in
    paramDist = paramDist[-c(1:testData$burnIn),]
    for ( d in 1:testData$d ) {
        # Check 0 contained within confidence interval
        confInt = quantile( paramDist[,d], c( testData$alpha, (1 - testData$alpha) ) )
        checkTrue(0 >= confInt[1] & 0 <= confInt[2])
        # Check width of the confidence interval is small
        checkTrue(sum( abs( confInt ) ) <= testData$width)
    }
}
