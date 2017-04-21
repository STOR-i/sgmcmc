library(tensorflow)

setwd("../R/")
source("sgld.r")
source("sghmc.r")
source("sgnht.r")

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
    return( tf$reduce_sum( - ( data$x - params$theta )^2 / 2 ) )
}

logPrior = function( params, data ) {
    return( 1 )
}

test.sgld = function() {
    testData = declareConsts()
    stepsize = list( "theta" = 1e-4 )
    storage = runSGLD( logLik, logPrior, testData$data, testData$params, stepsize, testData$n, nIters = testData$nIters, verbose = FALSE )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    checkTrue(0 >= confInt[1] & 0 <= confInt[2])
    # Check width of the confidence interval is small
    checkTrue(sum( abs( confInt ) ) <= testData$width)
}

test.sgldCV = function() {
    testData = declareConsts()
    stepsize = list( "theta" = 1e-4 )
    storage = runSGLDCV( logLik, logPrior, testData$data, testData$params, stepsize, testData$optStepsize, testData$n, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    # Check optimization found reasonable mode
    checkTrue( storage$theta[1] < 1 )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    checkTrue(0 >= confInt[1] & 0 <= confInt[2])
    checkTrue(sum( abs( confInt ) ) <= testData$width)
}

test.sghmc = function() {
    testData = declareConsts()
    eta = list( "theta" = 1e-5 )
    alpha = list( "theta" = 1e-1 )
    L = 3
    storage = runSGHMC( logLik, logPrior, testData$data, testData$params, eta, alpha, L, testData$n, nIters = testData$nIters, verbose = FALSE )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    checkTrue(0 >= confInt[1] & 0 <= confInt[2])
    # Check width of the confidence interval is small
    checkTrue(sum( abs( confInt ) ) <= testData$width)
}

test.sghmcCV = function() {
    testData = declareConsts()
    eta = list( "theta" = 1e-4 )
    alpha = list( "theta" = 1e-1 )
    L = 3
    storage = runSGHMCCV( logLik, logPrior, testData$data, testData$params, eta, alpha, L, testData$optStepsize, testData$n, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    checkTrue(0 >= confInt[1] & 0 <= confInt[2])
    # Check width of the confidence interval is small
    checkTrue(sum( abs( confInt ) ) <= testData$width)
}

test.sgnht = function() {
    testData = declareConsts()
    eta = list( "theta" = 1e-6 )
    a = list( "theta" = 1e-1 )
    storage = runSGNHT( logLik, logPrior, testData$data, testData$params, eta, a, testData$n, nIters = testData$nIters, verbose = FALSE )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    checkTrue(0 >= confInt[1] & 0 <= confInt[2])
    # Check width of the confidence interval is small
    checkTrue(sum( abs( confInt ) ) <= testData$width)
}


test.sgnhtCV = function() {
    testData = declareConsts()
    eta = list( "theta" = 1e-5 )
    a = list( "theta" = 1e-1 )
    storage = runSGNHTCV( logLik, logPrior, testData$data, testData$params, eta, a, testData$optStepsize, testData$n, nIters = testData$nIters, nItersOpt = testData$nItersOpt, verbose = FALSE )
    thetaOut = storage$theta[-c(1:testData$burnIn)]
    # Check 0 contained within confidence interval
    confInt = quantile( thetaOut, c( testData$alpha, (1 - testData$alpha) ) )
    checkTrue(0 >= confInt[1] & 0 <= confInt[2])
    # Check width of the confidence interval is small
    checkTrue(sum( abs( confInt ) ) <= testData$width)
}
