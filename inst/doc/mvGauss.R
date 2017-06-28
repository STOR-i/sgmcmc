## ------------------------------------------------------------------------
library(sgmcmc)
library(MASS)
# Declare number of observations
N = 10^4
# Set theta to be 0 and simulate the data
theta = c( 0, 0 )
Sigma = diag(2)
X = mvrnorm( N, theta, Sigma )
dataset = list("X" = X)

## ------------------------------------------------------------------------
params = list( "theta" = c( 0, 0 ) )

## ------------------------------------------------------------------------
logLik = function( params, dataset ) {
    # Declare Sigma as a Tensorflow object
    Sigma = tf$constant( c(1, 1), dtype = tf$float32 )
    # Declare distribution of each observation
    baseDist = MultivariateNormalDiag( params$theta, Sigma )
    # Declare log likelihood function and return
    logLik = tf$reduce_sum( baseDist$log_pdf( dataset$X ) )
    return( logLik )
}

## ------------------------------------------------------------------------
logPrior = function( params ) {
    baseDist = Normal( 0, 10 )
    logPrior = tf$reduce_sum( baseDist$log_pdf( params$theta ) )
    return( logPrior )
}

## ------------------------------------------------------------------------
stepsize = list( "theta" = 1e-5 )
n = 100

## ------------------------------------------------------------------------
chains = sgld( logLik, logPrior, dataset, params, stepsize, n, nIters = 10^4, verbose = FALSE )

## ------------------------------------------------------------------------
library(ggplot2)
burnIn = 10^3
thetaOut = as.data.frame( chains$theta[-c(1:burnIn),] )
ggplot( thetaOut, aes( x = V1, y = V2 ) ) +
    stat_density2d( size = 1.5 )

