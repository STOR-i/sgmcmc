library(tensorflow)
library(MASS)
library(ggplot2)

setwd("../../R/")
source("sgld.r")

# Declare constants
N = 10^4    # Number of obeservations
n = 100     # Minibatch size

# Simulate data -- bivariate normal with correlation X ~ N( theta, Sigma )
sigmax = 1
theta1 = 0
theta2 = 0
x = rnorm( N, theta1 + theta2^2, sigmax )

# Declare log likelihood
logLik = function( params, data ) {
    # Declare sigmax (assumed known), make sure has datatype float32
    sigmax = tf$constant( sigmax, dtype = tf$float32 )
    mu = tf$add( params$theta1, tf$square( params$theta2 ) )
    baseDist = tf$contrib$distributions$Normal( mu, sigmax )
    logLik = tf$reduce_sum( baseDist$log_pdf( data$x ) )
    return( logLik )
}

# Declare log prior ( theta ~ N( 0, 1 ) )
logPrior = function( params, data ) {
    priorDistn = tf$contrib$distributions$Normal( 0, 1 )
    logPrior = priorDistn$log_pdf( params$theta1 ) + priorDistn$log_pdf( params$theta2 )
    return( logPrior )
}

# Declare main parameters
data = list( "x" = x )
params = list( "theta1" = 0, "theta2" = 0 )
stepsize = list( "theta1" = 1e-5, "theta2" = 1e-5 )
burnIn = 1000
nIters = 10^4 + burnIn

# Run sgld
thetaOut = runSGLD( logLik, logPrior, data, params, stepsize, n, nIters = nIters )

# Discard burn-in
thetaOut = as.data.frame( cbind( thetaOut$theta1[-c(1:burnIn)] , thetaOut$theta2[c(1:burnIn)] ) )
head( thetaOut )
p = ggplot( thetaOut, aes( x = V1, y = V2 ) ) +
    stat_density2d( size = 1.5 )
dir.create( "../plots/warpedGauss/", recursive = TRUE, showWarnings = FALSE )
ggsave( "../plots/warpedGauss/warpedGauss.pdf" )
