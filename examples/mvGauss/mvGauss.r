library(tensorflow)
library(MASS)
library(ggplot2)

setwd("../../R/")
source("sgld.r")

# Declare constants
N = 10^4    # Number of obeservations
n = 100     # Minibatch size

# Simulate data -- bivariate normal with correlation X ~ N( theta, Sigma )
theta = c( 0, 0 )
Sigma = matrix( c( 1, 0.9, 1, 0.9 ), ncol = 2 )
X = mvrnorm( N, theta, Sigma )

# Declare log likelihood
logLik = function( params, data ) {
    # Declare Sigma (assumed known), make sure has datatype float64
    Sigma = tf$constant( matrix( c( 1, 0.3, 1, 0.3 ), ncol = 2 ), dtype = tf$float64 )
    baseDist = tf$contrib$distributions$MultivariateNormalFull( params$theta, Sigma )
    logLik = tf$reduce_sum( baseDist$log_pdf( data$X ) )
    return( logLik )
}

# Declare log prior ( theta ~ N( 0, 10 ) )
logPrior = function( params, data ) {
    baseDist = tf$contrib$distributions$Normal( 0, 10 )
    logPrior = tf$reduce_sum( baseDist$log_pdf( params$theta ) )
    return( logPrior )
}

# Declare main parameters
data = list( "X" = X )
params = list( "theta" = theta )
stepsize = list( "theta" = 1e-5 )
burnIn = 1000
nIters = 10^4 + burnIn

# Run sgld
thetaOut = runSGLD( logLik, logPrior, data, params, stepsize, n, nIters = nIters )

# Discard burn-in
thetaOut = as.data.frame( thetaOut$theta[-c(1:burnIn),] )
p = ggplot( thetaOut, aes( x = V1, y = V2 ) ) +
    stat_density2d( size = 1.5 )
dir.create( "../plots/mvGauss/", recursive = TRUE, showWarnings = FALSE )
ggsave( "../plots/mvGauss/mvGaussSGLD.pdf" )
