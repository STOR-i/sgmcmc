library(tensorflow)
library(MASS)
library(ggplot2)

setwd("../../R/")
source("sgld.r")

# Declare constants
N = 10^4    # Number of obeservations
n = 100     # Minibatch size

# Simulate data -- bimodal, bivariate normal X ~  0.5 N( theta1, Sigma ) + 0.5 N( theta2, Sigma )
Sigma = diag(2)
theta1 = c( 0, 0 )
theta2 = c( 0.25, 0.25 )
z = sample( 2, N, replace = TRUE )
X = matrix( rep( NA, 2*N ), ncol = 2 )
for ( i in 1:N ) {
    if ( z[i] == 1 ) {
        X[i,] = mvrnorm( 1, theta1, Sigma )
    } else {
        X[i,] = mvrnorm( 1, theta2, Sigma )
    }
}

# Declare log likelihood
logLik = function( params, data ) {
    # Declare Sigma (assumed known), make sure has datatype float32
    Sigma = tf$constant( diag(2), dtype = tf$float32 )
    component1 = tf$contrib$distributions$MultivariateNormalFull( params$theta1, Sigma )
    component2 = tf$contrib$distributions$MultivariateNormalFull( params$theta2, Sigma )
    logLik = tf$reduce_sum( tf$log( component1$pdf( data$X ) + component2$pdf( data$X ) ) )
    return( logLik )
}

# Declare log prior ( theta_ij ~ N( 0, 10 ) )
logPrior = function( params, data ) {
    mu0 = tf$constant( c( 0, 0 ), dtype = tf$float32 )
    Sigma0 = tf$constant( 10 * diag(2), dtype = tf$float32 )
    priorDistn = tf$contrib$distributions$MultivariateNormalFull( mu0, Sigma0 )
    logPrior = priorDistn$log_pdf( params$theta1 ) + priorDistn$log_pdf( params$theta2 )
    return( logPrior )
}

# Declare main parameters
data = list( "X" = X )
params = list( "theta1" = c(0,0), "theta2" = c(0.2,0.2) )
stepsize = list( "theta1" = 1e-4, "theta2" = 1e-4 )
burnIn = 1000
nIters = 10^4 + burnIn

# Run sgld
thetaOut = runSGLD( logLik, logPrior, data, params, stepsize, n, nIters = nIters )

# Discard burn-in
thetaOut$theta1 = thetaOut$theta1[-c(1:burnIn),]
thetaOut$theta2 = thetaOut$theta2[-c(1:burnIn),]

# Concatenate chains and plot
plotData = rbind( as.data.frame( thetaOut$theta1 ), as.data.frame( thetaOut$theta2 ) )
p = ggplot( plotData, aes( x = V1, y = V2 ) ) +
    stat_density2d( size = 1.5 )
dir.create( "../plots/gaussMixture/", recursive = TRUE, showWarnings = FALSE )
ggsave( "../plots/gaussMixture/gaussMixtureSGLD.pdf" )
