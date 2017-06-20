## ------------------------------------------------------------------------
library(MASS)
# Declare number of observations
N = 10^4
# Set locations of two modes, theta1 and theta2
theta1 = c( 0, 0 )
theta2 = c( 0.25, 0.25 )
# Allocate observations to each component
z = sample( 2, N, replace = TRUE, prob = c( 0.7, 0.3 ) )
# Predeclare data matrix
X = matrix( rep( NA, 2*N ), ncol = 2 )
# Simulate each observation depending on the component its been allocated
for ( i in 1:N ) {
    if ( z[i] == 1 ) {
        X[i,] = mvrnorm( 1, theta1, diag(2) )
    } else {
        X[i,] = mvrnorm( 1, theta2, diag(2) )
    }
}

## ------------------------------------------------------------------------
data = list( "X" = X )

## ------------------------------------------------------------------------
params = list( "theta1" = c( 0, 0 ), "theta2" = c( 0.25, 0.25 ) )

## ------------------------------------------------------------------------
logLik = function( params, data ) {
    # Declare Sigma as tensorflow constant (assumed known)
    Sigma = tf$constant( diag(2), dtype = tf$float32 )
    # Declare distribution of each component
    component1 = tf$contrib$distributions$MultivariateNormalFull( params$theta1, Sigma )
    component2 = tf$contrib$distributions$MultivariateNormalFull( params$theta2, Sigma )
    # Declare log likelihood
    logLik = tf$reduce_sum( tf$log( 0.7 * component1$pdf(data$X) + 0.3 * component2$pdf(data$X) ) )
    return( logLik )
}

## ------------------------------------------------------------------------
logPrior = function( params ) {
    # Declare hyperparameters mu0 and Sigma0 as tensorflow constants
    mu0 = tf$constant( c( 0, 0 ), dtype = tf$float32 )
    Sigma0 = tf$constant( 10 * diag(2), dtype = tf$float32 )
    # Declare prior distribution
    priorDistn = tf$contrib$distributions$MultivariateNormalFull( mu0, Sigma0 )
    # Declare log prior density and return
    logPrior = priorDistn$log_pdf( params$theta1 ) + priorDistn$log_pdf( params$theta2 )
    return( logPrior )
}

## ------------------------------------------------------------------------
eta = list( "theta1" = 5e-5, "theta2" = 7e-5 )
alpha = list( "theta1" = 0.1, "theta2" = 0.1 )
L = 3
minibatchSize = 200

## ------------------------------------------------------------------------
chains = sghmc( logLik, logPrior, data, params, eta, alpha, L, minibatchSize, nIters = 10^4, verbose = FALSE )

## ------------------------------------------------------------------------
library(ggplot2)
# Remove burn in
burnIn = 10^3
chains = list( "theta1" = as.data.frame( chains$theta1[-c(1:burnIn),] ),
               "theta2" = as.data.frame( chains$theta2[-c(1:burnIn),] ) )
# # Setup data for plotting
# chains$theta1$parameter = "theta1"
# chains$theta2$parameter = "theta2"
plotData = chains$theta1
ggplot( plotData, aes( x = V1, y = V2 ) ) +
    stat_density2d( size = 1.5, alpha = 0.7 )

