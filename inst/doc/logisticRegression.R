## ---- echo=FALSE---------------------------------------------------------
data("covertype")

## ------------------------------------------------------------------------
testObservations = sample(nrow(covertype), 10^4)
testSet = covertype[testObservations,]
X = covertype[-c(testObservations),2:ncol(covertype)]
y = covertype[-c(testObservations),1]
dataset = list( "X" = X, "y" = y )

## ------------------------------------------------------------------------
# Get the dimension of X, needed to set shape of params$beta
d = ncol(dataset$X)
params = list( "bias" = 0, "beta" = matrix( rep( 0, d ), nrow = d ) )

## ------------------------------------------------------------------------
logLik = function(params, dataset) {
    yEstimated = 1 / (1 + tf$exp( - tf$squeeze(params$bias + tf$matmul(dataset$X, params$beta))))
    logLik = tf$reduce_sum(dataset$y * tf$log(yEstimated) + (1 - dataset$y) * tf$log(1 - yEstimated))
    return(logLik)
}

## ------------------------------------------------------------------------
logPrior = function(params) {
    logPrior = - (tf$reduce_sum(tf$abs(params$beta)) + tf$reduce_sum(tf$abs(params$bias)))
    return(logPrior)
}

## ------------------------------------------------------------------------
stepsizesMCMC = list("beta" = 2e-5, "bias" = 2e-5)
stepsizesOptimization = 1e-1
# Set minibatch size
n = 500

## ------------------------------------------------------------------------
output = sgldcv(logLik, logPrior, dataset, params, stepsizesMCMC, stepsizesOptimization, n, nIters = 1.1e4, verbose = FALSE)

## ------------------------------------------------------------------------
yTest = testSet[,1]
XTest = testSet[,2:ncol(testSet)]
# Remove burn-in
output$bias = output$bias[-c(1:1000)]
output$beta = output$beta[-c(1:1000),,]
iterations = seq(from = 1, to = 10^4, by = 10)
avLogPred = rep(0, length(iterations))
# Calculate log predictive every 10 iterations
for ( iter in 1:length(iterations) ) {
    j = iterations[iter]
    # Get parameters at iteration j
    beta0_j = output$bias[j]
    beta_j = output$beta[j,]
    for ( i in 1:length(yTest) ) {
        pihat_ij = 1 / (1 + exp(- beta0_j - sum(XTest[i,] * beta_j)))
        y_i = yTest[i]
        # Calculate log predictive at current test set point
        LogPred_curr = - (y_i * log(pihat_ij) + (1 - y_i) * log(1 - pihat_ij))
        avLogPred[iter] = avLogPred[iter] + 1 / length(yTest) * LogPred_curr
    }
}
library(ggplot2)
plotFrame = data.frame("iteration" = iterations, "logPredictive" = avLogPred)
ggplot(plotFrame, aes(x = iteration, y = logPredictive)) +
    geom_line() +
    xlab("Average log predictive of test set")

