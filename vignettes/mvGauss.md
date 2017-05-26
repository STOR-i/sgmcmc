---
title: "Multivariate Gaussian"
author: "Jack Baker"
date: "2017-05-26"
output: rmarkdown::html_vignette
vignette: >
  %\VignetteIndexEntry{Vignette Title}
  %\VignetteEngine{knitr::rmarkdown}
  %\VignetteEncoding{UTF-8}
---

# Multivariate Gaussian

In this example we use the package to infer the mean of a 2d Gaussian using [stochastic gradient Langevin dynamics](http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf). So we assume we have iid data $x_1, \dots, x_N$ with $x_i | \theta \sim N( \theta, I_2 )$, and we want to infer $\theta$.

First let's simulate the data with the following code, we set $N$ to be $10^4$

```r
library(MASS)
# Declare number of observations
N = 10^4
# Set theta to be 0 and simulate the data
theta = c( 0, 0 )
Sigma = diag(2)
X = mvrnorm( N, theta, Sigma )
```

Now we'll declare the data associated with our problem. This takes the form of a list with the name of the object and its associated value. It is used in our declaration of the log likelihood. In our problem, we have one entry `X` which has the values we just simulated.

```r
data = list( "X" = X )
```
We assume that observations are always accessed on the first dimension of each object, i.e. the point $x_i$ is located at `X[i,]` rather than `X[,i]`. Similarly the observation $i$ from a 3d object `Y` would be located at `Y[i,,]`.

The parameters are declared very similarly, but this time the value associated with each entry is its starting point. We have one parameter `theta`, which we'll just start at 0.

```r
params = list( "theta" = c( 0, 0 ) )
```

Next we'll declare the log likelihood for the problem. This is a function which takes `params` as the first input and `data` as the second. The function should calculate the log likelihood using standard tensorflow functions. Assume that `params` is a list of tensor variables each with the same names and shape as those declared in the list you made. Similarly `data` is a placeholder with the same names as the one in the list you declared. You'll want to sum over the different observations in `data`, which can be done using the tensorflow function `tf$reduce_sum`. So the likelihood function for our Gaussian example is

```r
logLik = function( params, data ) {
    # Declare Sigma as a tensorflow object
    Sigma = tf$constant( diag(2), dtype = tf$float32 )
    # Declare distribution of each observation
    baseDist = tf$contrib$distributions$MultivariateNormalFull( params$theta, Sigma )
    # Declare log likelihood function and return
    logLik = tf$reduce_sum( baseDist$log_pdf( data$X ) )
    return( logLik )
}
```
So this function basically states that our likelihood is $\sum_{i=1}^N \log \mathcal N( x_i | \theta, I_2 )$, where $\mathcal N( x | \mu, \Sigma )$ is a Gaussian density at $x$ with mean $\mu$ and variance $\Sigma$.

Next we want to define our log prior, which we assume is $\log p( \theta_j ) = \mathcal N(\theta_j | 0,10)$, for each dimension $j$ of $\theta$. As for the log likelihood definition, the log prior is defined as a function with input `params` and `data`. In our case the definition is

```r
logPrior = function( params, data ) {
    baseDist = tf$contrib$distributions$Normal( 0, 10 )
    logPrior = tf$reduce_sum( baseDist$log_pdf( params$theta ) )
    return( logPrior )
}
```

Before we begin running our SGLD algorithm, we need to specify the stepsize and minibatch size. A stepsize is required for each parameter, so this must be a list of numbers with names that are exactly the same as each of the parameters. The minibatch size is simply a number that is less than $N$. It specifies how long the SGLD algorithm takes to run, at the cost of accuracy.

```r
stepsize = list( "theta" = 1e-5 )
n = 100
```
The stepsize parameters may require a bit of tuning before you get good results.

Now we can run our SGLD algorithm using the `sgmcmc` function `sgld`, which returns a list of Markov chains for each parameter as output. Use the argument `verbose = FALSE` to hide the output of the function




