# sgmcmc: a stochastic gradient MCMC package for R

`sgmcmc` implements a lot of the popular stochastic gradient MCMC methods including [SGLD](http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf), [SGHMC](https://arxiv.org/pdf/1402.4102v2.pdf) and [SGNHT](http://papers.nips.cc/paper/5592-bayesian-sampling-using-stochastic-gradient-thermostats.pdf). The package uses automatic differentiation, so all the differentiation needed for the methods is calculated automatically. Control variate methods can be used in order to improve the efficiency of the methods as proposed in the [recent publication](https://github.com/jbaker92/stochasticGradientMCMC).

The package is built on top of the [tensorflow library for R](https://tensorflow.rstudio.com/), which has a lot of support for statistical distributions and operations, which allows a large class of posteriors to be built. For more details on the [tensorflow R library](https://tensorflow.rstudio.com/) follow the link, also see the [tensorflow API](https://www.tensorflow.org/api_docs/) for full documentation.

## Installation

To start with, tensorflow for R needs to be properly installed, this requires dependencies that can't be handled by the standard `install.packages` framework, so follow the tensorflow for R library [installation instructions](https://tensorflow.rstudio.com/installation.html).

To install the `sgmcmc` library, ensure `devtools` are installed and run `devtools::install_github("STOR-i/sgmcmc")`.

## Documentation

It's recommended you start [here](https://stor-i.github.io/sgmcmc///articles/sgmcmc.html). This getting started page outlines the general structure of the package and its usage.

There's also worked examples for the following models (these will be extended as the package matures)
 - [Multivariate Gaussian](https://stor-i.github.io/sgmcmc///articles/mvGauss.html)
 - [Gaussian Mixture](https://stor-i.github.io/sgmcmc///articles/gaussMixture.html)
 - [Logistic Regression](https://stor-i.github.io/sgmcmc///articles/logisticRegression.html)

Finally full details of the API can be found [here](https://stor-i.github.io/sgmcmc///reference/index.html)
