# sgmcmc: a stochastic gradient MCMC package for R

[![Travis-CI Build Status](https://travis-ci.org/STOR-i/sgmcmc.svg?branch=master)](https://travis-ci.org/STOR-i/sgmcmc)

`sgmcmc` implements popular stochastic gradient Markov chain Monte Carlo (SGMCMC) methods including [stochastic gradient Langevin dynamics (SGLD)](http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf), [stochastic gradient Hamiltonian Monte Carlo (SGHMC)](https://arxiv.org/pdf/1402.4102v2.pdf) and [stochastic gradient Nos&eacute;-Hoover thermostat (SGNHT)](http://papers.nips.cc/paper/5592-bayesian-sampling-using-stochastic-gradient-thermostats.pdf). The package uses automatic differentiation, so all the differentiation needed for the methods is calculated automatically. Control variate methods can be used in order to improve the efficiency of the methods as proposed in the [recent publication](https://arxiv.org/pdf/1706.05439.pdf).

The package is built on top of the [TensorFlow library for R](https://tensorflow.rstudio.com/), which has a lot of support for statistical distributions and operations, which allows a large class of posteriors to be built. For more details can be found at the [TensorFlow R library webpage](https://tensorflow.rstudio.com/), also see the [TensorFlow API](https://www.tensorflow.org/api_docs/) for full documentation.

## Installation

`sgmcmc` requires [TensorFlow for R](https://github.com/rstudio/tensorflow) to be installed, which requires packages that can't be automatically built by `R`, so has a few steps:
- Install [TensorFlow for R](https://github.com/rstudio/tensorflow), first run `install.packages("tensorflow")`, then after run `tensorflow::install_tensorflow()`.
- Now run `install.packages("sgmcmc")`.

The TensorFlow API changes a lot, so it's best to make sure your TensorFlow for python installation is up to date.

## Documentation

It's recommended you start [here](https://stor-i.github.io/sgmcmc///articles/sgmcmc.html). This getting started page outlines the general structure of the package and its usage.

There's also worked examples for the following models (these will be extended as the package matures):
 - [Multivariate Gaussian](https://stor-i.github.io/sgmcmc///articles/mvGauss.html)
 - [Gaussian Mixture](https://stor-i.github.io/sgmcmc///articles/gaussMixture.html)
 - [Logistic Regression](https://stor-i.github.io/sgmcmc///articles/logisticRegression.html)

Finally full details of the API can be found [here](https://stor-i.github.io/sgmcmc///reference/index.html).

For the source code, and bug reporting, see the [Github page](https://github.com/STOR-i/sgmcmc).

## Issues Running Examples

If you are having issues running the examples, as a first port of call please make sure your TensorFlow installation is the most up to date version. A lot of issues are simply because the TensorFlow API has changed. If you're still having issues, please file a [bug report](https://github.com/STOR-i/sgmcmc/issues).
