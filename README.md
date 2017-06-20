# sgmcmc: a stochastic gradient MCMC package for R

`sgmcmc` implements a lot of the popular stochastic gradient MCMC methods including SGLD, SGHMC and SGNHT. The package uses automatic differentiation, so all the differentiation needed for the methods is calculated automatically. Control variate methods can be used in order to improve the efficiency of the methods as proposed in the [recent publication](https://github.com/jbaker92/stochasticGradientMCMC).

The package is built on top of the [tensorflow library for R](https://tensorflow.rstudio.com/), which has a lot of support for statistical distributions and operations, which allows a large class of posteriors to be built. For more details on the [tensorflow R library](https://tensorflow.rstudio.com/) follow the link, also see the [tensorflow API](https://www.tensorflow.org/api_docs/) for full documentation.

## Installation

To start with, tensorflow for R needs to be properly installed, this requires dependencies that can't be handled by the standard `install.packages` framework, so follow the tensorflow for R library [installation instructions](https://tensorflow.rstudio.com/installation.html).

To install the `sgmcmc` library, ensure `devtools` are installed and run `devtools::install_github("jbaker92/sgmcmc")`.

## Documentation

TO BE ADDED, for now see the vignettes and man documentation for the API and example usage. `inst/doc` contains vignettes that have already been built.
