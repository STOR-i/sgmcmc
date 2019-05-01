## Previous Submissions

>  Found the following (possibly) invalid URLs: https://www.kaggle.com/wiki/LogLoss

Fixed

## Test environments
System requirements fully met:
* Ubuntu 14.04 (on travis-ci), R release and R-devel
* Windows Server 2016 (Microsoft Azure Virtual Machine), R 3.4.1

TensorFlow package not properly installed with `sgmcmc::installTF()`:
* win-builder (devel)

## R CMD check restults
There were no ERRORs, WARNINGs or NOTEs.

## Downstream dependencies
There are two downstream dependencies: tensorflow, reticulate.

* tensorflow: R CMD check: 0 ERRORs | 0 WARNING | 0 NOTE
* reticulate: R CMD check: 0 ERRORs | 0 WARNING | 0 NOTE
