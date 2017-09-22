## Test environments
System requirements fully met:
* Ubuntu 14.04 (on travis-ci), R 3.4.1 and R-devel
* Windows Server 2016 (Microsoft Azure Virtual Machine), R 3.4.1

TensorFlow package not properly installed with tensorflow::install_tensorflow():
* win-builder (devel)

## R CMD check restults
There were no ERRORs or WARNINGs.

There was 1 NOTE:
checking installed package size ... NOTE
  installed size is  9.4Mb
    sub-directories of 1Mb or more:
        data   8.9Mb

The package requires a couple of large datasets for the vignettes.

## Downstream dependencies
There is one downstream dependency: tensorflow.

* tensorflow: R CMD check: 0 ERRORs | 0 WARNING | 0 NOTE
