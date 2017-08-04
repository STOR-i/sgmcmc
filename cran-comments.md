## Previous submissions
* Expanded acronyms and added examples.
* Changed DESCRIPTION as per previous comment 'But it is an R package, not a library.  Pls write...'
* Conditionalized runtime code so automatic checks still run when tensorflow package not installed properly. This package needs an extra step tensorflow::install_tensorflow() to be run after install.packages("tensorflow").

## Test environments
System requirements fully met:
* Ubuntu 14.04 (on travis-ci), R 3.4.0 and R-devel
* Windows Server 2016 (Microsoft Azure Virtual Machine), R 3.4.1

TensorFlow package not properly installed with tensorflow::install_tensorflow():
* win-builder (devel)
* Ubuntu 16.04 LTS, R 3.2.3

## R CMD check restults
There were no ERRORs or WARNINGs.

There was 1 NOTE:
checking installed package size ... NOTE
  installed size is  9.4Mb
    sub-directories of 1Mb or more:
        data   8.9Mb

There is one quite large data file covtype.rda. As the package is for big data analysis this provides a more realistic vignette.


## Downstream dependencies
There is one downstream dependency: tensorflow.

* tensorflow: R CMD check: 0 ERRORs | 1 WARNING | 1 NOTE

WARNING:
checking Rd \usage sections ... WARNING
Undocumented arguments in documentation object 'unique_dir'
  ‘format’

NOTE:
checking top-level files ... NOTE
Non-standard file/directory found at top level:
  ‘pkgdown’
