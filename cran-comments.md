## Test environments
* Ubuntu 14.04 (on travis-ci), R 3.4.0 and R-devel
* Windows Server 2016 (Microsoft Azure Virtual Machine), R 3.4.1


## R CMD check restults
There were no ERRORs or WARNINGs.

There was 1 NOTE:
checking installed package size ... NOTE
  installed size is  9.4Mb
    sub-directories of 1Mb or more:
        data   8.9Mb

There is one quite large data file covtype.rda. As the package is for big data analysis this provides one more realistic vignette.


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

tensorflow requires tensorflow::install_tensorflow() to be run after install.packages("tensorflow").
