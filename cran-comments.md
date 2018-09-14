## Test environments
System requirements fully met:
* Ubuntu 14.04 (on travis-ci), R release and R-devel
* Windows Server 2016 (Microsoft Azure Virtual Machine), R 3.4.1

TensorFlow package not properly installed with `tensorflow::install_tensorflow()`:
* win-builder (devel)

## R CMD check restults
There were no ERRORs, WARNINGs or NOTEs.

## Downstream dependencies
There are two downstream dependency: tensorflow, reticulate.

* tensorflow: R CMD check: 0 ERRORs | 0 WARNING | 0 NOTE
* reticulate: R CMD check: 0 ERRORs | 0 WARNING | 0 NOTE
