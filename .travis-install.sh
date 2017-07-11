#!/usr/bin/env bash
R -e "install.packages('tensorflow', repos='http://cran.us.r-project.org')"
R -e "tensorflow::install_tensorflow()"
