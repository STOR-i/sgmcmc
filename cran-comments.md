## Previous submissions
> your have a tarball size of 17671928 bytes. Please reduce to less than 5 MB.
Done
> Can you provide some references in the 'Description' field of your DESCRIPTION file
Done
> Please replace all double spaces in your description by single spaces
Done
> Your examples are all in dontrun. Please add small executable examples
My examples need to be set to dontrun because the package relies on tensorflow which is on CRAN but has a nonstandard install step, so the examples cannot be run. I have cleared this with another maintainer, and the maintainer which flagged this.

## Test environments
System requirements fully met:
* Ubuntu 14.04 (on travis-ci), R 3.4.1 and R-devel
* Windows Server 2016 (Microsoft Azure Virtual Machine), R 3.4.1

TensorFlow package not properly installed with tensorflow::install_tensorflow():
* win-builder (devel)

## R CMD check restults
There were no ERRORs or WARNINGs.

There was 1 NOTE:
Possibly mis-spelled words in DESCRIPTION:
  Babbush (11:804)
  Fearnhead (11:957)
  Guestrin (11:748)
  Nemeth (11:982)
  Neven (11:838)
  Skeel (11:828)
  Teh (11:658)

These words are names and so not typos.

## Downstream dependencies
There is one downstream dependency: tensorflow.

* tensorflow: R CMD check: 0 ERRORs | 0 WARNING | 0 NOTE
