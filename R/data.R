#' Forest Covertype data
#'
#' The samples in this dataset correspond to 30×30m patches of forest in the US, 
#' collected for the task of predicting each patch’s cover type, 
#' i.e. the dominant species of tree. 
#' We use the LIBSVM dataset, which transforms the data to a binary problem rather than multiclass.
#'
#' @format A matrix with 581012 rows and 55 variables. The first column is the classification labels, the other columns are the 54 explanatory variables.
#' @source \url{https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html}
"covertype"
