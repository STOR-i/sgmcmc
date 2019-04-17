#' Install TensorFlow and TensorFlow Probability
#'
#' Install the python packages required by sgmcmc, including TensorFlow and TensorFlow probability.
#'  Uses the tensorflow::install_tensorflow function.
#' @export
installTF = function() {
    install_tensorflow(version = "1.13.1")
}
