#' Install TensorFlow and TensorFlow Probability
#'
#' Install the python packages required by sgmcmc, including TensorFlow and TensorFlow probability.
#'  Uses the tensorflow::install_tensorflow function.
#' @export
installTF = function() {
    required_pkgs = c("keras", "tensorflow-hub", "tensorflow-probability")
    install_tensorflow(extra_packages = required_pkgs)
}
