# Load TensorFlow Probability and add the contents to tf$distributions.
.onLoad <- function(libname, pkgname) {
    # Check TensorFlow is installed and load tensorflow_probability
    # If either are not installed display a custom install message
    # Set tf$distributions to be tfp$distributions
    tryCatch(tfp <- loadTF(),
            error = function(e) {
                tfMissing()
            }
    )

}


# Build message if TensorFlow Probability missing
tfMissing <- function() {
    message("\nNo TensorFlow or TensorFlow Probability python installation found.")
    message("This can be installed using the installTF() function.\n")
    # Set custom error message incase user still tries to use tf
    assign("on_error", function (e) error_fn(e), env = tf)
}


# Check TensorFlow installed and load TensorFlow probability
loadTF = function() {
    # Check tensorflow installed by doing a dummy operation that will throw an error
    temp <- tf$constant(4)

    # Delay load tensorflow_probability as tfp using reticulate package.
    tfp <- reticulate::import("tensorflow_probability", delay_load = list(
            priority = 5,
            environment = "r-tensorflow"
    ))

    # Set tfp$distributions to be tf$distributions
    tf$distributions <- tfp$distributions
}


error_fn = function(e) {
    stop(tfErrorMsg(), call. = FALSE)
}


# Build error message for TensorFlow configuration errors
tfErrorMsg <- function() {
    message <- "Installation of TensorFlow or TensorFlow Probability not found.\n"
    message <- paste0(message,
            "These can be installed by running the installTF() function.")
    return(message)
}
