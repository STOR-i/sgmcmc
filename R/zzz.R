# Environment determining status of the TensorFlow Installation.
# This allows a custom error message to be displayed.
tf_status <- new.env()


# Load TensorFlow Probability and add the contents to tf$distributions.
.onLoad <- function(libname, pkgname) {
    # Set default tf_status that everything is installed correctly.
    assign("TF", TRUE, envir = tf_status)
    assign("TFP", TRUE, envir = tf_status)
    # Check TensorFlow is installed. Update tf_status accordingly.
    checkTF()
    # If checkTF was not successful, return to avoid printing multiple messages
    if (!get("TF", envir = tf_status)) {
        return()
    }
    # Check TensorFlow Probability is installed, and load in. Update tf_status accordingly.
    tryCatch(loadTFP(), error = function(e) tfpMissing(e))
}


# Check tensorflow installed by doing a dummy operation that will throw an error
checkTF = function() {
    tryCatch(temp <- tf$constant(4),
            error = function (e) tfMissing())
}


# Load tensorflow probability and assign distns to tf$distributions.
# If this fails, print message and update tf_status
loadTFP <- function() {
    import_opts <- list(priority = 5, environment = "r-tensorflow")
    tfp <- reticulate::import("tensorflow_probability", delay_load = import_opts)
    tf$distributions <- tfp$distributions
}


# Build message if TensorFlow missing. Update tf_status
tfMissing <- function() {
    message("\nNo TensorFlow python installation found.")
    message("This can be installed using the installTF() function.\n")
    assign("TF", FALSE, envir = tf_status)
    assign("TFP", FALSE, envir = tf_status)
}


# Build message if TensorFlow Probability missing. Update tf_status
tfpMissing <- function(e) {
    message("\nNo TensorFlow Probability python installation found.")
    message("This can be installed using the installTF() function.\n")
    assign("TFP", FALSE, envir = tf_status)
}
