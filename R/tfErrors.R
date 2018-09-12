# Check tensorflow is installed. If it isn't throw an error.
checkTFInstall <- function() {
    if ( !get("TF", envir = tf_status) ) {
        stop(tfErrorMsg(), call. = FALSE)
    } else if ( !get("TFP", envir = tf_status) ) {
        stop(tfpErrorMsg(), call. = FALSE)
    }
}

# If there is an error building the posterior print a hopefully more readable error message
getPosteriorBuildError <- function(e) {
    stop(buildErrorMsg(e), call. = FALSE)
}


tfErrorMsg <- function() {
    msg <- "\nNo TensorFlow python installation found.\n"
    msg <- paste0(msg, "This can be installed using the installTF() function.\n")
    return(msg)
}


tfpErrorMsg <- function() {
    msg <- "\nNo TensorFlow Probability python installation found.\n"
    msg <- paste0(msg, "This can be installed using the installTF() function.\n")
    return(msg)
}


buildErrorMsg = function(e) {
    msg <- "Problem building log posterior estimate from supplied logLik and logPrior functions.\n\n"
    msg <- paste0(msg, "Python error output:\n", e)
    msg <- paste0(msg, "\n", 
            "Check your tensorflow code specifying the logLik and logPrior functions is correct.\n")
    msg <- paste0(msg, "Ensure constants in logLik and logPrior functions are specified as ",
            "type float32 using \ntf$constant(.., dtype = tf$float32) -- see the tutorials for some examples.")
    return(msg)
}
