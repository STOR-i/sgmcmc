# Add the contents of tf$contrib$distributions to tf$distributions.
# This should make creating logLik and logPrior functions much cleaner!
.onLoad <- function(libname, pkgname) {
    # Change verbosity level so as not to display deprecation errors while moving objects
    defaultLogger <- tf$logging$get_verbosity()
    tf$logging$set_verbosity(tf$logging$ERROR)
    extra_distns <- names(tf$contrib$distributions)
    current_distns <- names(tf$distributions)
    for (distn in extra_distns) {
        # If the distribution name is not in tf$distributions, add it to the Module
        if (!(distn %in% current_distns)) {
            tf$distributions[[distn]] <- tf$contrib$distributions[[distn]]
        }
    }
    # Reset verbosity to standard levels
    tf$logging$set_verbosity(defaultLogger)
}
