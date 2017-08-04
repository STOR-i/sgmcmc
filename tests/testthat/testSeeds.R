logLik = function( params, data ) {
    baseDist = tf$contrib$distributions$Normal( params$theta, 1 )
    return( tf$reduce_sum( baseDist$log_prob( data$X ) ) )
}

test_that("Check setting seeds works", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    # Set precision for seeds
    precision = 1e-8
    # Build function arguments
    params = list("theta" = 0)
    dataset = list( "X" = rnorm(10^4) )
    stepsize = 1e-6
    argsStd = list( "logLik" = logLik, "dataset" = dataset, "params" = params, 
            "stepsize" = stepsize, nIters = 10, minibatchSize = 5, verbose = FALSE, seed = 1 )
    # Check standard methods
    for (method in c("sgld", "sghmc", "sgnht")) {
        output1 = do.call(method, argsStd)$theta
        output2 = do.call(method, argsStd)$theta
        expect_lte(sum(output1 - output2), precision)
    }

    # Check control variate methods after adding extra arguments
    argsStd$optStepsize = 1e-5
    argsStd$nItersOpt = 10
    for (method in c("sgldcv", "sghmccv", "sgnhtcv")) {
        output1 = do.call(method, argsStd)$theta
        output2 = do.call(method, argsStd)$theta
        expect_lte(sum(output1 - output2), precision)
    }
} )
