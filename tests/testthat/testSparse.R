logLik = function(params, dataset) {
    uCurr = tf$gather(params$u, tf$to_int32((dataset$Group - 1)))
    uDistn = tf$contrib$distributions$Normal(uCurr, 1)
    logLik = tf$reduce_sum(uDistn$log_prob(dataset$X))
    return(logLik)
}

createData = function(ng, N = 10^3, seed = 13) {
    set.seed(seed)
    ng = 200
    X = c()
    alloc = c()
    for (i in 1:ng) {
        n_obs = sample(5:15, 1)
        X = c(X, rnorm(n_obs, mean = i))
        alloc = c(alloc, rep(i, n_obs))
    }
    return(list("X" = X, "Group" = alloc))
}

test_that("Check sparsity works", {
    tryCatch({
        tf$constant(c(1, 1))
    }, error = function (e) skip("tensorflow not fully built, skipping..."))
    # Build function arguments
    nGroups = 200
    params = list("u" = 1:nGroups)
    dataset = createData(nGroups)
    stepsize = 1e-6
    argsStd = list( "logLik" = logLik, "dataset" = dataset, "params" = params, 
            "stepsize" = stepsize, nIters = 10, minibatchSize = 100, verbose = FALSE, seed = 1 )
    # Check standard methods
    for (method in c("sgld", "sghmc", "sgnht")) {
        output = do.call(method, argsStd)
    }
    # Check control variate methods after adding extra arguments
    argsStd$optStepsize = 1e-5
    argsStd$nItersOpt = 10
    for (method in c("sgldcv", "sghmccv", "sgnhtcv")) {
        output = do.call(method, argsStd)
    }
} )
