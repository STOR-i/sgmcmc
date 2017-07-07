#' sgmcmc: A package for stochastic gradient MCMC
#'
#' The sgmcmc package implements some of the most popular stochastic gradient MCMC methods
#'  including SGLD, SGHMC, SGNHT. It also implements control variates as a way to increase
#'  the efficiency of these methods. The algorithms are implemented using tensorflow
#'  which means no gradients need to be specified by the user as these are calculated
#'  automatically. It also means the algorithms are efficient.
#' 
#' @section sgmcmc functions:
#' The main functions of the package are sgld, sghmc and sgnht which implement the methods
#'  stochastic gradient Langevin dynamics, stochastic gradient Hamiltonian Monte Carlo and
#'  stochastic gradient Nose-Hoover Thermostat respectively. Also included are control variate
#'  versions of these algorithms, which uses control variates to increase their efficiency.
#'  These are the functions sgldcv, sghmccv and sgnhtcv.
#'
#' @docType package
#' @import tensorflow
#' @name sgmcmc
#'
#' @references Baker, J., Fearnhead, P., Fox, E. B., & Nemeth, C. (2017) 
#'      control variates for stochastic gradient Langevin dynamics. Preprint.
#'
#' @references Welling, M., & Teh, Y. W. (2011). 
#'      Bayesian learning via stochastic gradient Langevin dynamics. ICML (pp. 681-688).
#'
#' @references Chen, T., Fox, E. B., & Guestrin, C. (2014). 
#'      stochastic gradient Hamiltonian Monte Carlo. In ICML (pp. 1683-1691).
#'
#' @references Ding, N., Fang, Y., Babbush, R., Chen, C., Skeel, R. D., & Neven, H. (2014). 
#'      Bayesian sampling using stochastic gradient thermostats. NIPS (pp. 3203-3211).
#'
NULL
