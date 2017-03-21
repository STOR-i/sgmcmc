library(tensorflow)
source("../R/sgld.r")

# Load in data
X_train = as.matrix( read.table( "../data/cover_type_small/X_train.dat" ) )[,c(-2)]
X_test = as.matrix( read.table( "../data/cover_type_small/X_test.dat" ) )[,c(-2)]
y_train = as.matrix( read.table( "../data/cover_type_small/y_train.dat" ) )
y_test = as.matrix( read.table( "../data/cover_type_small/y_test.dat" ) )

# Declare dimensions
N = dim( X_train )[1]
N_test = dim( X_test )[1]
d = dim( X_train )[2]
minibatch_size = 500

# Declare input, will be dripfed data
input = tf$placeholder( tf$float32, shape(minibatch_size, d) )
y_true = tf$placeholder( tf$float32, shape(minibatch_size) )

# Declare parameters 
beta = tf$Variable( tf$zeros( shape(d,1) ) )
bias = tf$Variable( 0, dtype = tf$float32 )

# Declare log posterior estimate
y = 1 / ( 1 + tf$exp(-tf$squeeze(bias + tf$matmul(input,beta))) )
ll = tf$reduce_sum( y_true * tf$log(y) + ( 1 - y_true ) * tf$log( 1 - y ) )
lprior = - tf$reduce_sum( tf$abs( beta ) )
correction = tf$constant( N / minibatch_size, dtype = tf$float32 )
estlpost = correction * ll + lprior

# Declare parameters and data
stepsize = list( "beta" = 1e-4, "bias" = 1e-4 )
params = list( "beta" = beta, "bias" = bias )
data = list( "input" = X_train, "y_true" = y_train )
placeholders = list( "input" = input, "y_true" = y_true )

sgld( estlpost, data, params, placeholders, stepsize, n_iters = 10^4 )
