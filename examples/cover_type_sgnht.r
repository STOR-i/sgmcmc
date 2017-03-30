library(tensorflow)
source("../R/sgnht.r")

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

# Declare placeholders for input, these will be dripfed data
input = tf$placeholder( tf$float32, shape(minibatch_size, d) )
y_true = tf$placeholder( tf$float32, shape(minibatch_size) )

# Declare parameters
beta = tf$Variable( tf$random_normal( shape(d,1) ) )
bias = tf$Variable( 0, dtype = tf$float32 )

# Declare log posterior estimate -- uses tensorflow built in distributions etc
y = 1 / ( 1 + tf$exp(-tf$squeeze(bias + tf$matmul(input,beta))) )
ll = tf$reduce_sum( y_true * tf$log(y) + ( 1 - y_true ) * tf$log( 1 - y ) )
lprior = - tf$reduce_sum( tf$abs( beta ) )
estlpost = N / minibatch_size * ll + lprior # Push this into the fn
accuracy = tf$reduce_mean( tf$cast( tf$equal( y_true, tf$round(y) ), tf$float32 ) )

# Declare data and placeholders of interest
data = list( "input" = X_train, "y_true" = y_train )
# Placeholder names must correspond to data names (Can replace this, I'll explain)
placeholders = list( "input" = input, "y_true" = y_true )
# Declare parameters and respective stepsizes
params = list( "beta" = beta, "bias" = bias )
stepsizes = list( "beta" = 1e-6, "bias" = 1e-6 )

n_iters = 10^4
step_list = sgnht_init( estlpost, params, stepsizes )

# Initialise tensorflow session
sess = tf$Session()
init = tf$global_variables_initializer()
sess$run(init)

# Run Langevin dynamics on each parameter for n_iters
for ( i in 1:n_iters ) {
    sgnht_step( sess, step_list, data, placeholders )
    if ( i %% 10 == 0 ) {
        lpostest = sess$run( step_list$momentum$beta, feed_dict = data_feed( data, placeholders ) )
        writeLines( paste0( "Iteration: ", i, "\t\tLog posterior estimate: ", lpostest ) )
    }
}
