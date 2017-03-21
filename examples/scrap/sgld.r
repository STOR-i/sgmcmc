library(tensorflow)

# Load in data
X_train = as.matrix( read.table( "../../data/cover_type_small/X_train.dat" ) )
X_test = as.matrix( read.table( "../../data/cover_type_small/X_test.dat" ) )
y_train = as.matrix( read.table( "../../data/cover_type_small/y_train.dat" ) )
y_test = as.matrix( read.table( "../../data/cover_type_small/y_test.dat" ) )

# Declare dimensions and input
N = dim( X_train )[1]
N_test = dim( X_test )[1]
d = dim( X_train )[2]
minibatch_size = 5000
input = tf$placeholder( tf$float32, shape(minibatch_size, d) )
y_true = tf$placeholder( tf$float32, shape(minibatch_size) )

# Declare predictor
beta = tf$Variable( tf$zeros( shape(d,1) ) )
y = 1 / ( 1 + tf$exp(-tf$squeeze(tf$matmul(input,beta))) )

# Declare log prior and loglikelihood
ll = tf$reduce_sum( y_true * tf$log(y) + ( 1 - y_true ) * tf$log( 1 - y ) )
lprior = - tf$abs( beta )
correction = tf$constant( N / minibatch_size, dtype = tf$float32 )
estlpost = correction * ll + lprior
accuracy = tf$reduce_mean( tf$cast( tf$equal( y_true, tf$round(y) ), tf$float32 ) )
logloss = - ll / tf$constant( N, dtype = tf$float32 )

# Dynamics of discretized Langevin Dynamics
eps = tf$placeholder( tf$float32, shape() )
logpostgrad = tf$gradients( estlpost, beta )
beta_t = beta + 0.5 * eps * logpostgrad + tf$sqrt( eps ) * tf$random_normal( shape(d,1) )
step = beta$assign( tf$squeeze( beta_t, squeeze_dims = 0 ) )

# Declare logloss for evaluations
logloss = tf$reduce_mean( tf$multiply( y_true, tf$log(y) ) + tf$multiply( ( 1 - y_true ), tf$log( 1 - y ) ) )

# Start session
sess <- tf$Session()
init <- tf$global_variables_initializer()
sess$run(init)

for ( i in 1:1000 ) {
    eps_current = 1e-6
    # Training
    sample_current = sample( N, minibatch_size )
    X_current = X_train[sample_current,]
    y_current = y_train[sample_current]
    sess$run(step, feed_dict = dict( input = X_current, y_true = y_current, eps = eps_current ) )
    
    # Evaluate test set
    sample_current = sample( N_test, minibatch_size )
    X_current = X_test[sample_current,]
    y_current = y_test[sample_current]
    print(accuracy$eval( session=sess, feed_dict = dict( input = X_current, y_true = y_current, eps = eps_current ) ))
}
