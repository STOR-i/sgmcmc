library(tensorflow)

# Load in data
X_train = as.matrix( read.table( "../data/cover_type/X_train.dat" ) )
X_test = as.matrix( read.table( "../data/cover_type/X_test.dat" ) )
y_train = as.matrix( read.table( "../data/cover_type/y_train.dat" ) )
y_test = as.matrix( read.table( "../data/cover_type/y_test.dat" ) )

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
ll = - tf$reduce_mean( y_true * tf$log(y) + ( 1 - y_true ) * tf$log( 1 - y ) )
accuracy = tf$reduce_mean( tf$cast( tf$equal( y_true, tf$round(y) ), tf$float32 ) )
#lprior = - tf$abs( beta )
#correction = tf$constant( N / minibatch_size, dtype = tf$float32 )
#estlpost = correction * ll + lprior

# Declare loss
optimizer <- tf$train$AdamOptimizer(0.001)
train <- optimizer$minimize(ll)

# Start session
sess <- tf$Session()
init <- tf$global_variables_initializer()
sess$run(init)

for ( i in 1:10000 ) {
    sample_current = sample( N, minibatch_size )
    X_current = X_train[sample_current,]
    y_current = y_train[sample_current]
    sess$run(train, feed_dict = dict( input = X_current, y_true = y_current ) )
    
    # Evaluate test set
    sample_current = sample( N_test, minibatch_size )
    X_current = X_test[sample_current,]
    y_current = y_test[sample_current]
    print(sess$run(accuracy, feed_dict = dict( input = X_current, y_true = y_current ) ))
}
