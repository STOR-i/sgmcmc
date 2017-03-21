import numpy as np
import tensorflow as tf

def sgld_step( lpost ):
    params = tf.trainable_variables()
    grads = tf.gradients( lpost, tf.trainable_variables() )
    step_list = []
    for i, param in enumerate(params):
        step_list.append( param.assign( param + 0.5 * eps * grads[i] + tf.sqrt( eps ) * tf.random_normal( param.get_shape() ) ) )
    tf.group( step_list )


X_train = np.loadtxt( "../data/cover_type_small/X_train.dat" )[:,2:]
X_test = np.loadtxt( "../data/cover_type_small/X_test.dat" )[:,2:]
y_train = np.loadtxt( "../data/cover_type_small/y_train.dat" )
y_test = np.loadtxt( "../data/cover_type_small/y_test.dat" )

# Declare dimensions and input
N, d = X_train.shape
minibatch_size = 500
input = tf.placeholder( tf.float32, ( minibatch_size, d ) )
y_true = tf.placeholder( tf.float32, ( minibatch_size ) )
eps = tf.placeholder( tf.float32, () )
 
# Declare predictor
beta = tf.Variable( tf.zeros( (d,1) ) )
bias = tf.Variable( 0, dtype = tf.float32 )
y = 1 / ( 1 + tf.exp(-tf.squeeze(bias + tf.matmul(input,beta))) )

# Declare log prior and loglikelihood
ll = tf.reduce_sum( y_true * tf.log(y) + ( 1 - y_true ) * tf.log( 1 - y ) )
lprior = - tf.abs( beta )
correction = tf.constant( N / minibatch_size, dtype = tf.float32 )
estlpost = correction * ll + lprior
accuracy = tf.reduce_mean( tf.cast( tf.equal( y_true, tf.round(y) ), tf.float32 ) )
logloss = - ll / tf.constant( N, dtype = tf.float32 )

sgld_step( estlpost )
