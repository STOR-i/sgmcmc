library(tensorflow)

# Create 100 phony x, y data points, y = x * 0.1 + 0.3
x_data <- runif( 10000, min = -5, max = 5 )
y_data = as.numeric( rnorm( 10000, mean = 0.7*x_data + 0.6, sd = 1 ) > 0 )

# Declare estimates
W <- tf$Variable(0.3)
b <- tf$Variable(0.4)
y = 1 / ( 1 + tf$exp( -( W * x_data + b) ) )
#y = tf$nn$softmax( W * x_data + b )

# Minimize the mean squared errors.
ll <- - tf$reduce_mean( y_data * tf$log( y ) + ( 1 - y_data ) * tf$log( 1 - y ) )
optimizer <- tf$train$AdamOptimizer(0.001)
train <- optimizer$minimize(ll)

# Launch the graph and initialize the variables.
sess = tf$Session()
sess$run(tf$global_variables_initializer())

# Fit the line (Learns best fit is W: 0.1, b: 0.3)
for (step in 1:100000) {
    sess$run(train)
    if ( step %% 100 == 0 ) {
        print( sess$run(W) )
    }
}
