library(tensorflow)
# Load MNIST dataset from TensorFlow datasets
tfData = tf$contrib$learn$datasets
mnistTF = tfData$mnist$read_data_sets("MNIST-data", one_hot = TRUE)
# Rebuild dataset as list so it will save properly
mnist = list() 
mnist$train = list( "images" = mnistTF$train$images, "labels" = mnistTF$train$labels )
mnist$test = list( "images" = mnistTF$test$images, "labels" = mnistTF$test$labels )
# Resave in data so the vignette won't write the dataset to the root directory
devtools::use_data(mnist, overwrite = TRUE)
# Delete downloaded files
message("Deleting temporary files")
unlink("MNIST-data", recursive = TRUE)
