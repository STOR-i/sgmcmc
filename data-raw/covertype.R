library(SparseM)
library(e1071)

# Download file from LIBSVM
temp = tempfile()
download.file("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2", temp)
# Extract and read sparse file
data = read.matrix.csr(bzfile(temp),fac=FALSE)
# Convert from sparse objects to dense
X = as.matrix(data$x)
# Change y values from 1 & 2 to 0 & 1
y = as.vector(data$y) - 1
covertype = as.matrix( cbind(y,X) )
colnames(covertype) = NULL
devtools::use_data(covertype, overwrite = TRUE)
