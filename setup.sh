mkdir dataset
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P dataset/
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P dataset/
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P dataset/
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P dataset/
gzip -d dataset/*.gz

