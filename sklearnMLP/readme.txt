1. in terminal run the following 4 commands

①gcc -g -O -o nntrain nntrain.c backprop.c globals.c network.c patterns.c weights.c -lm -lc

②gcc -g -O -o nntest nntest.c backprop.c globals.c network.c patterns.c weights.c -lm -lc

③./nntrain digit.net digit.pat

④./nntest digit.net digit.pat weights.dat

2. The python file for scikit-learn is also working. 
In terminal run command "py Perceptron.py".
