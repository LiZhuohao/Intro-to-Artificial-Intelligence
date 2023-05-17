COMP 307 Assignment1 Part1
KNN Method
"knn.py" and "knn.ipynb" are the program code for k-Nearest Neighbour Classifier. 
I cannot see any output at home if I use command line arguments to compile the file.  
To make sure the whole file can be compiled, command line argument code has commented in the program file. 
So please put two files in the same directory with data files. 
They are python files, compile and execute the "knn.py" under Python 3 or run "knn.ipynb" by using Jupyther Notebook.

1. 
knn predicted labels(k=1):
 ['3', '3', '3', '1', '1', '1', '1', '2', '1', '2', '2', '3', '3', '3', '1', '2', '3', '3', '1', '1', '3', '2', '2', '3', '2', '3', '2', '3', '2', '1', '2', '1', '2', '1', '2', '2', '2', '2', '2', '1', '2', '2', '3', '1', '2', '1', '3', '2', '2', '1', '3', '1', '1', '3', '3', '1', '1', '3', '1', '3', '3', '1', '2', '3', '2', '3', '3', '1', '1', '2', '1', '3', '2', '2', '1', '1', '1', '3', '1', '1', '2', '2', '3', '1', '2', '1', '1', '2']
accuracy is 94.32%
ps. You can see this output by running the program code.

2. When k equals 1, accuracy is 94.32%
When k equals 3, accuracy is 95.45%

3. Advantage:
KNN does not learn anything in the training period. It stores the training dataset and learns from it.
So this is faster than other algorithms that require training like Linear Regression. 
KNN is very easy to implement cause it only needs the value of K and the distance funcion.
Disadvantage:
KNN is not suitable in large datasets. It requires high memory because the algorithm stores all training data. 
So it could predict for long. KNN is sensitice to noise in the dataset. 
We need to manually remove missing values and outliers.

4. k-fold cross validation: 
Usually, we divide a part of the training data as validation data to evaluate the training effect of the model.  
Validation data are taken from training data, but do not participate in the model training. 
For this question, we chop the data into 5 equal subsets. For each subset, we treat it as the test set, 
the rest 4 subsets as the training set and train classifier using the training set, apply it to the test set. 
The training/test process is repeated 5 times (the folds), with each of the 5 subsets used exactly once as the test set. 
The 5 results from the folds are averaged to produce a single estimation. 

5. We use K-mean Clustering algorithm here. 
K-mean is a unsupervised learning algorithm. It doesn't need class labels. 
①Set k initial "means" randomly from the data set. 
②Create k clusters by associating every instance with the nearest mean based on a distance measure. 
③Replace the old means with the centroid of each of the k clusters (as the new means). 
④Repeat the above two steps until convergence (no change in each cluster center). 
