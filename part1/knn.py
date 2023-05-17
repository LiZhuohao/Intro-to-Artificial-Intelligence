#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import sys

#load training set and test set
def load_data(training, test):
    #training set
    training_set = open(training)
    title = training_set.readline()
    training_data_line = training_set.readlines()
    training_set.close()
    training_data_list = []
    temp1 = []
    temp2 = []
    temp3 = []
    temp4 = []
    temp5 = []
    temp6 = []
    temp7 = []
    temp8 = []
    temp9 = []
    temp10 = []
    temp11 = []
    temp12 = []
    temp13 = []
    
    #split training data
    for line in training_data_line:
        line.replace("\n", "")
        train_data = line.split()
        training_data_list.append(([float(train_data[0]), float(train_data[1]), float(train_data[2]), float(train_data[3]), 
                                    float(train_data[4]), float(train_data[5]), float(train_data[6]), float(train_data[7]), 
                                    float(train_data[8]), float(train_data[9]), float(train_data[10]), float(train_data[11]), 
                                    float(train_data[12])], train_data[13]))
        temp1.append(float(train_data[0]))
        temp2.append(float(train_data[1]))
        temp3.append(float(train_data[2]))
        temp4.append(float(train_data[3]))
        temp5.append(float(train_data[4]))
        temp6.append(float(train_data[5]))
        temp7.append(float(train_data[6]))
        temp8.append(float(train_data[7]))
        temp9.append(float(train_data[8]))
        temp10.append(float(train_data[9]))
        temp11.append(float(train_data[10]))
        temp12.append(float(train_data[11]))
        temp13.append(float(train_data[12]))
    r1 = max(temp1) - min(temp1)
    r2 = max(temp2) - min(temp2)
    r3 = max(temp3) - min(temp3)
    r4 = max(temp4) - min(temp4)
    r5 = max(temp5) - min(temp5)
    r6 = max(temp6) - min(temp6)
    r7 = max(temp7) - min(temp7)
    r8 = max(temp8) - min(temp8)
    r9 = max(temp9) - min(temp9)
    r10 = max(temp10) - min(temp10)
    r11 = max(temp11) - min(temp11)
    r12 = max(temp12) - min(temp12)
    r13 = max(temp13) - min(temp13)
    ranging = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13]
        
    # test set
    test_set = open(test)
    title = test_set.readline()
    test_data_line = test_set.readlines()
    test_data_line.pop(-1)
    test_set.close()
    test_data_list = []
    
    for line in test_data_line:
        line.replace("\n", "")
        test_data = line.split()
        test_data_list.append(([float(test_data[0]), float(test_data[1]), float(test_data[2]), float(test_data[3]), 
                                   float(test_data[4]), float(test_data[5]), float(test_data[6]), float(test_data[7]), 
                                   float(test_data[8]), float(test_data[9]), float(test_data[10]), float(test_data[11]), 
                                   float(test_data[12])], test_data[13]))
        
    return training_data_list, test_data_list, ranging

#Euclidean distance
def distance_measure(v1, v2, ranging):
    v=0
    for i in range(0,13):
        v += ((v1[i]-v2[i])/ranging[i])**2
    return v**0.5

#knn classifier
def knn_classifier(training_data_list, test_data_list, ranging, k):
    #correct prediction
    correct = 0
    #predicted labels
    label_result = []
    
    #compare each test instance with all training instances
    for test in test_data_list:
        temp = []
        for train in training_data_list:
            distance = distance_measure(train[0], test[0], ranging)
            temp.append((distance, train[1]))
        temp1 = sorted(temp, key=lambda s: s[0])
        distance_list = temp1[:k]
        label_list = []
        for i in distance_list:
            label_list.append(i[1])
            
        #predict label
        if len(label_list) == 1:
            predict_label = label_list[0]
        else:
            count1 = label_list.count("1")
            count2 = label_list.count("2")
            count3 = label_list.count("3")
            max_value = max(count1, count2, count3)
            if count1 == max_value:
                predict_label = "1"
            elif count2 == max_value:
                predict_label = "2"
            else:
                predict_label = "3"
        label_result.append(predict_label)
        
        if predict_label == test[1]:
            correct += 1
            
    print("knn predicted labels(" + "k=" + str(k) + "):\n", label_result)
    print("accuracy is " + "{:.2f}%\n".format(float(correct)/float(len(test_data_list))*100))

def main():
    #load data
    training_data_list, test_data_list, tempArrary= load_data("wine-training", "wine-test")
#def main(training, test):
    #training_data_list, test_data_list, tempArrary= load_data(training, test)
    
    #test trained knn classifier
    #k=1
    knn_classifier(training_data_list, test_data_list, tempArrary, 1)
    #k=3
    knn_classifier(training_data_list, test_data_list, tempArrary, 3)
    
if __name__ == "__main__":
    main()
    #the first argument is training set
    #training_file = sys.argv[1] if len(sys.argv) > 1 else ".\wine-training"
    #the second argument is test set
    #test_file = sys.argv[2] if len(sys.argv) > 2 else ".\wine-test"
    #main(training_file, test_file)
    

