#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy

#perceptron
def perceptron_training(instance):
    if numpy.dot(instance.features, instance.weights) > 0:
        instance.predicted_value = 1
    else :
        instance.predicted_value = 0
    for i in range(1, len(instance.weights)):
        instance.weights[i] += instance.learning_rate * (instance.classification - instance.predicted_value) * instance.features[i]
    instance.weights[0] += instance.learning_rate * (instance.classification - instance.predicted_value)

#main
def main():
    #read "dataset" file
    instances = list()
    with open("dataset") as data:
        lines = [line.rstrip() for line in data]
    #construct instances
    for line in lines[1:]:
        attributes = line.split()
        instance = Instance(int(attributes[len(attributes)-1]), numpy.zeros(len(attributes)))
        instance.features[0] = 1  #bias
        for i in range(len(attributes)-1):
            instance.features[i+1] = attributes[i]
        instance.weights = numpy.zeros(len(instance.features))
        instances.append(instance)

    #classify
    count = 0
    for instance in instances:
        for _ in range(100):
            perceptron_training(instance)
        if instance.predicted_value == instance.classification: count += 1
    print(count/len(instances)*100, "% classified correctly")

if __name__ == "__main__":    
    #construct instance
    class Instance:
        def __init__(self, classification, features):
            self.classification = classification
            self.predicted_value = None
            self.learning_rate = 0.01
            self.features = features
            self.weights = None
    main()

