#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import sys
import math
import copy

#training data
def training_process(filename):
    instances = []
    training_set = open(filename)
    training_data = training_set.readlines()
    training_set.close()
    attributes = training_data[0].split()
    attributes.remove("Class")
    #count live instances
    count = 0
    for line in training_data[1:]:
        data = line.split()
        label = data[0]
        if label == "live":
            count += 1
        temp_list = data[1:]
        data_dict = {}
        for i in range(len(temp_list)):
            data_dict[attributes[i]] = temp_list[i]
        instances.append((label, data_dict))
    lives_prob = float(count) / float(len(instances))
    dies_prob = 1 - lives_prob
    if lives_prob>dies_prob:
        return instances, attributes, "live", lives_prob
    else:
        return instances, attributes, "die", dies_prob

#instances: the set of training instances that have been provided to the node being constructed
#attributes: the list of attributes that were not used on the path from the root to this node
#most_label: higher probability class label
#prob: probability of most_label
def BuildTree(instances, attributes, most_label, prob):
    #if instances is empty
    if len(instances) == 0:
        return Leaf(most_label, prob, 0)
    pure, label, ins_left = detect_pure(instances)
    
    #if instances are pure
    if pure == 0:
        return Leaf(label, 1, ins_left)
    
    #if attributes is empty, else find best attribute
    if len(attributes) == 0:
        prob, label, ins_left = detect_pure(instances)
        return Leaf(label, prob, ins_left)
    else:
        bestAtt, bestInstsTrue, bestInstsFalse = calculate_impurity(instances, attributes)
        new_attributes = copy.copy(attributes)
        new_attributes.remove(bestAtt)
        left = BuildTree(bestInstsTrue, new_attributes, most_label, prob)
        right = BuildTree(bestInstsFalse, new_attributes, most_label, prob)
    return Node(bestAtt, left, right)

#classify
def classifier(filename, tree):
    classify_list = []
    classify_set = open(filename)
    classify_data = classify_set.readlines()
    classify_set.close()
    attributes = classify_data[0].split()
    attributes.remove("Class")
    #count live instances
    count = 0
    for line in classify_data[1:]:
        data = line.split()
        label = data[0]
        if label == "live":
            count += 1
        temp_list = data[1:]
        data_dict = {}
        for i in range(len(temp_list)):
            data_dict[attributes[i]] = temp_list[i]
        classify_list.append((label, data_dict))
    if count > len(classify_list)/2:
        baseline = "live"
        other = "die"
    else:
        baseline = "die"
        other = "live"
    #30 test instances
    print("{} instances".format(len(classify_list)))
    node = tree
    correct = 0
    baselinecorrect = 0
    for instance in classify_list:
        while isinstance(node, Node):
            if instance[1][node.bestAtt] == "true":
                node = node.left
            else:
                node = node.right
        if instance[0] == node.label:
            correct += 1
            if node.label == baseline:
                baselinecorrect += 1
        node = tree
    accuracy = correct/len(classify_list) * 100
    baseline_accuracy = baselinecorrect/count * 100
    print("{}: {} correct out of {}".format(baseline, baselinecorrect, count))
    print("{}: {} correct out of {}".format(other, correct-baselinecorrect, len(classify_list)-count))
    print("Accuracy: {:.2f}%,".format(accuracy) + "baseline accuracy: {:.2f}%\n".format(baseline_accuracy))
    return accuracy

#calculate impurity
#find bestAtt, bestInstsTrue, bestInstsFalse
def calculate_impurity(instances, attributes):
    impurity_dict = {}
    temp = {}
    temp[0]=math.inf;
    for attribute in attributes:
        true_set = []
        false_set = []
        for instance in instances:
            if instance[1][attribute] == "true":
                true_set.append(instance)
            else:
                false_set.append(instance)
        true_weight = len(true_set)/len(instances)
        false_weight = len(false_set)/len(instances)
        true_count = 0
        false_count = 0
        if true_weight == 0:
            true_prob = 0
        else:
            for obj in true_set:
                if obj[0] == "live":
                    true_count += 1
            m = true_count/len(true_set)
            n = 1-m
            true_prob = true_weight*(2*m*n)/(m+n)**2
        if false_weight == 0:
            false_prob = 0
        else:
            for obj in false_set:
                if obj[0] == "live":
                    false_count += 1
            m = false_count/len(false_set)
            n = 1-m
            false_prob = false_weight*(2*m*n)/(m+n)**2
        impurity = true_prob+false_prob
        impurity_dict[attribute] = (impurity, true_set, false_set)
        if temp[0] >= impurity:
            temp[0] = impurity
            temp[1] = attribute
            temp[2] = true_set
            temp[3] = false_set
    bestAtt = temp[1]
    bestInstsTrue = temp[2]
    bestInstsFalse = temp[3]
    return bestAtt, bestInstsTrue, bestInstsFalse

#detect instances are pure
def detect_pure(instances):
    count = 0
    for instance in instances:
        if instance[0] == "live":
            count += 1
    if count == len(instances):
        return (0, "live", len(instances))
    elif count == 0:
        return (0, "die", len(instances))
    else:
        if count >= len(instances) - count:
            label = "live"
            prob = count/ len(instances)
            ins_left = count
        else:
            label = "die"
            prob = 1 - count/ len(instances)
            ins_left = len(instances) - count
        return prob, label, ins_left
    
#Decision Tree Node class
class Node:
    def __init__(this, bestAtt, left, right):
        this.bestAtt = bestAtt
        this.left = left
        this.right = right
    #report current node
    def report(this, space):
        print("{}{} = True\n". format(space, this.bestAtt))
        this.left.report(space+"    ")
        print("{}{} = False\n".format(space, this.bestAtt))
        this.right.report(space + "    ")
        
#Decision Tree Leaf class
class Leaf:
    def __init__(this, label, prob, ins_left):
        this.label = label
        this.prob = str(prob)
        this.ins_left = ins_left
    #report current leaf
    def report(this, space):
        print("{}Class {}, prob = {}, /{}\n".format(space, this.label, this.prob, this.ins_left))

def main():
#def main(training, test):
    #instances, attributes, label, prob = training_process(training)
    #tree.report("    ")
    #classifier(training, tree)
    #classifier(test, tree)
    
    #training data & construct binary tree
    trainingFileName = "hepatitis-training"
    instances, attributes, label, prob = training_process(trainingFileName)
    tree = BuildTree(instances, attributes, label, prob)
    
    #tree and classify
    tree.report("    ")
    print("Decision tree to training data:")
    classifier("hepatitis-training", tree)
    print("Decision tree to test data:")
    classifier("hepatitis-test", tree)
    
    #load & run files
    accuracy_sum = 0
    for i in range(0,10):
        trainingFileName = "hepatitis-training-run-" + str(i)
        testFileName = "hepatitis-test-run-" + str(i)
        print("Training set: " + trainingFileName)
        print("Test set: " + testFileName)
        
        #training data & construct binary tree
        instances, attributes, label, prob = training_process(trainingFileName)
        tree = BuildTree(instances, attributes, label, prob)

        #classify
        accuracy_sum += classifier(testFileName, tree)
    average = accuracy_sum/10
    print("Average accuracy: {:.2f}%".format(average))

if __name__ == "__main__":
    main()
    #the first argument is training set
    #training_file = sys.argv[1] if len(sys.argv) > 1 else ".\hepatitis-training"
    #the second argument is test set
    #test_file = sys.argv[2] if len(sys.argv) > 2 else ".\hepatitis-test"
    #main(training_file, test_file)

