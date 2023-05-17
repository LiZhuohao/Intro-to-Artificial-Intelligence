#!/usr/bin/env python
# coding: utf-8

# In[49]:


#COMP307
#Zhuohao Li
#300452426

#read and load training set and test set
def load_data(spam_labelled, spam_unlabelled):
    #training set
    training_data = open(spam_labelled)
    training_set = training_data.readlines()
    training_data.close()
    training_list = []
    for line in training_set[:]:
        if line != "\n":
            attr = line.split()
            attributes = []
            #12 binary attributes
            for i in range(12):
                attributes.append(attr[i])
            training_list.append([attributes, attr[12]])
    
    #test set
    test_data = open(spam_unlabelled)
    test_set = test_data.readlines()
    test_data.close()
    test_list = []
    for line in test_set[:]:
        if line != "\n":
            attr = line.split()
            attributes = []
            #12 binary attributes
            for i in range(12):
                attributes.append(attr[i])
            test_list.append(attributes)

    return training_list, test_list

#calculate the probabilities P(Fi|c) for each feature i
def computing_probabilities(training_set, spam, nonspam):
    zerocounts = 0
    prob = []
    counts = []
    for i in range(12):
        st = 0 #spam(1), true(1)
        sf = 0 #spam(1), false(0)
        nt = 0 #nonspam(0), true(1)
        nf = 0 #nonspam(0), false(0)
        for instance in training_set:
            if instance[1] == "1" and instance[0][i] == "1": st += 1
            if instance[1] == "1" and instance[0][i] == "0": sf += 1
            if instance[1] == "0" and instance[0][i] == "1": nt += 1
            if instance[1] == "0" and instance[0][i] == "0": nf += 1
        if st == 0 or sf == 0 or nt == 0 or nf ==0:
            zerocounts += 1
        counts.append([st, sf, nt, nf])
        prob.append([float(st)/spam, float(sf)/spam, float(nt)/nonspam, float(nf)/nonspam])

    #when there is zero counts, add 1 to all counts and calculate the probability
    if zerocounts > 0:
        spam += 1
        nonspam += 1
        new_prob = []
        for i in counts:
            i[0] += 1
            i[1] += 1
            i[2] += 1
            i[3] += 1
            new_prob.append([float(i[0])/spam, float(i[1])/spam, float(i[2])/nonspam, float(i[3])/nonspam])
        prob = new_prob

    for i in range(12):
        print("Feature {}:".format(i+1))
        print("spam True: {}".format(prob[i][0]))#P(F = 1|C = 1)        
        print("spam False: {}".format(prob[i][1]))#P(F = 0|C = 1)        
        print("non-spam True: {}".format(prob[i][2]))#P(F = 1|C = 0)        
        print("non-spam False: {}\n".format(prob[i][3]))#P(F = 0|C = 0)print("P(F{} = 1| C = 1) = {}".format(i, prob[i][0]))
        #print("F{} 1: {:.6f} {:.6f}".format(i+1, prob[i][0], prob[i][1]))
        #print("F{} 0: {:.6f} {:.6f}\n".format(i+1, prob[i][2], prob[i][3]))
    return prob, spam, nonspam

#Naive Bayes classify
def classifier(test_set, prob, spam, nonspam):
    for instance in test_set:
        pa = spam/(spam + nonspam)
        pb = nonspam/(spam + nonspam)
        for i in range(12):
            if instance[i] == "1":
                pa *= prob[i][0]
                pb *= prob[i][2]
            else:
                pa *= prob[i][1]
                pb *= prob[i][3]
        if pa > pb:
            print("Spam: {:.6f}, Non-spam: {:.6f}, Probably: spam".format(pa, pb))
        else:
            print("Spam: {:.6f}, Non-spam: {:.6f}, Probably: non-spam".format(pa, pb))

#main function
if __name__ == "__main__":
    #parse training set and test set
    training_set, test_set = load_data("spamLabelled.dat", "spamUnlabelled.dat")

    #calculate P(C)
    spam = 0
    nonspam = 0
    for instance in training_set:
        if instance[1] == "0": nonspam += 1
        else: spam += 1

    #the probabilities P(Fi|c) for each feature i
    prob, spam, nonspam = computing_probabilities(training_set, spam, nonspam)

    #Naive Bayes classify
    classifier(test_set, prob, spam, nonspam)

