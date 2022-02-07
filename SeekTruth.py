# SeekTruth.py : Classify text objects into two categories
#
# PLEASE PUT YOUR NAMES AND USER IDs HERE
#
# Based on skeleton code by D. Crandall, October 2021
#

import sys
import re
import math

def load_file(filename):
    objects=[]
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ',1)
            labels.append(parsed[0] if len(parsed)>0 else "")
            objects.append(parsed[1] if len(parsed)>1 else "")

    return {"objects": objects, "labels": labels, "classes": list(set(labels))}

# classifier : Train and apply a bayes net classifier
#
# This function should take a train_data dictionary that has three entries:
#        train_data["objects"] is a list of strings corresponding to reviews
#        train_data["labels"] is a list of strings corresponding to ground truth labels for each review
#        train_data["classes"] is the list of possible class names (always two)
#
# and a test_data dictionary that has objects and classes entries in the same format as above. It
# should return a list of the same length as test_data["objects"], where the i-th element of the result
# list is the estimated classlabel for test_data["objects"][i]
#
# Do not change the return type or parameters of this function!
#
def classifier(train_data, test_data):
    
    truthcount,deceptivecount=0,0
    truthwords={}
    deceptivewords={}
    
    for i in range(len(train_data["labels"])):
        
        #Spliting the sentense into words
        words=re.split("[\s\\,\\-\\/\\(\\)\\.]", train_data["objects"][i])
        
        #Building dictionary for truth statements
        if train_data["labels"][i]=="truthful":
            truthcount+=1
            
            for word in words:
                
                if word in truthwords:
                    truthwords[word]+=1
                else:
                    #Removing the spaces
                    if word=="":continue
                    truthwords[word]=1
        
        #Building Dictionary for Deceptive statements            
        else:
            deceptivecount+=1
            for word in words:
                
                if word in deceptivewords:
                    deceptivewords[word]+=1
                else:
                    #Removing the spaces
                    if word=="":continue
                    deceptivewords[word]=1
                    
   
    result=[]
    #smoothing factor
    k=1
    
    tuthfulwordscount=sum(truthwords.values())
    
    deceptivewordscount=sum(deceptivewords.values())
    
    #iterating  through the test dataset
    for i in range(len(test_data["objects"])):
        #print(words)
        T_prob=0
        D_prob=0
        words=re.split("[\s\\,\\-\\/\\(\\)\\.]", test_data["objects"][i])
        for word in words:
            #Removing the spaces 
            if word=="":continue
            
            #Calculating bayseian for known true words with smoothing 
            try:
                T_prob=T_prob+math.log(((truthwords[word]+k)/(tuthfulwordscount+(len(truthwords)*k))))
            
            #Calculating bayseian for unknown true words with smoothing  
            except:
                T_prob=T_prob+math.log((k/(tuthfulwordscount+(len(truthwords)*k))))
            
            #Calculating bayseian for known Deceptive words with smoothing   
            try:
                 
                D_prob=D_prob+math.log(((deceptivewords[word]+k)/(deceptivewordscount+(len(deceptivewords)*k))))
            
            #Calculating bayseian for unknown Deceptive words with smoothing     
            except:
                D_prob=D_prob+math.log((k/(deceptivewordscount+(len(deceptivewords)*k))))
        
        #Calculating bayseian for base conditions p(truth) and p(deceptive)        
        T_prob=T_prob+math.log(((truthcount+k)/(truthcount+deceptivecount+(2*k))))
        D_prob=D_prob+math.log(((deceptivecount+k)/(truthcount+deceptivecount+(2*k))))
        
        #Comparing Probabilities of truthful and deceptive 
        if(T_prob>D_prob):
            result.append("truthful")
            
        else:
            result.append("deceptive")
            
    return result
    



if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = sys.argv
    # Load in the training and test datasets. The file format is simple: one object
    # per line, the first word one the line is the label.
    train_data = load_file(train_file)
    test_data = load_file(test_file)
    if(sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    # make a copy of the test data without the correct labels, so the classifier can't cheat!
    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}

    results= classifier(train_data, test_data_sanitized)

    # calculate accuracy
    correct_ct = sum([ (results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"])) ])
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))
