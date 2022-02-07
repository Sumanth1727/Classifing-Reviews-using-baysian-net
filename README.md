# Classifing-Reviews-using-baysian-net
I’ve given you a dataset of user-generated reviews. people can fake the reviews and try to either defame a good hotel or promote a bad one.
Our task is to classify reviews into faked or legitimate, for 20 hotels in Chicago.
This problem is the simple bayesian classifier problem.
First after taking the dataset. I iterating through each review and build a dictionary of both truth and deceptive words.
```python
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
```
I am removing all the symbols and empty spaces in the words as I go along

After This I will have a dictionary which contains all the words for deceptive and truth and frequency of each word in the true and deceptive dictionary.


## Finding the solution for test dataset
After this I take the test and iterate over it and calculate the probality of each being deceptive or truthful and multiply with their corresponding probabilities
### Problem faced
I can see a few words which I haven't seen in the trained data. I can skip this words but probability will decrease rapidly if I do so. I are using laplace smoothing with smoothing factor of k=1 to aviod this.
Another problem I faced is that since all the values are in decimals after some limit python is converting them into 0's. I avoid that I are taking the log of the value and instead of multiplying the parameters I are adding them since log(ab)=loga + logb  
```python
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

```

After calculating the probabilities for both truthful and deceptive. I are comparing them to see which is higher. I are returning the class which has high probability.
```python
#Comparing Probabilities of truthful and deceptive 
        if(T_prob>D_prob):
            result.append("truthful")
            
        else:
            result.append("deceptive")
            
    return result
    

```

