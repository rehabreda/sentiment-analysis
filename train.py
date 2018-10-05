# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:32:40 2018

@author: rehab
"""
import nltk
import pickle
import random
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB ,MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression ,SGDClassifier
from nltk.classify import ClassifierI
from statistics import mode

class vote_classifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers=classifiers
        
    def classify(self,features):
        votes=[]
        for c in  self._classifiers:
            votes.append(c.classify(features))
            
        return mode(votes)  
    
    def confidence(self,features):
        votes=[]
        for c in  self._classifiers:
            votes.append(c.classify(features))
            
        choice=votes.count(mode(votes))
        return choice/len(votes)
    
    
neg=open('negative.txt','r').read()    
pos=open('positive.txt','r').read()    

all_words=[]
documents=[]
allowed_word_types = ["J"]

for line in pos.split('\n'):
    documents.append((line,'pos'))
    words=word_tokenize(line)
    pos=nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
            
            
for line in neg.split('\n'):
    documents.append((line,'neg'))
    words=word_tokenize(line)
    pos=nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())   
            
with open('documents.pickle','wb') as f:
    pickle.dump(documents,f)
    
all_words=nltk.FreqDist(all_words)

word_features=list(all_words.keys() )[:5000]

with open('word_features.pickle','wb') as f:
    pickle.dump(word_features,f)
    
def find_feature(text):
    words=word_tokenize(text)
    features={}
    for w in word_features:
        features[w]=(w in words)
    return  features
   
feature_set=[(find_feature(rev),cat) for (rev,cat) in documents]    
random.shuffle(feature_set)

train=feature_set[:10000]
test=feature_set[10000:]

classifier=nltk.NaiveBayesClassifier.train(train)
with open('naivebayes.pickle','wb') as f:
    pickle.dump(classifier,f)
    
    
MNB_classifier=SklearnClassifier(MultinomialNB())
MNB_classifier.train(train)

with open('MultinomialNB.pickle','wb') as f:
    pickle.dump(MNB_classifier,f)
    
    
    
Ber_classifier=SklearnClassifier(BernoulliNB())
Ber_classifier.train(train)

with open('BernoulliNB.pickle','wb') as f:
    pickle.dump(Ber_classifier,f)    
    
    
log_classifier=SklearnClassifier(LogisticRegression())
log_classifier.train(train)

with open('LogisticRegression.pickle','wb') as f:
    pickle.dump(log_classifier,f)     
    
    
sgd_classifier=SklearnClassifier(SGDClassifier())
sgd_classifier.train(train)

with open('SGDClassifier.pickle','wb') as f:
    pickle.dump(sgd_classifier,f)         
    
svc_classifier=SklearnClassifier(SVC())
svc_classifier.train(train)

with open('SVC.pickle','wb') as f:
    pickle.dump(svc_classifier,f)             
    

    
            
            
            