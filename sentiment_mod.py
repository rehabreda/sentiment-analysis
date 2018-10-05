# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:54:33 2018

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
with open('word_features.pickle','rb') as f:
    word_features=pickle.load(f) 
    
def find_feature(text):
    words=word_tokenize(text)
    features={}
    for w in word_features:
        features[w]=(w in words)
    return  features 

with open('naivebayes.pickle','rb') as f:
    naivebayes=pickle.load(f) 
    
with open('MultinomialNB.pickle','rb') as f:
    MNB_classifier=pickle.load(f)    
    
with open('BernoulliNB.pickle','rb') as f:
    Ber_classifier=pickle.load(f)  

with open('LogisticRegression.pickle','rb') as f:
    log_classifier=pickle.load(f)      
    
    
with open('SGDClassifier.pickle','rb') as f:
    sgd_classifier=pickle.load(f) 

with open('SVC.pickle','rb') as f:
    svc_classifier=pickle.load(f)     


voted_classifier=vote_classifier(naivebayes,MNB_classifier,Ber_classifier,log_classifier,sgd_classifier,svc_classifier) 

def sentiment(text):
    feats = find_feature(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)        