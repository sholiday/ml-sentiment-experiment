#!/usr/bin/env python
# encoding: utf-8
"""
classifier.py

Created by Stephen Holiday on 2011-08-01.
Copyright (c) 2011 Stephen Holiday. All rights reserved.
"""

import nltk

class Classifier():
    name=None
    
    def train(self):
        raise NotImplementedError('Should be implemented by subclass')
        
    def classify(self):
        raise NotImplementedError('Should be implemented by subclass')
        
    def accuracy(self):
        raise NotImplementedError('Should be implemented by subclass')
        
    def __repr__(self):
        return '%s'%self.name
        
class NaiveBayes(Classifier):
    name='NaiveBayes'
    
    def train(self,train_set):
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    def test(self,test_set):
        return nltk.classify.accuracy(self.classifier, test_set)