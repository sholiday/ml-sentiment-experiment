#!/usr/bin/env python
# encoding: utf-8
"""
classifier.py

Created by Stephen Holiday on 2011-08-01.
Copyright (c) 2011 Stephen Holiday. All rights reserved.
"""

import nltk
from nltk.classify.weka import WekaClassifier
import random
import os

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
        
        
class Weka(Classifier):
    name='Weka'
    weka_classifier='naivebayes'

    def train(self,train_set):
        nltk.classify.config_java(bin="/usr/bin/java",options=["-Xmx5g"])
        nltk.classify.config_weka(classpath='/Applications/MacPorts/Weka.app/Contents/Resources/Java/weka.jar')
        
        WekaClassifier._CLASSIFIER_CLASS = {
            'naivebayes': 'weka.classifiers.bayes.NaiveBayes',
            'C4.5': 'weka.classifiers.trees.J48',
            'log_regression': 'weka.classifiers.functions.Logistic',
            'svm': 'weka.classifiers.functions.SMO',
            'kstar': 'weka.classifiers.lazy.kstar',
            'ripper': 'weka.classifiers.rules.JRip',
            'MultilayerPerceptron':'weka.classifiers.functions.MultilayerPerceptron'
            }
        
        self.fname='/tmp/weka.%s.model'%random.randint(99, 9999999)
        self.classifier = WekaClassifier.train(self.fname, train_set, classifier=self.weka_classifier)
    def test(self,test_set): 
        
        classifications = self.classifier.batch_classify(test_set)
        
        wrong=0
        right=0
        for i in xrange(len(test_set)):
            if test_set[i][1] ==  classifications[i]:
                right+=1
            else:
                wrong+=1
        
        
        os.remove(self.fname)
        
        return float(right)/float(len(test_set))
        
class Weka_naivebayes(Weka):
    name='Weka_naivebayes'
    weka_classifier='naivebayes'
    
class Weka_C45(Weka):
    name='Weka_C4.5'
    weka_classifier='C4.5'

class Weka_kstar(Weka):
    name='Weka_kstar'
    weka_classifier='MultilayerPerceptron'
    
class Weka_log_regression(Weka):
    name='Weka_log_regression'
    weka_classifier='log_regression'
    
class Weka_ripper(Weka):
    name='Weka_ripper'
    weka_classifier='ripper'
    
class Weka_svm(Weka):
    name='Weka_svm'
    weka_classifier='svm'