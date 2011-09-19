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

class DecisionTree(Classifier):
    name='DecisionTree'

    def train(self,train_set):
        self.classifier = nltk.classify.decisiontree.DecisionTreeClassifier.train(train_set, binary=True)

    def test(self,test_set):
        return nltk.classify.accuracy(self.classifier, test_set)
        
class Maxent(Classifier):
    name='Maxent'

    def train(self,train_set):
        '''self.f= nltk.classify.maxent.BinaryMaxentFeatureEncoding.train(train_set)
        
        encoded_feat_set=list()
        for pair in train_set:
            #print pair
            encoded=self.f.encode(pair[0],pair[1])
            #print encoded
            encoded_feat_set.append((encoded,pair[1]=='pos'))
            #exit()
        print 'encoded'
        self.classifier = nltk.classify.maxent.MaxentClassifier.train(encoded_feat_set)
        '''
        self.classifier = nltk.classify.maxent.MaxentClassifier.train(train_set)
        
    def test(self,test_set):
        '''encoded_feat_set=list()
        for pair in test_set:
            #print pair
            encoded=self.f.encode(pair[0],pair[1])
            #print encoded
            encoded_feat_set.append((encoded,pair[1]=='pos'))
        return nltk.classify.accuracy(self.classifier, encoded_feat_set)
        '''
        return nltk.classify.accuracy(self.classifier,test_set)
        
class Weka(Classifier):
    name='Weka'
    weka_classifier='naivebayes'

    def train(self,train_set):
        nltk.classify.config_java(bin="/usr/bin/java",options=["-Xmx15g"])
        #classpath='/Applications/MacPorts/Weka.app/Contents/Resources/Java/weka.jar
        classpath='/Applications/weka-3-6-5.app/Contents/Resources/Java/weka.jar'
        nltk.classify.config_weka(classpath=classpath)
        
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