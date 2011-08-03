#!/usr/bin/env python
# encoding: utf-8
"""
experiment.py

Created by Stephen Holiday on 2011-08-01.
Copyright (c) 2011 Stephen Holiday. All rights reserved.
"""

import time

class Experiment():
    feature_extractor = None
    
    classifier = None
        
        
    def __init__(self,feature_extractor,classifier):
        #self.cut_point=cut_point
        self.feature_extractor = feature_extractor
        self.classifier = classifier
    
    def __repr__(self):
        res='Experiment<%s,%s>'%(self.feature_extractor.name,self.classifier.name)
        return res
    
    def extract_features(self,documents):
        
        past_set=self.feature_extractor(documents)
        
        
        
    def train(self):
        
        self._train()
        
        
    def test(self):
        
        self._test()
        
    def _extract_features(self):
        raise NotImplementedError('Should be implemented by subclass')
        
    def _train(self):
        raise NotImplementedError('Should be implemented by subclass')
        
    def _test(self):
        raise NotImplementedError('Should be implemented by subclass')
    
    