#!/usr/bin/env python
# encoding: utf-8
"""
experiment.py

Created by Stephen Holiday on 2011-08-01.
Copyright (c) 2011 Stephen Holiday. All rights reserved.
"""

import time

class Experiment():
    feature_extractor_name = None
    feature_extractor = None
    
    classifier_name = None
    classifier = None
        
        
    def __init__(self,name):
        self.name=name
    
    def __str__(self):
        res='Experiment<%s,%s,%s>'%(self.name,self.feature_extractor_name,self.classifier_name)
        
    def extract_features(self):
        
        self._extract_features(self)
        
        
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
    
    