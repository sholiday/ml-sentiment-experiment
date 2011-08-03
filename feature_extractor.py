#!/usr/bin/env python
# encoding: utf-8
"""
feature_extractor.py

Created by Stephen Holiday on 2011-08-01.
Copyright (c) 2011 Stephen Holiday. All rights reserved.
"""

import mkgram
import nltk

class Feature_Extractor():
    name=None
    
    def extract_features(self,documents):
        raise NotImplementedError('Should be implemented by subclass')
        
    def __repr__(self):
           return '%s'%self.name
    
class Ngrams(Feature_Extractor):
    n=1
    
    name='Ngrams'
    
    def extract_features(self,documents):
        return [(self._extract_document_feats(d), c) for (d,c) in documents]
        
    def _extract_document_feats(self,document):
        document_words = mkgram.ngrams(document,self.n) 
        features = {}
        for word in document_words:
            features['%s' % word]=True
        return features
        
class Unigrams(Ngrams):
    name='Unigrams'
    n=1
    
class Bigrams(Ngrams):
    name='Bigrams'
    n=2