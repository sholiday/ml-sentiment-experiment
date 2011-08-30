#!/usr/bin/env python
# encoding: utf-8
"""
feature_extractor.py

Created by Stephen Holiday on 2011-08-01.
Copyright (c) 2011 Stephen Holiday. All rights reserved.
"""

import mkgram
import nltk
import itertools
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

class Feature_Extractor():
    name=None
    
    def extract_features(self,documents):
        raise NotImplementedError('Should be implemented by subclass')
        
    def __repr__(self):
           return '%s'%self.name
    
class Ngrams(Feature_Extractor):
    n=1
    
    name='Ngrams'
    ngrammer=mkgram.ngrams
    
    def extract_features(self,documents):
        return [(self._extract_document_feats(d), c) for (d,c) in documents]
        
    def _extract_document_feats(self,document):
        document_words = self.ngrammer(token_list=document,n=self.n)
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
    
class Trigrams(Ngrams):
    name='Trigrams'
    n=3
    
class Ngrams_stopwords(Ngrams):
    name='Ngrams_stopwords'
    ngrammer=mkgram.ngrams_stopwrods
    
class Unigrams_stopwords(Ngrams_stopwords):
    name='Unigrams_stopwords'
    n=1

class Bigrams_stopwords(Ngrams_stopwords):
    name='Bigrams_stopwords'
    n=2

class Trigrams_stopwords(Ngrams_stopwords):
    name='Trigrams_stopwords'
    n=3
    
    
class Bigram_Collocation_200(Ngrams):
    name='Bigram_Collocation_200'
    score_fn=BigramAssocMeasures.chi_sq
    top=200
    n=1
    def _extract_document_feats(self,document):
        document_words = self.ngrammer(token_list=document,n=self.n)
        bigram_finder = BigramCollocationFinder.from_words(document_words)
        bigrams = bigram_finder.nbest(self.score_fn, self.top)
        return dict([(ngram, True) for ngram in itertools.chain(document_words, bigrams)])
        
class Bigram_Collocation_100(Bigram_Collocation_200):
    name='Bigram_Collocation_100'
    top=100

class Bigram_Collocation_300(Bigram_Collocation_200):
    name='Bigram_Collocation_300'
    top=300
    
class Bigram_Collocation_400(Bigram_Collocation_200):
    name='Bigram_Collocation_400'
    top=400

class Trigram_Collocation_200(Ngrams):
    name='Trigram_Collocation_200'
    score_fn=TrigramAssocMeasures.chi_sq
    top=200
    n=1
    def _extract_document_feats(self,document):
        document_words = self.ngrammer(token_list=document,n=self.n)
        trigram_finder = TrigramCollocationFinder.from_words(document_words)
        trigrams = trigram_finder.nbest(self.score_fn, self.top)
        return dict([(ngram, True) for ngram in itertools.chain(document_words, trigrams)])
        
class Trigram_Collocation_300(Trigram_Collocation_200):
    name='Trigram_Collocation_300'
    top=300
    
class Trigram_Collocation_100(Trigram_Collocation_200):
    name='Trigram_Collocation_100'
    top=100
    
class Trigram_Collocation_400(Trigram_Collocation_200):
    name='Trigram_Collocation_400'
    top=400