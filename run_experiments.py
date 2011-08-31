#!/usr/bin/env python
# encoding: utf-8
"""
run_experiments.py

Created by Stephen Holiday on 2011-08-01.
Copyright (c) 2011 Stephen Holiday. All rights reserved.
"""

import sys
import os
import random
import time
import json
import collections

from nltk.corpus import movie_reviews

from experiment import Experiment
import feature_extractor
import classifier

def main():
    filename='stats'
    
    def save_stats():
        json.dump(stats,open('%s.json'%filename,'w'))
    
    stats={'feat_extractors':{}}
    try:
        stats=json.load(open('%s.json'%filename))
    except IOError:
        print 'Creating Stats'
        stats={'feat_extractors':{}}
        save_stats()
    CUT_PERCENT=0.8
    
    feat_extractors=[
        feature_extractor.Unigrams,feature_extractor.Bigrams,feature_extractor.Trigrams,
        
        feature_extractor.Trigram_Collocation_200, feature_extractor.Trigram_Collocation_100,
        feature_extractor.Trigram_Collocation_300, feature_extractor.Trigram_Collocation_400,
        
        feature_extractor.Bigram_Collocation_200,feature_extractor.Bigram_Collocation_100,
        feature_extractor.Bigram_Collocation_300,feature_extractor.Bigram_Collocation_400,
        feature_extractor.Unigrams_stopwords, feature_extractor.Bigrams_stopwords, feature_extractor.Trigrams_stopwords,
        ]
    classifiers=[
        classifier.NaiveBayes,
        #classifier.Maxent,
        #classifier.Weka_naivebayes,
        #classifier.Weka_C45,
        #classifier.Weka_kstar,
        #classifier.Weka_log_regression,
        #classifier.Weka_ripper,
        #classifier.Weka_svm,
        
    ]

    documents = [(list(movie_reviews.words(fileid)), category)#=='pos') 
        for category in movie_reviews.categories() 
        for fileid in movie_reviews.fileids(category)]
    
    pos_documents = list()
    neg_documents = list()
    
    for doc in documents:
        if doc[1]=='pos':
            pos_documents.append(doc)
        else:
            neg_documents.append(doc)
            
    pos_cut_point=int(len(pos_documents)*CUT_PERCENT)
    neg_cut_point=int(len(neg_documents)*CUT_PERCENT)
    print 'pos_cut_point %d'%pos_cut_point
    print 'neg_cut_point %d'%neg_cut_point
    
    trainfeats = pos_documents[:pos_cut_point] + neg_documents[:neg_cut_point]
    testfeats = pos_documents[pos_cut_point:] + neg_documents[neg_cut_point:]
    
    
    print 'Starting Feature Extraction'
    for pheat_extractor_name in feat_extractors:
        pheat_extractor=pheat_extractor_name()
        pheat_extractor_name='%s'%pheat_extractor_name.name
        
        if not stats['feat_extractors'].has_key(pheat_extractor_name):
            stats['feat_extractors'][pheat_extractor_name]=dict()
        
        print ' - Running %s'%pheat_extractor
        
        train_docs = pos_documents[:pos_cut_point] + neg_documents[:neg_cut_point]
        test_docs = pos_documents[pos_cut_point:] + neg_documents[neg_cut_point:]
        
        start_time=time.time()
        train_set = pheat_extractor.extract_features(train_docs)
        test_set = pheat_extractor.extract_features(test_docs)
        end_time=time.time()
        
        print ' - - Ran %s in %f seconds'%(pheat_extractor,end_time-start_time)
        stats['feat_extractors'][pheat_extractor_name]['time']=end_time-start_time
        
        if not stats['feat_extractors'][pheat_extractor_name].has_key('classifiers'):
            stats['feat_extractors'][pheat_extractor_name]['classifiers']=dict()
        for klassifier_name in classifiers:
            klassifier=klassifier_name()
            klassifier_name='%s'%klassifier_name.name
            
            if stats['feat_extractors'][pheat_extractor_name]['classifiers'].has_key(klassifier_name):
                print ' + Skipping classifier %s because it was already run'%klassifier_name
            
            else:
                #try:
                print ' + Running classifier %s'%klassifier_name
        
                stats['feat_extractors'][pheat_extractor_name]['classifiers'][klassifier_name]=dict()
        
        
                print ' + + Training'
                start_time=time.time()
                klassifier.train(train_set)
                end_time=time.time()
        
                print ' + + Trained %s in %f seconds'%(klassifier,end_time-start_time)
                stats['feat_extractors'][pheat_extractor_name]['classifiers'][klassifier_name]['train_time']=end_time-start_time
        
                print ' + + Testing'
                start_time=time.time()
                accuracy= klassifier.test(test_set)
                end_time=time.time()
        
                print ' + + Tested %s in %f seconds with accuracy %f'%(klassifier,end_time-start_time,accuracy)
                stats['feat_extractors'][pheat_extractor_name]['classifiers'][klassifier_name]['test_time']=end_time-start_time
                stats['feat_extractors'][pheat_extractor_name]['classifiers'][klassifier_name]['accuracy']=accuracy
            
                #except Exception, e:
                #    print 'Caught Exception %s'%e
            
                save_stats()
        save_stats()
    print stats
if __name__ == '__main__':
    main()

