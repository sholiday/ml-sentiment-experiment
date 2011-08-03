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

from nltk.corpus import movie_reviews

from experiment import Experiment
import feature_extractor
import classifier

def main():
    stats=list()
    filename='stats.%d.json'%time.time()
    def save_stats():
        fw=open(filename,'a')
        fw.write('%s\n'%json.dumps(stats))
        
    feat_extractors=[feature_extractor.Unigrams,feature_extractor.Bigrams,feature_extractor.Trigrams]
    classifiers=[classifier.NaiveBayes]

    all_documents = [(list(movie_reviews.words(fileid)), category) 
        for category in movie_reviews.categories() 
        for fileid in movie_reviews.fileids(category)]

    for run in xrange(2):
        stats.append({'run':run,'feat_extractors':{}})

        CUT_PERCENT=0.8
        
        print ''
        print '---------'
        print 'Run %d'%(run+1)
        print '---------'
        
        print 'Shuffling Documents'
        
        documents=all_documents
        random.shuffle(documents)
        documents=documents[:100]
        
        cut_point=int(len(documents)*CUT_PERCENT)
        
        stats[run]['cut_point']=cut_point
        
        print 'Starting Feature Extraction'
        for pheat_extractor_name in feat_extractors:
            pheat_extractor=pheat_extractor_name()
            pheat_extractor_name='%s'%pheat_extractor_name
            stats[run]['feat_extractors'][pheat_extractor_name]=dict()
            
            print ' - Running %s'%pheat_extractor
            
            start_time=time.time()
            
            feature_sets = pheat_extractor.extract_features(documents)
            
            end_time=time.time()
            
            print ' - - Ran %s in %f seconds'%(pheat_extractor,end_time-start_time)
            stats[run]['feat_extractors'][pheat_extractor_name]['time']=end_time-start_time
            
            train_set, test_set = feature_sets[cut_point:], feature_sets[:cut_point]
            
            stats[run]['feat_extractors'][pheat_extractor_name]['classifiers']=dict()
            for klassifier_name in classifiers:
                print ' + Running classifier %s'%klassifier_name
                klassifier=klassifier_name()
                klassifier_name='%s'%klassifier_name
                
                stats[run]['feat_extractors'][pheat_extractor_name]['classifiers'][klassifier_name]=dict()
                
                
                print ' + + Training'
                start_time=time.time()
                klassifier.train(train_set)
                end_time=time.time()
                
                print ' + + Trained %s in %f seconds'%(klassifier,end_time-start_time)
                stats[run]['feat_extractors'][pheat_extractor_name]['classifiers'][klassifier_name]['train_time']=end_time-start_time
                
                print ' + + Testing'
                start_time=time.time()
                accuracy= klassifier.test(test_set)
                end_time=time.time()
                
                print ' + + Tested %s in %f seconds with accuracy %f'%(klassifier,end_time-start_time,accuracy)
                stats[run]['feat_extractors'][pheat_extractor_name]['classifiers'][klassifier_name]['test_time']=end_time-start_time
                stats[run]['feat_extractors'][pheat_extractor_name]['classifiers'][klassifier_name]['accuracy']=accuracy
        save_stats()
    print stats
if __name__ == '__main__':
    main()

