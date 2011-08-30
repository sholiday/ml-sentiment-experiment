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
    stats=list()
    filename='stats.%d'%time.time()
    
    def save_html():
        fw=open('%s.html'%filename,'w')
        page='<html>'
        
        for run in stats:
            page+='\n<table border="1">\n<tr><th>&nbsp;</th>'
            for klass in classifiers:
                page+='<th>%s</th>'%klass.name
            page+='</tr>\n'
            
            for extractor in feat_extractors:
                extractor=extractor.name
                page+='\n<tr><td>%s</td>'%extractor
                
                for klass in classifiers:
                    accuracy=run['feat_extractors'].get(extractor, {}).get('classifiers', {}).get(klass.name, {}).get('accuracy',None)
                    if accuracy is None:
                        accuracy=''
                    else:
                        accuracy='%.2f'%(accuracy*100)
                    page+='<td>%s</td>'%accuracy
                
                page+='</tr>'
            page +='</table>'
            
        fw.write(page)
        fw.close()
    def save_stats():
        fw=open('%s.json'%filename,'a')
        fw.write('%s\n'%json.dumps(stats))
        fw.close()
        save_html()
        
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
        #classifier.Weka_naivebayes,
        #classifier.Weka_C45,
        #classifier.Weka_kstar,
        #classifier.Weka_log_regression,
        #classifier.Weka_ripper,
        #classifier.Weka_svm,
        
    ]

    all_documents = [(list(movie_reviews.words(fileid)), category=='pos') 
        for category in movie_reviews.categories() 
        for fileid in movie_reviews.fileids(category)]
    
    for run in xrange(10):
        stats.append({'run':run,'feat_extractors':{}})

        CUT_PERCENT=0.8
        
        print ''
        print '---------'
        print 'Run %d'%(run+1)
        print '---------'
        
        print 'Shuffling Documents'
        
        documents=all_documents
        random.shuffle(documents)
        
        cut_point=int(len(documents)*CUT_PERCENT)
        print 'cut_point %d'%cut_point
        stats[run]['cut_point']=cut_point
        
        print 'Starting Feature Extraction'
        for pheat_extractor_name in feat_extractors:
            pheat_extractor=pheat_extractor_name()
            pheat_extractor_name='%s'%pheat_extractor_name.name
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
                try:
                    print ' + Running classifier %s'%klassifier_name
                    klassifier=klassifier_name()
                    klassifier_name='%s'%klassifier_name.name
                
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
                
                except Exception, e:
                    print 'Caught Exception %s'%e
                
                save_stats()
        save_stats()
    print stats
if __name__ == '__main__':
    main()

