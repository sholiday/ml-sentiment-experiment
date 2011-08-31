#!/usr/bin/env python
# encoding: utf-8
"""
format_stats.py

Created by Stephen Holiday on 2011-08-31.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""

import sys
import os
import json

import feature_extractor
import classifier
def main():
    
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
        classifier.Weka_naivebayes,
        classifier.Weka_C45,
        classifier.Weka_kstar,
        classifier.Weka_log_regression,
        classifier.Weka_ripper,
        classifier.Weka_svm,
        
    ]
    
    filename='stats'
    stats = json.load(open('%s.json'%filename,'r'))
    
    print stats
    
    fw=open('%s.html'%filename,'w')
    page='<html>'
    
    page+='\n<h2>Accuracy</h2><table border="1">\n<tr><th>&nbsp;</th>'
    for klass in classifiers:
        page+='<th>%s</th>'%klass.name
    page+='</tr>\n'
    
    for extractor in feat_extractors:
        extractor=extractor.name
        page+='\n<tr><td>%s</td>'%extractor
        
        for klass in classifiers:
            accuracy=stats['feat_extractors'].get(extractor, {}).get('classifiers', {}).get(klass.name, {}).get('accuracy',None)
            if accuracy is None:
                accuracy=''
            else:
                accuracy='%.2f'%(accuracy*100)
            page+='<td>%s</td>'%accuracy
        
        page+='</tr>'
    page +='</table>'
    
    
    page+='\n<h2>Classify Time (ms)</h2><table border="1">\n<tr><th>&nbsp;</th>'
    for klass in classifiers:
        page+='<th>%s</th>'%klass.name
    page+='</tr>\n'
    
    for extractor in feat_extractors:
        extractor=extractor.name
        page+='\n<tr><td>%s</td>'%extractor
        
        for klass in classifiers:
            test_time=stats['feat_extractors'].get(extractor, {}).get('classifiers', {}).get(klass.name, {}).get('test_time',None)
            if test_time is None:
                test_time=''
            else:
                feature_time=stats['feat_extractors'][extractor]['time']
                
                classification_time = float(feature_time)/2000.0 + float(test_time)/400.0
                classification_time = classification_time * 1000.0
                test_time='%.2f'%(classification_time)
            page+='<td>%s</td>'%test_time
        
        page+='</tr>'
    page +='</table>'
        
    fw.write(page)
    fw.close()


if __name__ == '__main__':
    main()

