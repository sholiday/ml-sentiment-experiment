#!/usr/bin/env python
# encoding: utf-8
"""
mkgram.py

Created by Stephen Holiday on 2011-08-01.
Copyright (c) 2011 Stephen Holiday. All rights reserved.
"""
import nltk
from nltk.corpus import stopwords

def ngrams(s=None,token_list=None,n=2):
    tokens=dict()
    
    for i in xrange(len(token_list)-n+1):
        token=list()
        for j in range(i,i+n):
            token.append(token_list[j])
        token=' '.join(token)
        
        if not tokens.has_key(token):
            tokens[token]=1
        else:
            tokens[token]+=1
    
    return tokens

def ngrams_stopwrods(s=None,token_list=None,n=2):
    words=stopwords.words('english')
    
    token_list=filter(lambda x: x not in words, token_list)
    
    return ngrams(token_list=token_list, n=n)
if __name__ == '__main__':
    print ngrams('hey there how are you over there ghjhg gh'.split(),3)