#!/usr/bin/env python
# encoding: utf-8
"""
mkgram.py

Created by Stephen Holiday on 2011-08-01.
Copyright (c) 2011 Stephen Holiday. All rights reserved.
"""

def ngrams(token_list,n=2):
    tokens=dict()
    
    for i in xrange(len(token_list)-n):
        token=list()
        for j in range(i,i+n):
            token.append(token_list[j])
        token=' '.join(token)
        
        if not tokens.has_key(token):
            tokens[token]=1
        else:
            tokens[token]+=1
    return tokens

    
if __name__ == '__main__':
    print ngrams('hey there how are you over there'.split(),3)