#!/usr/bin/env python
# encoding: utf-8
"""
mkgram.py

Created by Stephen Holiday on 2011-08-01.
Copyright (c) 2011 Stephen Holiday. All rights reserved.
"""

def ngrams(token_list,n=2):
    return token_list_to_bigrams(token_list)


def token_list_to_bigrams(token_list):
    tokens=dict()
    for i in xrange(0,len(token_list)-2):
        token=token_list[i]+' '+token_list[i+1]
        if not tokens.has_key(token):
            tokens[token]=1
        else:
            tokens[token]+=1
    return tokens