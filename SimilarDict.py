#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 01:59:28 2022

@author: ms
"""
import pickle
import numpy as np

class SimilarDict:
    def __init__(self, threshold = 0.69):        
        with open('similar_dict_glove.6B.300d.pickle', 'rb') as handle:
            self.W, self.vocab, self.ivocab = pickle.load(handle)
            self.threshold = threshold
    
   
    def __call__(self, words):
        for idx, term in enumerate(words.split(' ')):
            if term in self.vocab:
                # print('Word: %s  Position in vocabulary: %i' % (term, self.vocab[term]))
                if idx == 0:
                    vec_result = np.copy(self.W[self.vocab[term], :])
                else:
                    vec_result += self.W[self.vocab[term], :]
            else:
                # print('Word: %s  Out of dictionary!\n' % term)
                return []

        vec_norm = np.zeros(vec_result.shape)
        d = (np.sum(vec_result ** 2,) ** (0.5))
        vec_norm = (vec_result.T / d).T

        dist = np.dot(self.W, vec_norm.T)

        for term in words.split(' '):
            index = self.vocab[term]
            dist[index] = -np.Inf

        #if want number
        # a = np.argsort(-dist)[:100]
        a = np.argsort(-dist)
       
        result = []
        for x in a:
            if dist[x] > self.threshold:
                result.append([self.ivocab[x], dist[x]])
        
        return result
