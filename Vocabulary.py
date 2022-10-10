#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:01:58 2022

@author: ms
"""
from pycocotools.coco import COCO
from tqdm import tqdm
import numpy as np
from collections import Counter
import nltk
import torch
import pickle

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
        self.padding_word = "<pad>"
        self.start_word = "<start>"
        self.end_word="<end>"
        self.unk_word="<unk>"
    
        self.add_word(self.padding_word)
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
    
    def loadCaptions(self, annFile):
        counter = Counter()
        with open(annFile) as file:
            for line in file:
                caption = line.strip().lower()
                tokens = nltk.tokenize.word_tokenize(caption)
                counter.update(tokens)    
        words = [word for word, cnt in counter.items() if cnt > 0]
        for i, word in enumerate(words):
            self.add_word(word)
                
    
    def saveFile(self, file_name):
        with open(file_name, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def loadFile(self, file_name):
        with open(file_name, 'rb') as f:
            vocab = pickle.load(f)
            self.word2idx = vocab.word2idx
            self.idx2word = vocab.idx2word
  

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def convertSentenceToToken(self, caption):
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.__call__(self.start_word))
        caption.extend([self.__call__(token) for token in tokens])
        caption.append(self.__call__(self.end_word))
        caption = torch.Tensor(caption).long()
        return caption
    
    
    def convertMultipleSentenceToToken(self, captions):
        data = []
     
        for v in captions:
            d = self.convertSentenceToToken(v)
            data.append(d)
            
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
     
        return data
       
    def clean_sentence(self, output):
        sentence = ""
        for i in output:
            word = self.idx2word[i]
            if(word == self.start_word):
                continue
            elif(word == self.end_word):
                break
            else:
                sentence = sentence + " " + word
        return sentence


print("Generate Vocab...")
vocab = Vocabulary()
print("Load training caption...")
vocab.loadCaptions("caption_label.txt")
print("Save Vocab...")
vocab.saveFile("data/vocab.pickle")
