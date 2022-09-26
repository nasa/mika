# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 12:39:28 2022

@author: srandrad
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))

import unittest

from mika.kd.NER import *
import pandas as pd
import numpy as np
from spacy.training import offsets_to_biluo_tags
import spacy
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("sentencizer")

class test_NER(unittest.TestCase):
    def setUp(self):
        ex1 = {'id':1,
               'data': 'the quick brown fox jumped over the lazy dog',
               'label': [[4, 9, 'adj'],
                         [10, 15, 'adj'],
                         [16, 19, 'animal']]}
        ex2 = {'id':2,
               'data': 'the quick brown fox jumped over the lazy dog',
               'label': [[5, 9, 'adj'], #quick mislabeled as uick
                         [11, 15, 'adj'], #brown mislabeled as rown
                         [16, 19, 'animal']]}
        ex3 = {'id':3,
               'data': 'the quick brown fox jumped over the lazy dog',
               'label': [[4, 10, 'adj'], #quick mislabeled as 'quick '
                         [10, 16, 'adj'], #brown mislabeled as 'brown '
                         [16, 19, 'animal']]}
        ex4 = {'id':4,
               'data': 'the quick brown fox jumped over the lazy dog',
               'label': [[3, 9, 'adj'], #quick mislabeled as ' quick'
                         [9, 15, 'adj'], #brown mislabeled as ' brown'
                         [16, 19, 'animal']]}
        ex5 = {'id':5,
               'data': 'the quick brown fox jumped over the lazy dog',
               'label': [[3, 10, 'adj'], #quick mislabeled as ' quick '
                         [9, 16, 'adj'], #brown mislabeled as ' brown '
                         [16, 19, 'animal']]}
        ex6 = {'id':6,
               'data': 'the quick brown fox jumped over the lazy dog',
               'label': [[4, 8, 'adj'], #quick mislabeled as quic
                         [10, 14, 'adj'], #brown mislabeled as brow
                         [16, 19, 'animal']]}
        ex7 = {'id':7,
               'data': 'the quick brown fox jumped over the lazy dog',
               'label': [[3, 8, 'adj'], #quick mislabeled as ' quic'
                         [11, 16, 'adj'], #brown mislabeled as 'rown '
                         [16, 19, 'animal']]}
        ex8 = {'id':8,
               'data': 'the quick brown.fox jumped over the lazy dog',
               'label': [[3, 8, 'adj'], #. put in
                         [11, 16, 'adj'], 
                         [16, 19, 'animal']]}
        ex8_labels = [[4, 9, 'adj'],
                  [10, 15, 'adj'],
                  [18, 21, 'animal']]
        ex8_text = 'the quick brown . fox jumped over the lazy dog'
        ex9 = {'id':9,
               'data': 'the quick.brown fox jumped over the lazy dog',
               'label': [[3, 8, 'adj'], 
                         [11, 16, 'adj'], 
                         [16, 19, 'animal']]}
        ex9_text = 'the quick . brown fox jumped over the lazy dog'
        ex9_labels = [[4, 9, 'adj'],
                  [12, 17, 'adj'],
                  [18, 21, 'animal']]
        ex10 = {'id':10,
               'data': 'the.quick brown fox jumped over the lazy dog',
               'label': [[3, 8, 'adj'], #quick mislabeled as ' quic'
                         [11, 16, 'adj'], #brown mislabeled as 'rown '
                         [16, 19, 'animal']]}
        ex10_text = 'the. quick brown fox jumped over the lazy dog'
        ex10_labels = [[5, 10, 'adj'],
                  [11, 16, 'adj'],
                  [17, 20, 'animal']]
        #need to add the test cases where the raw text is connected
        self.correct_labels = [ex1['label'] for i in range(7)]+[ex8_labels]+[ex9_labels]+[ex10_labels]
        self.correct_text = [ex1['data'] for i in range(7)]+[ex8_text, ex9_text, ex10_text]
        data = [ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, ex9, ex10]
        self.df = pd.DataFrame.from_records(data)
        self.df_corrected = pd.DataFrame({'id':[i for i in range(1,11)],
                                        'data':self.correct_text,
                                        'label': self.correct_labels})
        self.test_filename = "doccano_test.jsonl"
        self.df.to_json(self.test_filename, orient='records', lines=True)
        
    def tearDown(self):
        os.remove(self.test_filename)

    def test_read_doccano_annots(self):
        df  = read_doccano_annots(self.test_filename, encoding=False)
        pd.testing.assert_frame_equal(df, self.df)
        df  = read_doccano_annots(self.test_filename, encoding=True)
        pd.testing.assert_frame_equal(df, self.df)
    
    def test_clean_doccano_annots(self):
        clean_df = clean_doccano_annots(self.df)
        pd.testing.assert_frame_equal(clean_df, self.df_corrected)
    
    def test_clean_annots_from_str(self):
        df = self.df.copy()
        df['label'] = df['label'].astype(str)
        df = clean_annots_from_str(df)
        pd.testing.assert_frame_equal(df, self.df)
        
    def test_clean_text_tags(self):
        for i in range(len(self.df)):
            correct_labels = self.correct_labels[i]
            correct_text = self.correct_text[i]
            text = self.df.at[i,'data']
            labels = self.df.at[i, 'label']
            test_labels, text = clean_text_tags(text, labels)
            self.assertEqual(test_labels, correct_labels, text)
            self.assertEqual(text, correct_text)
    
    def test_identify_bad_annotations(self):
        text_df = self.df_corrected.copy()
        docs = self.df_corrected['data']
        docs = [nlp(doc) for doc in docs]
        text_df['docs'] = docs
        text_df['tags'] = [offsets_to_biluo_tags(text_df.at[i,'docs'], text_df.at[i,'label']) for i in range(len(text_df))]
        bad_tokens = identify_bad_annotations(text_df)
        self.assertEqual(bad_tokens, [])
        text_df = self.df.copy()
        docs = self.df['data']
        docs = [nlp(doc) for doc in docs]
        text_df['docs'] = docs
        text_df['label'] = [self.df_corrected.at[i,'label'] for i in range(7)] + [self.df.at[i,'label'] for i in range(7, 10)]
        text_df['tags'] = [offsets_to_biluo_tags(text_df.at[i,'docs'], text_df.at[i,'label']) for i in range(len(text_df))]
        bad_tokens = identify_bad_annotations(text_df)
        correct_bad_tokens = ['quick', 'brown.fox', 'quick.brown', 'the.quick', 'brown']
        bad_tokens = [tok.text for tok in bad_tokens]
        self.assertEqual(bad_tokens, correct_bad_tokens, bad_tokens)
        
    def test_split_doc_to_sentences(self):
        text_df = self.df_corrected.copy()
        new_doc = 'the quick brown fox jumped over the lazy dog. the quick brown fox jumped over the lazy dog'
        new_doc_labels =  [[4, 9, 'adj'],
                            [10, 15, 'adj'],
                            [16, 19, 'animal'],
                            [50, 55, 'adj']]
        text_df = text_df.append(pd.DataFrame({'id':[11], 'data':[new_doc], 'label':[new_doc_labels]})).reset_index(drop=True)
        docs = text_df['data'].to_list()
        docs = [nlp(doc) for doc in docs]
        text_df['docs'] = docs
        text_df['tags'] = [offsets_to_biluo_tags(text_df.at[i,'docs'], text_df.at[i,'label']) for i in range(len(text_df))]
        
        sentence_df = pd.DataFrame()#self.df_corrected.copy()
        sentence_df['id'] = [i for i in range(1,11)] +[11, 11]
        sentence_df['data'] = [text_df.at[i,'data'] for i in range(10)] + ['the quick brown fox jumped over the lazy dog.',
                                                                           'the quick brown fox jumped over the lazy dog']
        
        docs = sentence_df['data']
        docs = [nlp(doc) for doc in docs]
        sentence_df['sentence'] = docs
        sentence_df = sentence_df.drop(['data'], axis=1) 
        sentence_df['tags'] = [['O','U-adj', 'U-adj','U-animal','O','O','O','O','O'] for i in range(7)] + [['O','U-adj', 'U-adj', 'O', 'U-animal','O','O','O','O','O'],
                                                                                                     ['O','U-adj', 'O', 'U-adj','U-animal','O','O','O','O','O'],
                                                                                                     ['O','O', 'U-adj', 'U-adj','U-animal','O','O','O','O','O'],
                                                                                                     ['O','U-adj', 'U-adj','U-animal','O','O','O','O','O', 'O'],
                                                                                                     ['O','U-adj', 'O','O','O','O','O','O','O']
                                                                                                     ]
        
        test_sentence_df = split_docs_to_sentances(text_df, id_col='id', tags=True)
        for col in test_sentence_df.columns:
            if col != 'sentence':
                self.assertEqual(test_sentence_df[col].to_list(), sentence_df[col].to_list())
            else: 
                for i in range(len(test_sentence_df[col])):
                    self.assertEqual(test_sentence_df.at[i,col].text, sentence_df.at[i,col].text)
        #pd.testing.assert_frame_equal(test_sentence_df.all, sentence_df.all, test_sentence_df)
    def test_check_doc_to_sentence_split(self):
        text_df = self.df_corrected.copy()
        new_doc = 'the quick brown fox jumped over the lazy dog. the quick brown fox jumped over the lazy dog'
        new_doc_labels =  [[4, 9, 'adj'],
                            [10, 15, 'adj'],
                            [16, 19, 'animal'],
                            [50, 55, 'adj']]
        text_df = text_df.append(pd.DataFrame({'id':[11], 'data':[new_doc], 'label':[new_doc_labels]})).reset_index(drop=True)
        docs = text_df['data'].to_list()
        docs = [nlp(doc) for doc in docs]
        text_df['docs'] = docs
        text_df['tags'] = [offsets_to_biluo_tags(text_df.at[i,'docs'], text_df.at[i,'label']) for i in range(len(text_df))]
        
        sentence_df = pd.DataFrame()#self.df_corrected.copy()
        sentence_df['id'] = [i for i in range(1,11)] +[11, 11]
        sentence_df['data'] = [text_df.at[i,'data'] for i in range(10)] + ['the quick brown fox jumped over the lazy dog.',
                                                                           'the quick brown fox jumped over the lazy dog']
        
        docs = sentence_df['data']
        docs = [nlp(doc) for doc in docs]
        sentence_df['sentence'] = docs
        sentence_df = sentence_df.drop(['data'], axis=1) 
        sentence_df['tags'] = [['O','U-adj', 'U-adj','U-animal','O','O','O','O','O'] for i in range(7)] + [['O','U-adj', 'U-adj', 'O', 'U-animal','O','O','O','O','O'],
                                                                                                     ['O','U-adj', 'O', 'U-adj','U-animal','O','O','O','O','O'],
                                                                                                     ['O','O', 'U-adj', 'U-adj','U-animal','O','O','O','O','O'],
                                                                                                     ['O','U-adj', 'U-adj','U-animal','O','O','O','O','O', 'O'],
                                                                                                     ['O','U-adj', 'O','O','O','O','O','O','O']
                                                                                                     ]
        
        self.assertEqual(None, check_doc_to_sentence_split(sentence_df))

    def test_get_cleaned_label(self):
        label = 'B-MOD'
        test_label = get_cleaned_label(label)
        self.assertEqual(test_label, "MOD")
        label = 'MOD'
        test_label = get_cleaned_label(label)
        self.assertEqual(test_label, "MOD")

if __name__ == '__main__':
    unittest.main()