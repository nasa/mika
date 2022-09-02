# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 12:39:37 2022

@author: srandrad
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))

import unittest

from mika.utils import Data
import pandas as pd
import numpy as np

def truncate(num, digits):
   sp = str(num).split('.')
   return '.'.join([sp[0], sp[1][:digits]])

class test_Data(unittest.TestCase):
    def setUp(self):
        self.test_class = Data()

    def tearDown(self):
        return
    
    def test__update_ids(self):
        return
    
    def test__load_preprocessed(self):
        return
    
    def test__load_raw(self):
        return
    
    def test_load(self):
        #from raw
        #from preprocessed
        #from preprocessed diff format
        return
    
    def test__check_for_ngrams(self):
        return
    
    def test__remove_quote_marks(self):
        return
    
    def test_save(self):
        return
    
    def test__create_unique_ids(self):
        return
    
    def test__remove_incomplete_rows(self):
        return
    
    def test__combine_columns(self):
        return
    
    def test_prepare(self):
        return
    
    def test__remove_words_in_pct_of_docs(self):
        in_df = pd.DataFrame({"docs": [['this', 'is', 'a', 'test'],['is', 'test'],['test'],
                                       ['cat'],['black','cat'],['python'],['python', 'is', 'good'],
                                       ['is'],['end']], "ids":[0,1,2,3,4,5,6,7,8]})
        self.test_class.text_columns = ["docs"]
        test_word_removal = self.test_class._Data__remove_words_in_pct_of_docs(data_df=in_df)
        correct_word_removal = pd.DataFrame({
            "docs":[["this", "a"], ["cat"], ["black", "cat"],["python"],["python", "good"], ["end"]],
            "ids": [0,3,4,5,6,8]})
        self.assertEqual(test_word_removal.equals(correct_word_removal),True)

    def test__trigram_texts(self):
        return
    
    def test__segment_text(self):
        test_segment_text_result = self.test_class._Data__segment_text([['devicesthe','nasa','correct']],[])
        correct_segmentation = [['devices','the','as','a','correct']]
        self.assertEqual(test_segment_text_result,correct_segmentation)
        
    def test__spellchecker(self):
        test_spellchecker_result = self.test_class._Data__spellchecker([['strted','nasa','NASA','CalTech','pyrolitic']],[])
        correct_spellcheck = [['started','casa','NASA','CalTech','pyrolytic']]
        self.assertEqual(test_spellchecker_result,correct_spellcheck)
        
    def test__quot_normalize(self):
        test_quot_normalize_result = self.test_class._Data__quot_normalize([['quotation','devicequot','quotring','adaquotpt']])
        correct_quot_normalization = [['quotation','device','ring','adapt']]
        self.assertEqual(test_quot_normalize_result,correct_quot_normalization)

    def test__remove_stopwords(self):
        test_remove_stopwords_result = self.test_class._Data__remove_stopwords([['system','that','can','be']],domain_stopwords=[])
        correct_remove_stopwords = [['system']]
        self.assertEqual(test_remove_stopwords_result,correct_remove_stopwords)
        
    def test__lemmatize_texts(self):
        test_lemmatize_texts_result = self.test_class._Data__lemmatize_texts([['start','started','starts','starting']])
        correct_lemmatize = [['start','start','start','start']]
        self.assertEqual(test_lemmatize_texts_result,correct_lemmatize)
        
    def test__lowercase_texts(self):
        test_lowercase_texts_result = self.test_class._Data__lowercase_texts([['The','the','THE']])
        correct_lowercase = [['the','the','the']]
        self.assertEqual(test_lowercase_texts_result,correct_lowercase)
        
    def test__tokenize_texts(self):
        #min, max word lengths do not effect output
        test_tokenize_texts_result = self.test_class._Data__tokenize_texts(['the quick brown fox jumps over the lazy dog'], 0, 20)
        correct_tokenization = [['the','quick','brown','fox','jumps','over','the','lazy','dog']]
        self.assertEqual(test_tokenize_texts_result,correct_tokenization)
        #min = 3, max = 20
        test_tokenize_texts_result = self.test_class._Data__tokenize_texts(['the quick brown fox jumps over the lazy dog'], 3, 20)
        correct_tokenization = [['quick','brown','jumps','over','lazy']]
        self.assertEqual(test_tokenize_texts_result,correct_tokenization)
        #min = 0, max = 4
        test_tokenize_texts_result = self.test_class._Data__tokenize_texts(['the quick brown fox jumps over the lazy dog'], 0, 4)
        correct_tokenization = [['the','fox','the','dog']]
        self.assertEqual(test_tokenize_texts_result,correct_tokenization)

    def test__drop_short_docs(self):
        return
    
    def test__drop_duplicate_docs(self):
        return
    
    def test_preprocess(self):
        test_data_df = pd.DataFrame({"docs":['this is a test','is test','test',
        'cat','black cat','python','python is good','is','end'],"ids":[0,1,2,3,4,5,6,7,8]})
        self.test_class.data_df = test_data_df
        self.test_class.id_col = 'ids'
        self.test_class.text_columns = ['docs']
        self.test_class.preprocess_data(quot_correction=True,spellcheck=True,segmentation=True,drop_short_docs_thres=2,percent=.9) # pct needs to be high because this does not scale well for small doc/set sizes
        test_preprocess_data_result = self.test_class.data_df
        correct_preprocess_data_result = pd.DataFrame({"docs":[['black','cat'],['python','good']],"ids":[4,6]})
        correct_col_1 = correct_preprocess_data_result['docs'][0]
        correct_col_2 = correct_preprocess_data_result['docs'][1]
        test_col_1 = test_preprocess_data_result['docs'][0]
        test_col_2 = test_preprocess_data_result['docs'][1]
        test_bool = False # since the trigram processing shuffles word order, need to test unordered lists
        if set(correct_col_1) == set(test_col_1) and set(correct_col_2) == set(test_col_2):
            test_bool = True
        self.assertEqual(test_bool,True)
        return
    
    def test_sentence_tokenization(self):
        return
    
if __name__ == '__main__':
    unittest.main()