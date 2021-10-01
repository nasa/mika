# -*- coding: utf-8 -*-
"""
@author: hswalsh, srandrad
Test code for preprocessing and related functions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))

import unittest

from module.topic_model_plus_class import Topic_Model_plus
import pandas as pd

class test_preprocessing_methods(unittest.TestCase):
    def test_tokenize_texts(self):
        test_class = Topic_Model_plus()
        test_tokenize_texts_result = test_class._Topic_Model_plus__tokenize_texts(['the quick brown fox jumps over the lazy dog'])
        correct_tokenization = [['the','quick','brown','fox','jumps','over','the','lazy','dog']]
        self.assertEqual(test_tokenize_texts_result,correct_tokenization)
    def test_quot_normalize(self):
        test_class = Topic_Model_plus()
        test_quot_normalize_result = test_class._Topic_Model_plus__quot_normalize([['quotation','devicequot','quotring','adaquotpt']])
        correct_quot_normalization = [['quotation','device','ring','adapt']]
        self.assertEqual(test_quot_normalize_result,correct_quot_normalization)
    def test_spellchecker(self):
        test_class = Topic_Model_plus()
        test_spellchecker_result = test_class._Topic_Model_plus__spellchecker([['strted','nasa','NASA','CalTech','pyrolitic']])
        correct_spellcheck = [['started','casa','NASA','CalTech','pyrolytic']]
        self.assertEqual(test_spellchecker_result,correct_spellcheck)
    def test_segment_text(self):
        test_class = Topic_Model_plus()
        test_segment_text_result = test_class._Topic_Model_plus__segment_text([['devicesthe','nasa','correct']])
        correct_segmentation = [['devices','the','as','a','correct']]
        self.assertEqual(test_segment_text_result,correct_segmentation)
    def test_lowercase_texts(self):
        test_class = Topic_Model_plus()
        test_lowercase_texts_result = test_class._Topic_Model_plus__lowercase_texts([['The','the','THE']])
        correct_lowercase = [['the','the','the']]
        self.assertEqual(test_lowercase_texts_result,correct_lowercase)
    def test_lemmatize_texts(self):
        test_class = Topic_Model_plus()
        test_lemmatize_texts_result = test_class._Topic_Model_plus__lemmatize_texts([['start','started','starts','starting']])
        correct_lemmatize = [['start','start','start','start']]
        self.assertEqual(test_lemmatize_texts_result,correct_lemmatize)
    def test_remove_stopwords(self):
        test_class = Topic_Model_plus()
        test_remove_stopwords_result = test_class._Topic_Model_plus__remove_stopwords([['system','that','can','be']],domain_stopwords=[])
        correct_remove_stopwords = [['system']]
        self.assertEqual(test_remove_stopwords_result,correct_remove_stopwords)
    def test_remove_frequent_words(self):
        in_df = pd.DataFrame({"docs": [['this', 'is', 'a', 'test'],['is', 'test'],['test'],
                                       ['cat'],['black','cat'],['python'],['python', 'is', 'good'],
                                       ['is'],['end']], "ids":[0,1,2,3,4,5,6,7,8]})
        test_class = Topic_Model_plus()
        test_word_removal = test_class._Topic_Model_plus__remove_words_in_pct_of_docs(data_df=in_df, list_of_attributes=['docs'])
        correct_word_removal = pd.DataFrame({
            "docs":[["this", "a"], ["cat"], ["black", "cat"],["python"],["python", "good"], ["end"]],
            "ids": [0,3,4,5,6,8]})
        self.assertEqual(test_word_removal.equals(correct_word_removal),True)
    def test_preprocess_data(self): # integration test
        test_data_df = pd.DataFrame({"docs":['this is a test','is test','test',
        'cat','black cat','python','python is good','is','end'],"ids":[0,1,2,3,4,5,6,7,8]})
        test_class = Topic_Model_plus(document_id_col = 'ids')
        test_class.data_df = test_data_df
        test_class.list_of_attributes = ['docs']
        test_class.preprocess_data(quot_correction=True,spellcheck=True,segmentation=True,drop_short_docs_thres=2,percent=.9) # pct needs to be high because this does not scale well for small doc/set sizes
        test_preprocess_data_result = test_class.data_df
        correct_preprocess_data_result = pd.DataFrame({"docs":[['black','cat'],['python','good']],"ids":[4,6]})
        correct_col_1 = correct_preprocess_data_result['docs'][0]
        correct_col_2 = correct_preprocess_data_result['docs'][1]
        test_col_1 = test_preprocess_data_result['docs'][0]
        test_col_2 = test_preprocess_data_result['docs'][1]
        test_bool = False # since the trigram processing shuffles word order, need to test unordered lists
        if set(correct_col_1) == set(test_col_1) and set(correct_col_2) == set(test_col_2):
            test_bool = True
        self.assertEqual(test_bool,True)

if __name__ == '__main__':
    unittest.main()
