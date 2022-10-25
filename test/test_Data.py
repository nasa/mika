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
        text_list = ['the quick brown fox jumps over the lazy dog',
                     'Hello, world!',
                     'How vexingly quick daft zebras jump!',
                     'the lazy dog slept all day',
                     'the kangaroo jumps over the bush']
        self.test_df = pd.DataFrame({"text":text_list,
                                "id":[1,2,3,4,5]})
        self.test_filename = "data_test.csv"
        self.test_id_col = 'id'
        self.test_text_cols = ['text']
        self.test_df.to_csv(self.test_filename)
        preprocessed_text_list = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog'],
                                  ['Hello,', 'world!'],
                                  ['How', 'vexingly', 'quick', 'daft', 'zebras', 'jump!'],
                                  ['the', 'lazy', 'dog', 'slept', 'all', 'day'],
                                  ['the', 'kangaroo', 'jumps', 'over', 'the', 'bush']]
        self.test_preprocessed_df = pd.DataFrame({"text":preprocessed_text_list,
                                "id":[1,2,3,4,5]})
        self.test_preprocessed_filename = "data_preprocessed_test.csv"
        self.test_id_col = 'id'
        self.test_text_cols = ['text']
        self.test_preprocessed_df.to_csv(self.test_preprocessed_filename)
        
    def tearDown(self):
        os.remove(self.test_filename)
        os.remove(self.test_preprocessed_filename)
    
    @unittest.expectedFailure
    def test__update_ids_failure(self):
        #no id col - should give error and does
        self.test_class.data_df = self.test_df
        self.test_class._Data__update_ids()
        
    def test__update_ids(self):
        #data df
        self.test_class.id_col = self.test_id_col
        self.test_class.data_df = self.test_df
        self.test_class._Data__update_ids()
        self.assertEqual(self.test_class.doc_ids, self.test_df[self.test_id_col].tolist())
        #drop a row
        self.test_class.data_df = self.test_class.data_df.drop([0,2,4]).reset_index(drop=True)
        correct_ids = [2,4]
        self.test_class._Data__update_ids()
        self.assertEqual(self.test_class.doc_ids, correct_ids)
    
    def test__load_preprocessed(self):
        #raw text
        self.test_class.id_col = self.test_id_col
        self.test_class.text_columns=self.test_text_cols
        self.test_class._Data__load_preprocessed(self.test_filename, drop_short_docs=False, drop_duplicates=False, tokenized=False)
        pd.testing.assert_frame_equal(self.test_class.data_df, self.test_df)
        #preprocessed text
        self.test_class.id_col = self.test_id_col
        self.test_class.text_columns=self.test_text_cols
        self.test_class._Data__load_preprocessed(self.test_preprocessed_filename, drop_short_docs=False, drop_duplicates=False, tokenized=True)
        pd.testing.assert_frame_equal(self.test_class.data_df, self.test_preprocessed_df)
    
    def test__load_raw(self):
        self.test_class.id_col = self.test_id_col
        self.test_class.text_columns=self.test_text_cols
        self.test_class._Data__load_raw(self.test_filename, kwargs={})
        pd.testing.assert_frame_equal(self.test_class.data_df, self.test_df)
    
    def test__set_id_col_to_index(self):
        self.test_class.data_df = self.test_df
        self.test_class._Data__set_id_col_to_index()
        correct_df = self.test_df.copy()
        correct_df['index'] = correct_df.index
        pd.testing.assert_frame_equal(self.test_class.data_df, correct_df)
        self.assertEqual(self.test_class.id_col, 'index')
        
    def test_load(self):
        #from raw
        self.test_class.load(self.test_filename, preprocessed=False, id_col=self.test_id_col, text_columns=self.test_text_cols)
        pd.testing.assert_frame_equal(self.test_class.data_df, self.test_df)
        #from preprocessed
        self.test_class.id_col = self.test_id_col
        self.test_class.text_columns=self.test_text_cols
        preprocessed_kwargs={"tokenized":True, 'drop_short_docs':False, 'drop_duplicates':False}
        self.test_class.load(self.test_preprocessed_filename, preprocessed=True, id_col=self.test_id_col, text_columns=self.test_text_cols, preprocessed_kwargs=preprocessed_kwargs)
        pd.testing.assert_frame_equal(self.test_class.data_df, self.test_preprocessed_df)
        #from preprocessed raw format
        self.test_class.id_col = self.test_id_col
        self.test_class.text_columns=self.test_text_cols
        preprocessed_kwargs={"tokenized":False, 'drop_short_docs':False, 'drop_duplicates':False}
        self.test_class.load(self.test_filename, preprocessed=True, id_col=self.test_id_col, text_columns=self.test_text_cols, preprocessed_kwargs=preprocessed_kwargs)
        pd.testing.assert_frame_equal(self.test_class.data_df, self.test_df)
        #no id col given, use index
        self.test_class.load(self.test_filename, preprocessed=False, id_col=None, text_columns=self.test_text_cols)
        correct_df = self.test_df.copy()
        correct_df['index'] = correct_df.index
        pd.testing.assert_frame_equal(self.test_class.data_df, correct_df)
        self.assertEqual(self.test_class.id_col, 'index')
        
    def test__remove_quote_marks(self):
        word_list = "['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']"
        corrected_list = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
        returned_list = self.test_class._Data__remove_quote_marks(word_list)
        self.assertEqual(returned_list, corrected_list)
    
    def test_save(self):
        #file name given
        correct_filename =  "preprocessed_data_save_test.csv"
        self.test_class.data_df = self.test_df
        self.test_class.save(correct_filename)
        self.assertTrue(os.path.isfile(correct_filename))
        os.remove(correct_filename)
        #no file name given
        correct_filename = "preprocessed_data.csv"
        self.test_class.save()
        self.assertTrue(os.path.isfile(correct_filename))
        os.remove(correct_filename)
    
    def test__create_unique_ids(self):
        self.test_class.id_col = self.test_id_col
        self.test_class.data_df = self.test_df
        #no repeat ids
        self.test_class._Data__create_unique_ids()
        correct_ids = ["1_0", "2_0", "3_0", "4_0", "5_0"]
        self.assertEqual(correct_ids, self.test_class.data_df['Unique IDs'].tolist())
        #add repeat ids
        temp_df = self.test_df.copy()
        temp_df[self.test_class.id_col] = ["1", "1", "3", "4", "5"]
        self.test_class.data_df = temp_df
        self.test_class._Data__create_unique_ids()
        correct_ids = ["1_0", "1_1", "3_0", "4_0", "5_0"]
        self.assertEqual(correct_ids, self.test_class.data_df['Unique IDs'].tolist())
    
    def test__remove_incomplete_rows(self):
        #default, no rows removed
        self.test_class.id_col = self.test_id_col
        self.test_class.data_df = self.test_df
        self.test_class.text_columns = ["text"]
        self.test_class._Data__remove_incomplete_rows()
        pd.testing.assert_frame_equal(self.test_class.data_df, self.test_df)
        #add extra col with missing text
        new_text_col = ["" for i in range(len(self.test_df))]
        temp_df = self.test_df.copy()
        temp_df['text 2'] = new_text_col
        self.test_class.data_df = temp_df
        self.test_class.text_columns = ["text", 'text 2']
        self.test_class._Data__remove_incomplete_rows()
        correct_df = temp_df.drop([i for i in temp_df.index])
        pd.testing.assert_frame_equal(self.test_class.data_df, correct_df)
    
    def test__combine_columns(self):
        #default, same df
        self.test_class.id_col = self.test_id_col
        self.test_class.data_df = self.test_df
        self.test_class.text_columns = ["text"]
        self.test_class._Data__combine_columns([])
        pd.testing.assert_frame_equal(self.test_class.data_df, self.test_df)
        #combined 
        new_text_col = ["new" for i in range(len(self.test_df))]
        temp_df = self.test_df.copy()
        temp_df['text 2'] = new_text_col
        self.test_class.data_df = temp_df
        self.test_class.text_columns = ["text", 'text 2']
        self.test_class._Data__combine_columns(['text', 'text 2'])
        correct_df = temp_df.copy()
        correct_df['Combined Text'] = [correct_df['text'].tolist()[i]+". new" for i in range(len(correct_df))]
        pd.testing.assert_frame_equal(self.test_class.data_df, correct_df)
        
    def test_prepare(self):
        #same result with all inputs as false
        self.test_class.id_col = self.test_id_col
        self.test_class.data_df = self.test_df
        self.test_class.text_columns = ["text"]
        self.test_class.prepare_data(combine_columns=[], remove_incomplete_rows=False, create_ids=False)
        pd.testing.assert_frame_equal(self.test_class.data_df, self.test_df)
    
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
        #2-grams
        self.test_class.id_col = self.test_id_col
        self.test_class.data_df = self.test_preprocessed_df
        self.test_class.text_columns = ["text"]
        texts = self.test_class.data_df["text"]
        ngram_range = 2
        min_count = 1
        threshold = 1
        ngrams = self.test_class._Data__trigram_texts(texts, ngram_range, threshold, min_count)
        ngrams = [ngram for ngram_list in ngrams for ngram in ngram_list]
        ngram_lengths = [len(ngram.split(" ")) for ngram in ngrams]
        self.assertEqual(max(ngram_lengths), 2)
        self.assertEqual(min(ngram_lengths), 1)
        #3-grams
        self.test_class.id_col = self.test_id_col
        temp_df = self.test_preprocessed_df.copy()
        for i in range(5):
            temp_df = temp_df.append(self.test_preprocessed_df.iloc[0][:])
        self.test_class.data_df = temp_df
        self.test_class.text_columns = ["text"]
        texts = self.test_class.data_df["text"]
        ngram_range = 3
        min_count = 1
        threshold = 1
        ngrams = self.test_class._Data__trigram_texts(texts, ngram_range, threshold, min_count)
        ngrams = [ngram for ngram_list in ngrams for ngram in ngram_list]
        ngram_lengths = [len(ngram.split(" ")) for ngram in ngrams]
        self.assertEqual(max(ngram_lengths), 3)
        self.assertEqual(min(ngram_lengths), 1)
    
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
        #thresh = 3, should drop 1
        self.test_class.id_col = self.test_id_col
        self.test_class.data_df = self.test_df
        self.test_class.text_columns = self.test_text_cols
        self.test_class._Data__drop_short_docs(thres=3)
        correct_df = self.test_df.drop(1).reset_index(drop=True)
        pd.testing.assert_frame_equal(correct_df, self.test_class.data_df)
        #thresh = 0, should drop none
        self.test_class.data_df = self.test_df
        self.test_class._Data__drop_short_docs(thres=0)
        pd.testing.assert_frame_equal(self.test_df, self.test_class.data_df)
        #thresh = 20, should drop all
        self.test_class.data_df = self.test_df
        self.test_class._Data__drop_short_docs(thres=20)
        correct_df = self.test_df.drop([i for i in self.test_df.index]).reset_index(drop=True)
        pd.testing.assert_frame_equal(correct_df, self.test_class.data_df)
    
    def test__drop_duplicate_docs(self):
        #drop none by default
        self.test_class.id_col = self.test_id_col
        self.test_class.data_df = self.test_df
        self.test_class._Data__drop_duplicate_docs(cols=self.test_text_cols)
        pd.testing.assert_frame_equal(self.test_class.data_df, self.test_df)
        #add 2 dups, should drop them
        temp_df = self.test_df.copy()
        temp_df = temp_df.append(self.test_df.iloc[0][:])
        temp_df = temp_df.append(self.test_df.iloc[0][:])
        self.test_class.data_df = temp_df
        self.test_class._Data__drop_duplicate_docs(cols=self.test_text_cols)
        pd.testing.assert_frame_equal(self.test_class.data_df, self.test_df)
        #add 2 dups with different ids, should not drop them depending on parameter
        temp_df[self.test_id_col] = [int(i)+1 for i in range(len(temp_df))]
        self.test_class.data_df = temp_df
        self.test_class._Data__drop_duplicate_docs(cols=self.test_text_cols)
        pd.testing.assert_frame_equal(self.test_class.data_df, self.test_df)
        temp_df[self.test_id_col] = [int(i)+1 for i in range(len(temp_df))]
        self.test_class.data_df = temp_df.reset_index(drop=True)
        temp_df = temp_df.reset_index(drop=True)
        self.test_class._Data__drop_duplicate_docs(cols=self.test_text_cols+[self.test_id_col])
        pd.testing.assert_frame_equal(self.test_class.data_df, temp_df)
    
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
        sentence_df = pd.DataFrame({'text':['the quick brown fox jumps over the lazy dog. Hello, world! How vexingly quick daft zebras jump! Hello, world'],
                       'id':[1]})
        self.test_class.data_df = sentence_df
        self.test_class.id_col = self.test_id_col
        self.test_class.text_columns = ['text']
        correct_df = sentence_df = pd.DataFrame({'text':['the quick brown fox jumps over the lazy dog. Hello, world! How vexingly quick daft zebras jump! Hello, world' for i in range(4)],
                                                 'id':[1, 1, 1, 1],
                                                 'text Sentences':['the quick brown fox jumps over the lazy dog.',
                                                                   'Hello, world!',
                                                                   'How vexingly quick daft zebras jump!',
                                                                   'Hello, world']})
        self.test_class.sentence_tokenization()
        pd.testing.assert_frame_equal(self.test_class.data_df, correct_df)
    
if __name__ == '__main__':
    unittest.main()