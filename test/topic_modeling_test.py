# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:33:36 2021

@author: srandrad
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))

from module.topic_model_plus_class import Topic_Model_plus
import pandas as pd
test_preprocessed_data = pd.DataFrame()

import unittest

def truncate(num, digits):
   sp = str(num).split('.')
   return '.'.join([sp[0], sp[1][:digits]])

class test_topic_modeling_methods(unittest.TestCase):
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
    
    def test_lda_functions(self): # integration test
        # add asserts for comaring the output csvs/dfs
        list_of_attributes = ['Lesson(s) Learned','Driving Event','Recommendation(s)']
        document_id_col = 'Lesson ID'
        csv_file_name = os.path.join("data","preprocessed_data_LLIS.csv")
        num_topics ={'Lesson(s) Learned':5, 'Driving Event':5, 'Recommendation(s)':5}
        test_lda = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col,database_name='test')
        test_lda.extract_preprocessed_data(csv_file_name)
        test_lda.lda(num_topics=num_topics)
        test_lda.save_lda_models()
        test_lda.save_lda_document_topic_distribution()
        test_lda.save_lda_coherence()
        test_lda.save_lda_taxonomy()
        file_path = test_lda.folder_path #path from saved
        
        doc_topics1 = pd.read_csv(os.path.join(file_path,"lda_topic_dist_per_doc.csv"))#.applymap(str)
        tax1 = pd.read_csv(os.path.join(file_path,"lda_taxonomy.csv")).applymap(str)
        coherence_1 = pd.read_csv(os.path.join(file_path,"lda_coherence.csv")).applymap(str)
        
        #testing functions from bin
        test_lda.lda_extract_models(file_path)
        test_lda.save_lda_document_topic_distribution()
        test_lda.save_lda_coherence()
        test_lda.save_lda_taxonomy()
        
        doc_topics2 = pd.read_csv(os.path.join(file_path,"lda_topic_dist_per_doc.csv"))#.applymap(str)
        tax2 = pd.read_csv(os.path.join(file_path,"lda_taxonomy.csv")).applymap(str)
        coherence_2 = pd.read_csv(os.path.join(file_path,"lda_coherence.csv")).applymap(str)
        
        #delete test folder/everything in it
        for root, dirs, files in os.walk(file_path):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(file_path)
        
        #rounding to account for differences in float number system
        for i in range(len(doc_topics1)):
            for col in list_of_attributes:
                nums = [num for num in doc_topics1.iloc[i][col].strip("[]").split(" ") if len(num)>1]
                doc_topics1.at[i,col] = [truncate(float(num),1) for num in nums]
                nums = [num for num in doc_topics2.iloc[i][col].strip("[]").split(" ") if len(num)>1]
                doc_topics2.at[i,col] = [truncate(float(num),1) for num in nums]
                #if doc_topics1.iloc[i][col] != doc_topics2.iloc[i][col]:
                #    print(doc_topics1.iloc[i][col], doc_topics2.iloc[i][col])
                #    print(type(doc_topics1.iloc[i][col]), type(doc_topics2.iloc[i][col]))
                #    print(col,i)
        #checking we get the same output when using a bin object
        self.assertEqual(doc_topics1.equals(doc_topics2), True)
        self.assertEqual(tax1.equals(tax2), True)
        self.assertEqual(coherence_1.equals(coherence_2), True)
    
    def test_hlda_functions(self): # integration test
        list_of_attributes = ['Lesson(s) Learned','Driving Event','Recommendation(s)']
        document_id_col = 'Lesson ID'
        csv_file_name = os.path.join('data','preprocessed_data_LLIS.csv')
        test_hlda = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col,database_name='test')
        test_hlda.extract_preprocessed_data(csv_file_name)
        test_hlda.hlda(training_iterations=100)
        test_hlda.save_hlda_models()
        test_hlda.save_hlda_document_topic_distribution()
        test_hlda.save_hlda_coherence()
        test_hlda.save_hlda_topics()
        test_hlda.save_hlda_level_n_taxonomy()
        test_hlda.save_hlda_taxonomy()
        file_path = test_hlda.folder_path
        
        doc_topics1 = pd.read_csv(os.path.join(file_path,"hlda_topic_dist_per_doc.csv"))#.applymap(str)
        tax1 = pd.read_csv(os.path.join(file_path,"hlda_taxonomy.csv")).applymap(str)
        coherence_1 = pd.read_csv(os.path.join(file_path,"hlda_coherence.csv")).applymap(str)
        level_1_tax1 = pd.read_csv(os.path.join(file_path,"hlda_level1_taxonomy.csv")).applymap(str)
        topics1 = {}
        for attr in list_of_attributes:
            topics1[attr] = pd.read_csv(os.path.join(file_path,attr+"_hlda_topics.csv")).applymap(str)
        
        #testing functions from bin
        file_path = test_hlda.folder_path 
        test_hlda.hlda_extract_models(file_path)
        test_hlda.save_hlda_document_topic_distribution()
        test_hlda.save_hlda_coherence()
        test_hlda.save_hlda_taxonomy()
        test_hlda.save_hlda_topics()
        test_hlda.save_hlda_level_n_taxonomy()
        
        doc_topics2 = pd.read_csv(os.path.join(file_path,"hlda_topic_dist_per_doc.csv"))#.applymap(str)
        tax2 = pd.read_csv(os.path.join(file_path,"hlda_taxonomy.csv")).applymap(str)
        coherence_2 = pd.read_csv(os.path.join(file_path,"hlda_coherence.csv")).applymap(str)
        level_1_tax2 = pd.read_csv(os.path.join(file_path,"hlda_level1_taxonomy.csv")).applymap(str)
        topics2 = {}
        for attr in list_of_attributes:
            topics2[attr] = pd.read_csv(os.path.join(file_path,attr+"_hlda_topics.csv")).applymap(str)
        
        #delete test folder/everything in it
        for root, dirs, files in os.walk(file_path):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(file_path)        
        
        #rounding to account for differences due to float number system
        for i in range(len(doc_topics1)):
            for col in list_of_attributes:
                nums = [num for num in doc_topics1.iloc[i][col].strip("[]").split(" ") if len(num)>1]
                doc_topics1.at[i,col] = [truncate(float(num),2) for num in nums]
                nums = [num for num in doc_topics2.iloc[i][col].strip("[]").split(" ") if len(num)>1]
                doc_topics2.at[i,col] = [truncate(float(num),2) for num in nums]
                #if doc_topics1.iloc[i][col] != doc_topics2.iloc[i][col]:
                #    print(doc_topics1.iloc[i][col], doc_topics2.iloc[i][col])
                #    print(type(doc_topics1.iloc[i][col]), type(doc_topics2.iloc[i][col]))
                #    print(col,i)
        #checking we get the same output when using a bin object
        self.assertEqual(doc_topics1.equals(doc_topics2), True)
        self.assertEqual(tax1.equals(tax2), True)
        self.assertEqual(coherence_1.equals(coherence_2), True)
        self.assertEqual(level_1_tax1.equals(level_1_tax2), True)
        for attr in list_of_attributes:
            #for i in range(len(topics1[attr])):
                #for col in topics1[attr].columns:
                    #if topics1[attr].iloc[i][col] != topics2[attr].iloc[i][col]:
                    #    print(topics1[attr].iloc[i][col], topics2[attr].iloc[i][col])
                    #    print(i,col)
            self.assertEqual(topics1[attr].equals(topics2[attr]), True, attr)

if __name__ == '__main__':
    unittest.main()
