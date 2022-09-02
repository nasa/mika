# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:33:36 2021

@author: srandrad
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))

from mika.kd import Topic_Model_plus
from mika.utils import Data
import pandas as pd
test_preprocessed_data = pd.DataFrame()

import unittest

def truncate(num, digits):
   sp = str(num).split('.')
   return '.'.join([sp[0], sp[1][:digits]])

class test_topic_modeling_methods(unittest.TestCase):
    def setUp(self):
        self.test_data = Data()
        text_columns = ['Lesson(s) Learned','Driving Event','Recommendation(s)']
        document_id_col = 'Lesson ID'
        csv_file_name = os.path.join(os.pardir, "data","LLIS","preprocessed_data_LLIS.csv")
        self.test_data.load(csv_file_name, preprocessed=True, id_col=document_id_col, text_columns=text_columns)
        
    def tearDown(self):
        return 
    def test_lda_functions(self): # integration test
        # add asserts for comaring the output csvs/dfs
        num_topics ={'Lesson(s) Learned':5, 'Driving Event':5, 'Recommendation(s)':5}
        test_lda = Topic_Model_plus(text_columns=self.test_data.text_columns, data=self.test_data)
        #test_lda.extract_preprocessed_data(csv_file_name)
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
            for col in self.test_data.text_columns:
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
        test_hlda = Topic_Model_plus(text_columns=self.test_data.text_columns, data=self.test_data)
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
        for attr in self.test_data.text_columns:
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
        for attr in self.test_data.text_columns:
            topics2[attr] = pd.read_csv(os.path.join(file_path,attr+"_hlda_topics.csv")).applymap(str)
        
        #delete test folder/everything in it
        for root, dirs, files in os.walk(file_path):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(file_path)        
        
        #rounding to account for differences due to float number system
        for i in range(len(doc_topics1)):
            for col in self.test_data.text_columns:
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
        for attr in self.test_data.text_columns:
            #for i in range(len(topics1[attr])):
                #for col in topics1[attr].columns:
                    #if topics1[attr].iloc[i][col] != topics2[attr].iloc[i][col]:
                    #    print(topics1[attr].iloc[i][col], topics2[attr].iloc[i][col])
                    #    print(i,col)
            self.assertEqual(topics1[attr].equals(topics2[attr]), True, attr)

if __name__ == '__main__':
    unittest.main()
