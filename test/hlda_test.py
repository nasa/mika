# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:33:36 2021

@author: srandrad
"""


import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from sys import platform
if platform == "darwin":
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
    smart_nlp_path = ''
elif platform == "win32":
    sys.path.append('../')
    smart_nlp_path = os.getcwd()
    smart_nlp_path = "\\".join([smart_nlp_path.split("\\")[i] for i in range(0,len(smart_nlp_path.split("\\"))-1)])
    
import unittest

from module.topic_model_plus_class import Topic_Model_plus
import pandas as pd
test_preprocessed_data = pd.DataFrame()

def truncate(num, digits):
   sp = str(num).split('.')
   return '.'.join([sp[0], sp[1][:digits]])

class test_hlda_methods(unittest.TestCase):
    """
    def test_hlda(self):
        #run on preprocessed data
        #self.assertEqual(first, second)
        return
    def test_save_extract_hlda_models(self):
        #run on preprocessed data
        #ensure the extracted == saved, make sure they both work
        return
    def test_save_hlda_document_topic_distribution(self):
        #make sure it saves
        return
    def test_save_hlda_coherence(self):
        #make sure it saves, use test object?
        return
    def test_hlda_save_taxonomy(self):
        #make sure it saves, use test object for consistency
        return
    """
    def test_hlda_functions(self): # integration test
        list_of_attributes = ['Lesson(s) Learned','Driving Event','Recommendation(s)']
        document_id_col = 'Lesson ID'
        csv_file_name = smart_nlp_path+r"\input data\preprocessed_data.csv"
        test_hlda = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col)
        test_hlda.extract_preprocessed_data(csv_file_name)
        test_hlda.hlda(training_iterations=100)
        test_hlda.save_hlda_models()
        test_hlda.save_hlda_document_topic_distribution()
        test_hlda.save_hlda_coherence()
        test_hlda.save_hlda_topics()
        test_hlda.save_hlda_level_n_taxonomy()
        test_hlda.save_hlda_taxonomy()
        file_path = test_hlda.folder_path
        
        doc_topics1 = pd.read_csv(file_path+"/hlda_topic_dist_per_doc.csv")#.applymap(str)
        tax1 = pd.read_csv(file_path+"/hlda_taxonomy.csv").applymap(str)
        coherence_1 = pd.read_csv(file_path+"/"+"hlda_coherence.csv").applymap(str)
        level_1_tax1 = pd.read_csv(file_path+"/hlda_level1_taxonomy.csv").applymap(str)
        topics1 = {}
        for attr in list_of_attributes:
            topics1[attr] = pd.read_csv(file_path+"/"+attr+"_hlda_topics.csv").applymap(str)
        
        #testing functions from bin
        file_path = test_hlda.folder_path 
        test_hlda.hlda_extract_models(file_path)
        test_hlda.save_hlda_document_topic_distribution()
        test_hlda.save_hlda_coherence()
        test_hlda.save_hlda_taxonomy()
        test_hlda.save_hlda_topics()
        test_hlda.save_hlda_level_n_taxonomy()
        
        doc_topics2 = pd.read_csv(file_path+"/hlda_topic_dist_per_doc.csv")#.applymap(str)
        tax2 = pd.read_csv(file_path+"/hlda_taxonomy.csv").applymap(str)
        coherence_2 = pd.read_csv(file_path+"/"+"hlda_coherence.csv").applymap(str)
        level_1_tax2 = pd.read_csv(file_path+"/hlda_level1_taxonomy.csv").applymap(str)
        topics2 = {}
        for attr in list_of_attributes:
            topics2[attr] = pd.read_csv(file_path+"/"+attr+"_hlda_topics.csv").applymap(str)
        
        #delete test folder/everything in it
        for root, dirs, files in os.walk(file_path):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(file_path)
        os.rmdir(os.getcwd()+"/output data")
        
        
        #rounding to account for differences due to float number system
        for i in range(len(doc_topics1)):
            for col in list_of_attributes:
                nums = [num for num in doc_topics1.iloc[i][col].strip("[]").split(" ") if len(num)>1]
                doc_topics1.at[i,col] = [truncate(float(num),2) for num in nums]
                nums = [num for num in doc_topics2.iloc[i][col].strip("[]").split(" ") if len(num)>1]
                doc_topics2.at[i,col] = [truncate(float(num),2) for num in nums]
                if doc_topics1.iloc[i][col] != doc_topics2.iloc[i][col]:
                    print(doc_topics1.iloc[i][col], doc_topics2.iloc[i][col])
                    print(type(doc_topics1.iloc[i][col]), type(doc_topics2.iloc[i][col]))
                    print(col,i)
        #checking we get the same output when using a bin object
        self.assertEqual(doc_topics1.equals(doc_topics2), True)
        self.assertEqual(tax1.equals(tax2), True)
        self.assertEqual(coherence_1.equals(coherence_2), True)
        self.assertEqual(level_1_tax1.equals(level_1_tax2), True)
        for attr in list_of_attributes:
            for i in range(len(topics1[attr])):
                for col in topics1[attr].columns:
                    if topics1[attr].iloc[i][col] != topics2[attr].iloc[i][col]:
                        print(topics1[attr].iloc[i][col], topics2[attr].iloc[i][col])
                        print(i,col)
            self.assertEqual(topics1[attr].equals(topics2[attr]), True, attr)
if __name__ == '__main__':
        unittest.main()
