# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:44:10 2021
integration test for lda functions. Run this to ensure updates do not break the lda functionality
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
    
class test_lda_methods(unittest.TestCase):
    """
    def test_lda(self):
        #run on preprocessed data
        #self.assertEqual(first, second)
        return
    def test_save_extract_lda_models(self):
        #run on preprocessed data
        #ensure the extracted == saved, make sure they both work
        return
    def test_save_lda_document_topic_distribution(self):
        #make sure it saves
        return
    def test_save_lda_coherence(self):
        #make sure it saves, use test object?
        return
    def test_lda_save_taxonomy(self):
        #make sure it saves, use test object for consistency
        return
    """
    def test_lda_functions(self): # integration test
    #add asserts for comaring the output csvs/dfs
        list_of_attributes = ['Lesson(s) Learned','Driving Event','Recommendation(s)']
        document_id_col = 'Lesson ID'
        csv_file_name = smart_nlp_path+"\input data\preprocessed_data.csv"
        num_topics ={'Lesson(s) Learned':5, 'Driving Event':5, 'Recommendation(s)':5}
        test_lda = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col)
        test_lda.extract_preprocessed_data(csv_file_name)
        test_lda.lda(num_topics=num_topics)
        test_lda.save_lda_models()
        test_lda.save_lda_document_topic_distribution()
        test_lda.save_lda_coherence()
        test_lda.save_lda_taxonomy()
        file_path = test_lda.folder_path #path from saved
        
        doc_topics1 = pd.read_csv(file_path+"/lda_topic_dist_per_doc.csv")#.applymap(str)
        tax1 = pd.read_csv(file_path+"/lda_taxonomy.csv").applymap(str)
        coherence_1 = pd.read_csv(file_path+"/"+"lda_coherence.csv").applymap(str)
        
        #testing functions from bin
        test_lda.lda_extract_models(file_path)
        test_lda.save_lda_document_topic_distribution()
        test_lda.save_lda_coherence()
        test_lda.save_lda_taxonomy()
        
        doc_topics2 = pd.read_csv(file_path+"/lda_topic_dist_per_doc.csv")#.applymap(str)
        tax2 = pd.read_csv(file_path+"/lda_taxonomy.csv").applymap(str)
        coherence_2 = pd.read_csv(file_path+"/"+"lda_coherence.csv").applymap(str)
        
        #delete test folder/everything in it
        for root, dirs, files in os.walk(file_path):
            for file in files:
                os.remove(os.path.join(root, file))
        #print("=========",file_path)
        os.rmdir(file_path)
        os.rmdir(os.getcwd()+"/output data")
        
        #rounding to account for differences in float number system
        for i in range(len(doc_topics1)):
            for col in list_of_attributes:
                nums = [num for num in doc_topics1.iloc[i][col].strip("[]").split(" ") if len(num)>1]
                doc_topics1.at[i,col] = [truncate(float(num),1) for num in nums]
                nums = [num for num in doc_topics2.iloc[i][col].strip("[]").split(" ") if len(num)>1]
                doc_topics2.at[i,col] = [truncate(float(num),1) for num in nums]
                if doc_topics1.iloc[i][col] != doc_topics2.iloc[i][col]:
                    print(doc_topics1.iloc[i][col], doc_topics2.iloc[i][col])
                    print(type(doc_topics1.iloc[i][col]), type(doc_topics2.iloc[i][col]))
                    print(col,i)
        #checking we get the same output when using a bin object
        self.assertEqual(doc_topics1.equals(doc_topics2), True)
        self.assertEqual(tax1.equals(tax2), True)
        self.assertEqual(coherence_1.equals(coherence_2), True)

if __name__ == '__main__':
    unittest.main()
