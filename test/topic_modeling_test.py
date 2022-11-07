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
        raw_text_columns = ['Raw '+col for col in text_columns]
        self.raw_test_data = Data()
        self.raw_test_data.load(csv_file_name, preprocessed=False, id_col=document_id_col, text_columns=raw_text_columns)
    
    def tearDown(self):
        return 
    def test_lda_functions(self): # integration test
        #add save_lda_results, label_lda_topics, lda_visual
        # add asserts for comparing the output csvs/dfs
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
                
        #checking we get the same output when using a bin object
        self.assertEqual(doc_topics1.equals(doc_topics2), True)
        self.assertEqual(tax1.equals(tax2), True)
        self.assertEqual(coherence_1.equals(coherence_2), True)
    
    def test_hlda_functions(self): # integration test
        #add label_hlda_topics, mixed_taxonomy, save_hlda_results, hlda_display?
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
                
        #checking we get the same output when using a bin object
        self.assertEqual(doc_topics1.equals(doc_topics2), True)
        self.assertEqual(tax1.equals(tax2), True)
        self.assertEqual(coherence_1.equals(coherence_2), True)
        self.assertEqual(level_1_tax1.equals(level_1_tax2), True)
        for attr in self.test_data.text_columns:
            self.assertEqual(topics1[attr].equals(topics2[attr]), True, attr) #this sometimes gives an failed test with the best document

    def test_bertopic_functions(self):
        #bert_topic, reduce_bert_topics, save_bert_topics_from_probs, save_bert_taxonomy
        #save_bert_document_topic_distribution, save_bert_results, save_bert_vis,
        #save_bert_coherence, save_bert_model, save_bert_topic_diversity
        test_bert = Topic_Model_plus(text_columns=self.raw_test_data.text_columns, data=self.raw_test_data)
        test_bert.bert_topic()
        test_bert.save_bert_model()
        test_bert.save_bert_document_topic_distribution()
        test_bert.save_bert_coherence()
        test_bert.save_bert_topics()
        test_bert.save_bert_topics_from_probs()
        test_bert.save_bert_taxonomy()
        test_bert.save_bert_topic_diversity()
        test_bert.save_bert_results()
        #test_bert.save_bert_vis() #too few topics for visualization?
        file_path = test_bert.folder_path
        #reduced topics
        test_bert.reduce_bert_topics(num=3)
        test_bert.save_bert_model()
        test_bert.save_bert_document_topic_distribution()
        test_bert.save_bert_coherence()
        test_bert.save_bert_topics()
        test_bert.save_bert_topics_from_probs()
        test_bert.save_bert_taxonomy()
        test_bert.save_bert_topic_diversity()
        test_bert.save_bert_results()
        #test_bert.save_bert_vis() #too few topics for visualization?
        # load original results
        doc_topics1 = pd.read_csv(os.path.join(file_path,"BERT_topic_dist_per_doc.csv"))#.applymap(str)
        coherence1 = pd.read_csv(os.path.join(file_path,"BERT_coherence_u_mass.csv")).applymap(str)
        topics1 = {}
        topics_from_probs1 = {}
        for attr in self.raw_test_data.text_columns:
            topics1[attr] = pd.read_csv(os.path.join(file_path,attr+"_BERT_topics.csv")).applymap(str)
            topics_from_probs1[attr] = pd.read_csv(os.path.join(file_path,attr+"_BERT_topics_modified.csv")).applymap(str)
        tax1 = pd.read_csv(os.path.join(file_path,"BERT_taxonomy.csv")).applymap(str)
        diversity1 =  pd.read_csv(os.path.join(file_path,"BERT_diversity.csv")).applymap(str)
        results1 =  pd.read_excel(os.path.join(file_path,"BERTopic_results.xlsx")).applymap(str)
        # load reduced reuslts
        doc_topics_reduced1 = pd.read_csv(os.path.join(file_path,"Reduced_BERT_topic_dist_per_doc.csv"))#.applymap(str)
        coherence_reduced1 = pd.read_csv(os.path.join(file_path,"Reduced_BERT_coherence_u_mass.csv")).applymap(str)
        topics_reduced1 = {}
        topics_from_probs_reduced1 = {}
        for attr in self.raw_test_data.text_columns:
            topics_reduced1[attr] = pd.read_csv(os.path.join(file_path,attr+"_Reduced_BERT_topics.csv")).applymap(str)
            topics_from_probs_reduced1[attr] = pd.read_csv(os.path.join(file_path,attr+"_reduced_BERT_topics_modified.csv")).applymap(str)
        tax_reduced1 = pd.read_csv(os.path.join(file_path,"Reduced_BERT_taxonomy.csv")).applymap(str)
        diversity_reduced1 =  pd.read_csv(os.path.join(file_path,"Reduced_BERT_diversity.csv")).applymap(str)
        results_reduced1 =  pd.read_excel(os.path.join(file_path,"Reduced_BERTopic_results.xlsx")).applymap(str)

        #testing results from saved model
        #normal model
        test_bert.load_bert_model(file_path, reduced=False, from_probs=True)
        test_bert.save_bert_document_topic_distribution()
        test_bert.save_bert_coherence()
        test_bert.save_bert_topics()
        test_bert.save_bert_topics_from_probs()
        test_bert.save_bert_taxonomy()
        test_bert.save_bert_topic_diversity()
        test_bert.save_bert_results()
        #test_bert.save_bert_vis() #too few topics for visualization?
        # load original results
        doc_topics2 = pd.read_csv(os.path.join(file_path,"BERT_topic_dist_per_doc.csv"))#.applymap(str)
        coherence2 = pd.read_csv(os.path.join(file_path,"BERT_coherence_u_mass.csv")).applymap(str)
        topics2 = {}
        topics_from_probs2 = {}
        for attr in self.raw_test_data.text_columns:
            topics2[attr] = pd.read_csv(os.path.join(file_path,attr+"_BERT_topics.csv")).applymap(str)
            topics_from_probs2[attr] = pd.read_csv(os.path.join(file_path,attr+"_BERT_topics_modified.csv")).applymap(str)
        tax2 = pd.read_csv(os.path.join(file_path,"BERT_taxonomy.csv")).applymap(str)
        diversity2 =  pd.read_csv(os.path.join(file_path,"BERT_diversity.csv")).applymap(str)
        results2 =  pd.read_excel(os.path.join(file_path,"BERTopic_results.xlsx")).applymap(str)
        
        #reduced model
        test_bert.load_bert_model(file_path, reduced=True, from_probs=True)
        test_bert.save_bert_document_topic_distribution()
        test_bert.save_bert_coherence()
        test_bert.save_bert_topics()
        test_bert.save_bert_topics_from_probs()
        test_bert.save_bert_taxonomy()
        test_bert.save_bert_topic_diversity()
        test_bert.save_bert_results()
        #test_bert.save_bert_vis() #too few topics for visualization?
        # load reduced reuslts
        doc_topics_reduced2 = pd.read_csv(os.path.join(file_path,"Reduced_BERT_topic_dist_per_doc.csv"))#.applymap(str)
        coherence_reduced2 = pd.read_csv(os.path.join(file_path,"Reduced_BERT_coherence_u_mass.csv")).applymap(str)
        topics_reduced2 = {}
        topics_from_probs_reduced2 = {}
        for attr in self.raw_test_data.text_columns:
            topics_reduced2[attr] = pd.read_csv(os.path.join(file_path,attr+"_Reduced_BERT_topics.csv")).applymap(str)
            topics_from_probs_reduced2[attr] = pd.read_csv(os.path.join(file_path,attr+"_reduced_BERT_topics_modified.csv")).applymap(str)
        tax_reduced2 = pd.read_csv(os.path.join(file_path,"Reduced_BERT_taxonomy.csv")).applymap(str)
        diversity_reduced2 =  pd.read_csv(os.path.join(file_path,"Reduced_BERT_diversity.csv")).applymap(str)
        results_reduced2 =  pd.read_excel(os.path.join(file_path,"Reduced_BERTopic_results.xlsx")).applymap(str)

        #delete test folder/everything in it
        for root, dirs, files in os.walk(file_path):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(file_path)

        #rounding to account for differences due to float number system
        for i in range(len(doc_topics1)):
            for col in self.raw_test_data.text_columns:
                nums = [num for num in doc_topics1.iloc[i][col].strip("[]").split(" ") if len(num)>1]
                doc_topics1.at[i,col] = [truncate(float(num),2) for num in nums]
                nums = [num for num in doc_topics2.iloc[i][col].strip("[]").split(" ") if len(num)>1]
                doc_topics2.at[i,col] = [truncate(float(num),2) for num in nums]
                
        #checking we get the same output when using a bin object
        self.assertEqual(doc_topics1.equals(doc_topics2), True)
        self.assertEqual(tax1.equals(tax2), True)
        self.assertEqual(coherence1.equals(coherence2), True)
        for attr in self.raw_test_data.text_columns:
            self.assertEqual(topics1[attr].equals(topics2[attr]), True, attr) #this sometimes gives an failed test with the best document
            self.assertEqual(topics_from_probs1[attr].equals(topics_from_probs2[attr]), True, attr)
        pd.testing.assert_frame_equal(diversity1, diversity2)
        for sheet in results1:
            self.assertTrue(results1[sheet].equals(results2[sheet]))
            
        #rounding to account for differences due to float number system
        for i in range(len(doc_topics_reduced1)):
            for col in self.raw_test_data.text_columns:
                nums = [num for num in doc_topics_reduced1.iloc[i][col].strip("[]").split(" ") if len(num)>1]
                doc_topics_reduced1.at[i,col] = [truncate(float(num),2) for num in nums]
                nums = [num for num in doc_topics_reduced2.iloc[i][col].strip("[]").split(" ") if len(num)>1]
                doc_topics_reduced2.at[i,col] = [truncate(float(num),2) for num in nums]
            
        #checking we get the same output when using a bin object
        self.assertEqual(doc_topics_reduced1.equals(doc_topics_reduced2), True)
        self.assertEqual(tax_reduced1.equals(tax_reduced2), True)
        self.assertEqual(coherence_reduced1.equals(coherence_reduced2), True)
        for attr in self.raw_test_data.text_columns:
            self.assertEqual(topics_reduced1[attr].equals(topics_reduced2[attr]), True, attr) #this sometimes gives an failed test with the best document
            self.assertEqual(topics_from_probs_reduced1[attr].equals(topics_from_probs_reduced2[attr]), True, attr)
        pd.testing.assert_frame_equal(diversity_reduced1, diversity_reduced2)
        for sheet in results_reduced1:
            self.assertTrue(results_reduced1[sheet].equals(results_reduced2[sheet]))
            
if __name__ == '__main__':
    unittest.main()
