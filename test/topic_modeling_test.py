# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:33:36 2021

@author: srandrad
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from mika.kd import Topic_Model_plus
from mika.utils import Data
import pandas as pd
import unittest

def clean_list(df, col):
            df_col = df[col].tolist()#[1:]
            df_col = [[j.strip("'") for j in i.strip("[]").split(", ") if len(j)>0] for i in df_col]
            return df_col

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
        #delete test folder/everything in it
        file_path = 'topic_model_results'
        if os.path.isdir(file_path) == True:
            for root, dirs, files in os.walk(file_path):
                for file in files:
                    os.remove(os.path.join(root, file))
            os.rmdir(file_path) 
        return 

    def test_lda_functions(self): # integration test
        #add label_lda_topics
        # add asserts for comparing the output csvs/dfs
        num_topics ={'Lesson(s) Learned':5, 'Driving Event':5, 'Recommendation(s)':5}
        test_lda = Topic_Model_plus(text_columns=self.test_data.text_columns, data=self.test_data)
        #test_lda.extract_preprocessed_data(csv_file_name)
        test_lda.lda(num_topics=num_topics)
        test_lda.save_lda_models()
        test_lda.save_lda_topics()
        test_lda.save_lda_document_topic_distribution()
        test_lda.save_lda_coherence()
        test_lda.save_lda_taxonomy()
        test_lda.save_lda_results()
        for col in self.test_data.text_columns:
            test_lda.lda_visual(col)
        file_path = test_lda.folder_path #path from saved
        
        doc_topics1 = pd.read_csv(os.path.join(file_path,"lda_topic_dist_per_doc.csv"), index_col=0)#.applymap(str)
        tax1 = pd.read_csv(os.path.join(file_path,"lda_taxonomy.csv"), index_col=0).applymap(str)
        coherence_1 = pd.read_csv(os.path.join(file_path,"lda_coherence.csv"), index_col=0).round(3).applymap(str)
        results1 = pd.read_excel(os.path.join(file_path,"lda_results.xlsx"), sheet_name=None)
        topics1 = {}
        for attr in self.test_data.text_columns:
            topics1[attr] = pd.read_csv(os.path.join(file_path,attr+"_lda_topics.csv"), index_col=0).round(3).applymap(str)

        #testing results sheets are correct
        pd.testing.assert_frame_equal(results1['taxonomy'].applymap(str), tax1)
        pd.testing.assert_frame_equal(results1['coherence'].round(3).applymap(str), coherence_1)
        pd.testing.assert_frame_equal(results1['document topic distribution'], doc_topics1)
        for attr in self.test_data.text_columns:
            pd.testing.assert_frame_equal(results1[attr].round(3).applymap(str), topics1[attr][results1[attr].columns])
        
        #testing functions from bin
        test_lda.lda_extract_models(file_path)
        test_lda.save_lda_topics()
        test_lda.save_lda_document_topic_distribution()
        test_lda.save_lda_coherence()
        test_lda.save_lda_taxonomy()
        test_lda.save_lda_results()
        for col in self.test_data.text_columns:
            test_lda.lda_visual(col)

        doc_topics2 = pd.read_csv(os.path.join(file_path,"lda_topic_dist_per_doc.csv"), index_col=0)#.applymap(str)
        tax2 = pd.read_csv(os.path.join(file_path,"lda_taxonomy.csv"), index_col=0).applymap(str)
        coherence_2 = pd.read_csv(os.path.join(file_path,"lda_coherence.csv"), index_col=0).round(3).applymap(str)
        results2 = pd.read_excel(os.path.join(file_path,"lda_results.xlsx"), sheet_name=None)
        topics2 = {}
        for attr in self.test_data.text_columns:
            topics2[attr] = pd.read_csv(os.path.join(file_path,attr+"_lda_topics.csv"), index_col=0).round(3).applymap(str)

        #delete test folder/everything in it
        for root, dirs, files in os.walk(file_path):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(file_path)
        
        #rounding to account for differences in float number system
        for i in range(len(doc_topics1)):
            for col in self.test_data.text_columns:
                nums = [num for num in doc_topics1.iloc[i][col].strip("[]").split(" ") if len(num)>1]
                doc_topics1.at[i,col] = [round(float(num),1) for num in nums]
                nums = [num for num in doc_topics2.iloc[i][col].strip("[]").split(" ") if len(num)>1]
                doc_topics2.at[i,col] = [round(float(num),1) for num in nums]
                
        #checking we get the same output when using a bin object
        pd.testing.assert_frame_equal(doc_topics1.round(3), doc_topics2.round(3), atol=0.5)
        self.assertEqual(tax1.equals(tax2), True, (tax1, tax2))
        self.assertEqual(coherence_1.equals(coherence_2), True)
        for attr in self.test_data.text_columns:
            self.assertEqual(topics1[attr].equals(topics2[attr]), True, attr) #this sometimes gives an failed test with the best document
        for sheet in results1:
            if sheet == "document topic distribution":
                for i in range(len(results1[sheet])):
                    for col in self.test_data.text_columns:
                        nums = [num for num in results1[sheet].iloc[i][col].strip("[]").split(" ") if len(num)>1]
                        results1[sheet].at[i,col] = [round(float(num),2) for num in nums]
                        nums = [num for num in results2[sheet].iloc[i][col].strip("[]").split(" ") if len(num)>1]
                        results2[sheet].at[i,col] = [round(float(num),2) for num in nums]
            pd.testing.assert_frame_equal(results1[sheet].round(3), results2[sheet].round(3), atol=0.5)
    
    def test_hlda_functions(self): # integration test
        #add label_hlda_topics, mixed_taxonomy
        test_hlda = Topic_Model_plus(text_columns=self.test_data.text_columns, data=self.test_data)
        test_hlda.hlda(training_iterations=100)
        test_hlda.save_hlda_models()
        test_hlda.save_hlda_document_topic_distribution()
        test_hlda.save_hlda_coherence()
        test_hlda.save_hlda_topics()
        test_hlda.save_hlda_level_n_taxonomy()
        test_hlda.save_hlda_taxonomy()
        test_hlda.save_hlda_results()
        for col in self.test_data.text_columns:
            test_hlda.hlda_display(col)
            test_hlda.hlda_visual(col)
        file_path = test_hlda.folder_path
        
        doc_topics1 = pd.read_csv(os.path.join(file_path,"hlda_topic_dist_per_doc.csv"), index_col=0)#.applymap(str)
        tax1 = pd.read_csv(os.path.join(file_path,"hlda_taxonomy.csv"), index_col=0).applymap(str)
        coherence_1 = pd.read_csv(os.path.join(file_path,"hlda_coherence.csv")).round(3).applymap(str)
        level_1_tax1 = pd.read_csv(os.path.join(file_path,"hlda_level1_taxonomy.csv"), index_col=0).applymap(str)
        topics1 = {}
        for attr in self.test_data.text_columns:
            topics1[attr] = pd.read_csv(os.path.join(file_path,attr+"_hlda_topics.csv"), index_col=0).round(3).applymap(str)
        results1 = pd.read_excel(os.path.join(file_path,"hlda_results.xlsx"), sheet_name=None)
        
        #testing results sheets are correct
        pd.testing.assert_frame_equal(results1['taxonomy'].applymap(str), tax1)
        pd.testing.assert_frame_equal(results1['coherence'].round(3).applymap(str), coherence_1.drop(["Unnamed: 0"], axis=1))
        pd.testing.assert_frame_equal(results1['document topic distribution'], doc_topics1)
        for attr in self.test_data.text_columns:
            pd.testing.assert_frame_equal(results1[attr].round(3).applymap(str), topics1[attr][results1[attr].columns])
        
        #testing functions from bin
        file_path = test_hlda.folder_path 
        test_hlda.hlda_extract_models(file_path)
        test_hlda.save_hlda_document_topic_distribution()
        test_hlda.save_hlda_coherence()
        test_hlda.save_hlda_taxonomy()
        test_hlda.save_hlda_topics()
        test_hlda.save_hlda_level_n_taxonomy()
        test_hlda.save_hlda_results()
        for col in self.test_data.text_columns:
            test_hlda.hlda_display(col)
            test_hlda.hlda_visual(col)
        
        doc_topics2 = pd.read_csv(os.path.join(file_path,"hlda_topic_dist_per_doc.csv"), index_col=0)#.applymap(str)
        tax2 = pd.read_csv(os.path.join(file_path,"hlda_taxonomy.csv"), index_col=0).applymap(str)
        coherence_2 = pd.read_csv(os.path.join(file_path,"hlda_coherence.csv")).round(3).applymap(str)
        level_1_tax2 = pd.read_csv(os.path.join(file_path,"hlda_level1_taxonomy.csv"), index_col=0).applymap(str)
        topics2 = {}
        for attr in self.test_data.text_columns:
            topics2[attr] = pd.read_csv(os.path.join(file_path,attr+"_hlda_topics.csv"), index_col=0).round(3).applymap(str)
        results2 = pd.read_excel(os.path.join(file_path,"hlda_results.xlsx"), sheet_name=None)
        
        #delete test folder/everything in it
        for root, dirs, files in os.walk(file_path):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(file_path)        
        
        #rounding to account for differences due to float number system
        for i in range(len(doc_topics1)):
            for col in self.test_data.text_columns:
                nums = [num for num in doc_topics1.iloc[i][col].strip("[]").split(" ") if len(num)>1]
                doc_topics1.at[i,col] = [round(float(num),2) for num in nums]
                nums = [num for num in doc_topics2.iloc[i][col].strip("[]").split(" ") if len(num)>1]
                doc_topics2.at[i,col] = [round(float(num),2) for num in nums]
                
        #checking we get the same output when using a bin object
        pd.testing.assert_frame_equal(doc_topics1.round(3), doc_topics2.round(3), atol=0.5)
        self.assertEqual(tax1.equals(tax2), True)
        self.assertEqual(coherence_1.drop(["Unnamed: 0"], axis=1).equals(coherence_2.drop(["Unnamed: 0"], axis=1)), True)
        self.assertEqual(level_1_tax1.equals(level_1_tax2), True)
        for attr in self.test_data.text_columns:
            self.assertEqual(topics1[attr].equals(topics2[attr]), True, attr) #this sometimes gives an failed test with the best document
        for sheet in results1:
            if sheet == "document topic distribution":
                for i in range(len(results1[sheet])):
                    for col in self.test_data.text_columns:
                        nums = [num for num in results1[sheet].iloc[i][col].strip("[]").split(" ") if len(num)>1]
                        results1[sheet].at[i,col] = [round(float(num),2) for num in nums]
                        nums = [num for num in results2[sheet].iloc[i][col].strip("[]").split(" ") if len(num)>1]
                        results2[sheet].at[i,col] = [round(float(num),2) for num in nums]
            pd.testing.assert_frame_equal(results1[sheet], results2[sheet], sheet)
    
    def test_bertopic_functions(self):
        test_bert = Topic_Model_plus(text_columns=self.raw_test_data.text_columns, data=self.raw_test_data)
        test_bert.bert_topic()
        
        test_bert.save_bert_model()
        test_bert.save_bert_document_topic_distribution()
        test_bert.save_bert_coherence()
        test_bert.save_bert_topics(coherence=True)
        test_bert.save_bert_topics_from_probs()
        test_bert.save_bert_taxonomy()
        test_bert.save_bert_topic_diversity()
        test_bert.save_bert_results(coherence=True)
        #test_bert.save_bert_vis() #too few topics for visualization?
        file_path = test_bert.folder_path
        # load original results
        doc_topics1 = pd.read_csv(os.path.join(file_path,"BERT_topic_dist_per_doc.csv"), index_col=0)#.applymap(str)
        coherence1 = pd.read_csv(os.path.join(file_path,"BERT_coherence_u_mass.csv"), index_col=0).round(3).applymap(str)
        topics1 = {}
        topics_from_probs1 = {}
        for attr in self.raw_test_data.text_columns:
            topics1[attr] = pd.read_csv(os.path.join(file_path,attr+"_BERT_topics.csv"), index_col=0).round(3).applymap(str)
            topics_from_probs1[attr] = pd.read_csv(os.path.join(file_path,attr+"_BERT_topics_modified.csv"), index_col=0).applymap(str)
            #check that the number of documents per topic is correct when using from_probs
            total_docs = clean_list(topics_from_probs1[attr], 'documents')
            number_of_docs = [int(i) for i in topics_from_probs1[attr]['number of documents in topic'].tolist()]
            for i in range(len(topics_from_probs1[attr])):
                self.assertEqual(len(total_docs[i]), number_of_docs[i])
            
        tax1 = pd.read_csv(os.path.join(file_path,"BERT_taxonomy.csv"), index_col=0).applymap(str)
        diversity1 =  pd.read_csv(os.path.join(file_path,"BERT_diversity.csv"), index_col=0).round(3).applymap(str)
        results1 =  pd.read_excel(os.path.join(file_path,"BERTopic_results.xlsx"), sheet_name=None)
        
        #testing results sheets are correct
        pd.testing.assert_frame_equal(results1['taxonomy'].applymap(str), tax1)
        pd.testing.assert_frame_equal(results1['coherence'].round(3).applymap(str), coherence1)
        pd.testing.assert_frame_equal(results1['document topic distribution'], doc_topics1)
        pd.testing.assert_frame_equal(results1['topic diversity'].round(3).applymap(str), diversity1)
        for attr in self.raw_test_data.text_columns:
            pd.testing.assert_frame_equal(results1[attr].round(3).applymap(str), topics1[attr][results1[attr].columns])
        
        #reduced topics
        test_bert.reduce_bert_topics(num=3, from_probs=True) #some kind of interaction between from probs and coherence????
        test_bert.save_bert_model()
        test_bert.save_bert_document_topic_distribution()
        test_bert.save_bert_coherence() #somethings wrong with the reduced coherence
        test_bert.save_bert_topics(coherence=True)
        test_bert.save_bert_topics_from_probs()
        test_bert.save_bert_taxonomy()
        test_bert.save_bert_topic_diversity() 
        test_bert.save_bert_results(coherence=True)
        #test_bert.save_bert_vis() #too few topics for visualization?
        
        # load reduced results
        doc_topics_reduced1 = pd.read_csv(os.path.join(file_path,"Reduced_BERT_topic_dist_per_doc.csv"), index_col=0)
        coherence_reduced1 = pd.read_csv(os.path.join(file_path,"Reduced_BERT_coherence_u_mass.csv"), index_col=0).round(3).applymap(str)
        topics_reduced1 = {}
        topics_from_probs_reduced1 = {}
        for attr in self.raw_test_data.text_columns:
            topics_reduced1[attr] = pd.read_csv(os.path.join(file_path,attr+"_Reduced_BERT_topics.csv"), index_col=0).round(3).applymap(str)
            topics_from_probs_reduced1[attr] = pd.read_csv(os.path.join(file_path,attr+"_reduced_BERT_topics_modified.csv"), index_col=0).round(3).applymap(str)
            #check that the number of documents per topic is correct when using from_probs
            total_docs = clean_list(topics_from_probs_reduced1[attr], 'documents')
            number_of_docs = [int(i) for i in topics_from_probs_reduced1[attr]['number of documents in topic'].tolist()]
            for i in range(len(topics_from_probs_reduced1[attr])):
                self.assertEqual(len(total_docs[i]), number_of_docs[i])
            
        tax_reduced1 = pd.read_csv(os.path.join(file_path,"Reduced_BERT_taxonomy.csv"), index_col=0).applymap(str)
        diversity_reduced1 =  pd.read_csv(os.path.join(file_path,"Reduced_BERT_diversity.csv"), index_col=0).round(3).applymap(str)
        results_reduced1 =  pd.read_excel(os.path.join(file_path,"Reduced_BERTopic_results.xlsx"), sheet_name=None)
        #testing results sheets are correct
        pd.testing.assert_frame_equal(results_reduced1['taxonomy'].applymap(str), tax_reduced1)
        pd.testing.assert_frame_equal(results_reduced1['coherence'].round(3).applymap(str), coherence_reduced1)
        pd.testing.assert_frame_equal(results_reduced1['document topic distribution'], doc_topics_reduced1)
        pd.testing.assert_frame_equal(results_reduced1['topic diversity'].round(3).applymap(str), diversity_reduced1)
        for attr in self.raw_test_data.text_columns:
            pd.testing.assert_frame_equal(results_reduced1[attr].round(3).applymap(str), topics_reduced1[attr][results_reduced1[attr].columns])
        
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
        doc_topics2 = pd.read_csv(os.path.join(file_path,"BERT_topic_dist_per_doc.csv"), index_col=0)#.applymap(str)
        coherence2 = pd.read_csv(os.path.join(file_path,"BERT_coherence_u_mass.csv"), index_col=0).round(3).applymap(str)
        topics2 = {}
        topics_from_probs2 = {}
        for attr in self.raw_test_data.text_columns:
            topics2[attr] = pd.read_csv(os.path.join(file_path,attr+"_BERT_topics.csv"), index_col=0).round(3).applymap(str)
            topics_from_probs2[attr] = pd.read_csv(os.path.join(file_path,attr+"_BERT_topics_modified.csv"), index_col=0).applymap(str)
        tax2 = pd.read_csv(os.path.join(file_path,"BERT_taxonomy.csv"), index_col=0).applymap(str)
        diversity2 =  pd.read_csv(os.path.join(file_path,"BERT_diversity.csv"), index_col=0).round(3).applymap(str)
        results2 =  pd.read_excel(os.path.join(file_path,"BERTopic_results.xlsx"), sheet_name=None)
        
        #reduced model
        test_bert.load_bert_model(file_path, reduced=True, from_probs=True)
        test_bert.save_bert_document_topic_distribution()
        test_bert.save_bert_coherence()
        test_bert.save_bert_topics(coherence=True)
        test_bert.save_bert_topics_from_probs()
        test_bert.save_bert_taxonomy()
        test_bert.save_bert_topic_diversity()
        test_bert.save_bert_results(coherence=True)
        #test_bert.save_bert_vis() #too few topics for visualization?
        # load reduced reuslts
        doc_topics_reduced2 = pd.read_csv(os.path.join(file_path,"Reduced_BERT_topic_dist_per_doc.csv"), index_col=0)#.applymap(str)
        coherence_reduced2 = pd.read_csv(os.path.join(file_path,"Reduced_BERT_coherence_u_mass.csv"), index_col=0).round(3).applymap(str)
        topics_reduced2 = {}
        topics_from_probs_reduced2 = {}
        for attr in self.raw_test_data.text_columns:
            topics_reduced2[attr] = pd.read_csv(os.path.join(file_path,attr+"_Reduced_BERT_topics.csv"), index_col=0).round(3).applymap(str)
            topics_from_probs_reduced2[attr] = pd.read_csv(os.path.join(file_path,attr+"_reduced_BERT_topics_modified.csv"), index_col=0).round(3).applymap(str)
        tax_reduced2 = pd.read_csv(os.path.join(file_path,"Reduced_BERT_taxonomy.csv"), index_col=0).applymap(str)
        diversity_reduced2 =  pd.read_csv(os.path.join(file_path,"Reduced_BERT_diversity.csv"), index_col=0).round(3).applymap(str)
        results_reduced2 =  pd.read_excel(os.path.join(file_path,"Reduced_BERTopic_results.xlsx"), sheet_name=None)

        #delete test folder/everything in it
        for root, dirs, files in os.walk(file_path):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(file_path)
                
        #checking we get the same output when using a bin object
        pd.testing.assert_frame_equal(doc_topics1.round(3), doc_topics2.round(3), atol=0.5)
        self.assertEqual(tax1.equals(tax2), True)
        self.assertEqual(coherence1.equals(coherence2), True)
        for attr in self.raw_test_data.text_columns:
            self.assertEqual(topics1[attr].equals(topics2[attr]), True, attr) #this sometimes gives an failed test with the best document
            self.assertEqual(topics_from_probs1[attr].equals(topics_from_probs2[attr]), True, attr)
        pd.testing.assert_frame_equal(diversity1, diversity2)
        for sheet in results1:
            self.assertTrue(results1[sheet].equals(results2[sheet]))

        #checking we get the same output when using a bin object
        pd.testing.assert_frame_equal(doc_topics_reduced1.round(3), doc_topics_reduced2.round(3), atol=0.5)
        self.assertEqual(tax_reduced1.equals(tax_reduced2), True)
        self.assertEqual(coherence_reduced1.equals(coherence_reduced2), True)
        for attr in self.raw_test_data.text_columns:
            pd.testing.assert_frame_equal(topics_reduced1[attr], topics_reduced2[attr])
            self.assertEqual(topics_from_probs_reduced1[attr].equals(topics_from_probs_reduced2[attr]), True, attr)
        pd.testing.assert_frame_equal(diversity_reduced1, diversity_reduced2)
        for sheet in results_reduced1:
            self.assertTrue(results_reduced1[sheet].equals(results_reduced2[sheet]))
        
if __name__ == '__main__':
    unittest.main()
