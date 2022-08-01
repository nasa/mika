# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 17:02:01 2022

@author: srandrad
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))

import unittest

from module.trend_analysis_functions import check_for_hazard_words, check_for_negation_words, get_hazard_info, get_results_info, set_up_docs_per_hazard_vars, get_hazard_df, get_hazard_topics
from module.trend_analysis_functions import get_hazard_doc_ids, get_topics_per_doc, get_hazard_topics_per_doc, get_hazard_words, get_negation_words, record_hazard_doc_info, get_doc_time, get_doc_text, identify_docs_per_hazard
import pandas as pd
import numpy as np

class test_identify_docs_per_hazard(unittest.TestCase):
    def setUp(self):
        #create hazard file (topic-focused sheet, hazard name, hazard words, topic numbers, negation words)
        self.hazard_df = pd.DataFrame({
            "Topic Number":["1, 2, 3", "4", "5, 6"],
            "Topic Level":[1, 1, 1],
            "Relevant hazard words":["broke, break, broken", "malfunction, part", "fail, failed"],
            "Negation words":[np.nan, "fix", "mitigate, not"],
            "Hazard Category":["Mechaincal", "Mechaincal", "Mechaincal"],
            "Hazard name":["Break", "Malfunction", "Failure"]})
        self.test_hazards = ["Break", "Malfunction", "Failure"]
        self.docs = ["US-1", "US-2", "US-3", "US-4"]
        self.hazard_file = "hazard_info_test.xlsx"
        with pd.ExcelWriter(self.hazard_file, engine='xlsxwriter') as writer:
                self.hazard_df.to_excel(writer, sheet_name = "topic-focused", index = False)
        self.hazard_info = {"topic-focused": self.hazard_df}
        #create preprocessed_df (time field, text field, id field)
        self.time_field = 'year'
        self.text_field = 'text'
        self.id_field = 'id'
        self.preprocessed_df = pd.DataFrame({
            self.id_field:["US-1", "US-2", "US-3", "US-4"],
            self.text_field:[["something broke onboard"], ["the controller malfunctioned"], 
                             ["the propeller was broken"], ["GPS on board failed"]],
            self.time_field:[2010, 2011, 2012, 2011]})
        #create results files (topic number, topic words, ids per topic, results text field)
        self.results_df = pd.DataFrame({
            'topic number':[0, 1, 2, 3, 4, 5, 6, 7],
            'topic words':["terrain, steep, access",
                           "broke, part, component",
                           "broke, break, mode",
                           "broken, wing, rotor",
                           "work, malfunction, part",
                           "mishap, fail, failed",
                           "failed, component, part",
                           "area, fire, resource"],
            'documents':["[US-1]",
                             "[US-2, US-1]",
                             "[US-3, US-1]",
                             "[US-2, US-1]",
                             "[US-2, US-4]",
                             "[US-4, US-1]",
                             "[US-2, US-4]",
                             "[US-3]"]})
        
        self.hazard_info = {"topic-focused": self.hazard_df}
        
        self.doc_topic_dist_field = "document topic distribution"
        self.result_text_field = 'text'
        self.doc_topic_dist_df = pd.DataFrame({ # filters US-3 out if thres>0.0, fine if thres=0.0
            "document number": ["US-1", "US-2", "US-3", "US-4"], #may need results text field
            self.result_text_field: ["0.9 0.8 0.8 0.8 0.1 0.1 0.1 0.0",
                              "0.9 0.8 0.8 0.8 0.8 0.1 0.1 0.0",
                              "0.9 0.001 0.01 0.0 0.1 0.1 0.1 0.0",
                              "0.9 0.8 0.8 0.8 0.1 0.9 0.9 0.0"]})
        self.results_csv_file = 'results_test.csv'
        self.results_df.to_csv(self.results_csv_file)
        self.results_dict = {self.result_text_field: self.results_df}
        self.results_excel_file = 'results_test.xlsx'
        with pd.ExcelWriter(self.results_excel_file, engine='xlsxwriter') as writer:
                self.results_df.to_excel(writer, sheet_name=self.result_text_field, index=False)
                self.doc_topic_dist_df.to_excel(writer, sheet_name=self.doc_topic_dist_field)
        #result text field different
        self.result_text_field_diff = 'result text'
        self.results_dict_diff = {self.result_text_field_diff: self.results_df}
        self.results_excel_file_diff = 'results_test_diff.xlsx'
        self.doc_topic_dist_df_diff = pd.DataFrame({ # filters US-3 out if thres>0.0, fine if thres=0.0
            self.id_field: ["US-1", "US-2", "US-3", "US-4"], #may need results text field
            self.result_text_field_diff: ["0.9 0.8 0.8 0.8 0.1 0.1 0.1 0.0",
                              "0.9 0.8 0.8 0.8 0.8 0.1 0.1 0.0",
                              "0.9 0.001 0.01 0.0 0.1 0.1 0.1 0.0",
                              "0.9 0.8 0.8 0.8 0.1 0.9 0.9 0.0"]})
        with pd.ExcelWriter(self.results_excel_file_diff, engine='xlsxwriter') as writer:
                self.results_df.to_excel(writer, sheet_name=self.result_text_field_diff, index=False)
                self.doc_topic_dist_df_diff.to_excel(writer, sheet_name=self.doc_topic_dist_field)
        #BERT variants - topic number begins at -1
        self.results_df_BERT = self.results_df.copy()
        self.results_df_BERT['topic number'] = [-1, 0, 1, 2, 3, 4, 5, 6]
        self.hazard_df_BERT = self.hazard_df.copy()
        self.hazard_df_BERT["Topic Number"] = ["0, 1, 2", "3", "4, 5"]
        self.results_dict_BERT = {self.result_text_field: self.results_df_BERT}
        self.hazard_file_BERT = "hazard_info_BERT_test.xlsx"
        with pd.ExcelWriter(self.hazard_file_BERT, engine='xlsxwriter') as writer:
                self.hazard_df.to_excel(writer, sheet_name = "topic-focused", index = False)
        self.results_csv_file_BERT = 'results_BERT_test.csv'
        self.results_df_BERT.to_csv(self.results_csv_file_BERT)
        self.results_dict_BERT = {self.result_text_field: self.results_df_BERT}
        self.results_excel_file_BERT = 'results_BERT_test.xlsx'
        with pd.ExcelWriter(self.results_excel_file_BERT, engine='xlsxwriter') as writer:
                self.results_df_BERT.to_excel(writer, sheet_name=self.result_text_field, index=False)
                self.doc_topic_dist_df.to_excel(writer, sheet_name=self.doc_topic_dist_field)
        #
        self.ids_per_hazard = [["US-1", "US-2", "US-3"],
                               ["US-2", "US-4"],
                               ["US-1", "US-2", "US-4"]]
        #
        self.hazard_nums = [[1,2,3], [4], [5,6]]
        #final results from this set up:
        self.frequency = {"Break":{'2010':1, '2011':0, '2012':1},
                          "Malfunction":{'2010':0, '2011':1, '2012':0},
                          "Failure":{'2010':0, '2011':1, '2012':0}}
        self.docs_per_hazard = {"Break":{'2010':["US-1"], '2011':[], '2012':["US-3"]},
                          "Malfunction":{'2010':[], '2011':["US-2"], '2012':[]},
                          "Failure":{'2010':[], '2011':["US-4"], '2012':[]}}
        self.hazard_words_per_doc = {"Break":['broke', 'none', 'broke', 'none'],
                                    "Malfunction": ['none', 'malfunction', 'none', 'none'],
                                    "Failure":['none', 'none', 'none', 'fail']}
        self.topics_per_doc = {"US-1": [0, 1, 2, 3, 5],
                               "US-2": [1, 3, 4, 6],
                               "US-3": [2, 7],
                               "US-4": [4, 5, 6]}
        self.blank_hazard_topics_per_doc = {"US-1": {"Break":[],
                                               "Malfunction": [],
                                               "Failure":[]},
                               "US-2":{"Break":[],
                                       "Malfunction": [],
                                       "Failure":[]},
                               "US-3": {"Break":[],
                                        "Malfunction": [],
                                        "Failure":[]},
                               "US-4": {"Break":[],
                                        "Malfunction": [],
                                        "Failure":[]}}
        self.hazard_topics_per_doc = {"US-1": {"Break":[1, 2, 3],
                                               "Malfunction": [],
                                               "Failure":[5]},
                               "US-2":{"Break":[1, 3],
                                       "Malfunction": [4],
                                       "Failure":[6]},
                               "US-3": {"Break":[2],
                                        "Malfunction": [],
                                        "Failure":[]},
                               "US-4": {"Break":[],
                                        "Malfunction": [4],
                                        "Failure":[5, 6]}}
        
    def tearDown(self):
        os.remove(self.hazard_file)
        os.remove(self.results_csv_file)
        os.remove(self.results_excel_file)
        os.remove(self.results_excel_file_BERT)
        os.remove(self.results_csv_file_BERT)
        os.remove(self.hazard_file_BERT)
        os.remove(self.results_excel_file_diff)
    
    def test_check_for_hazard_words(self):
        """Intended functionality: returns true if h_word is in text, false otherwise"""
        h_word = 'hazard'
        text = 'there is a hazard'
        self.assertTrue(check_for_hazard_words(h_word, text))
        h_word = 'none'
        self.assertFalse(check_for_hazard_words(h_word, text))
        h_word = 'hazardous event'
        text = 'there is a hazardous event'
        self.assertTrue(check_for_hazard_words(h_word, text))
        h_word = 'nonhazardous event'
        self.assertFalse(check_for_hazard_words(h_word, text))
        h_word = 'hazard'
        text = 'there is a hazardous event'
        self.assertTrue(check_for_hazard_words(h_word, text))
    
    def test_check_for_negation_words(self):
        """Intended functionality: 
            returns true if:
                - negation words is empty
                - no negation words in text
                - negation word is greater than 3 spaces from h_word
                - negation word has punctuation between h_word
                - there are more hazard words than negation words
            returns false if:
                - negation word in text, within 3 spaces from h_word, with no punctuation within distance"""
        #Case 1: negation words is empty
        h_word = 'hazard'
        negation_words = []
        text = 'there is a hazard'
        self.assertTrue(check_for_negation_words(negation_words, text, h_word))
        #Case 2: negation word not in text
        h_word = 'hazard'
        negation_words = ['non']
        text = 'there is a hazard'
        self.assertTrue(check_for_negation_words(negation_words, text, h_word))
        h_word = 'hazard'
        negation_words = ['non', 'never', 'no']
        text = 'there is a hazard'
        self.assertTrue(check_for_negation_words(negation_words, text, h_word))
        #Case 3: negation word is greater than 3 spaces from h_word
        h_word = 'hazard'
        negation_words = ['mitigate']
        text = 'there was a hazard and it was not mitigated'
        self.assertTrue(check_for_negation_words(negation_words, text, h_word))
        #Case 4: negation word has punctionatino
        h_word = 'hazard'
        negation_words = ['mitigating']
        text = 'there were multiple hazards. Mitigating is hard.'
        self.assertTrue(check_for_negation_words(negation_words, text, h_word))
        #Case 5: there are more hazard words than negation words
        h_word = 'hazard'
        negation_words = ['mitigate']
        text = 'there were multiple hazards. One hazard was mitigated.'
        self.assertTrue(check_for_negation_words(negation_words, text, h_word))
        #Case 6: negation word in text, within 3 spaces, no punctuation, etc.
        h_word = 'hazard'
        negation_words = ['mitigate', 'placeholder']
        text = 'The hazard was mitigated.'
        self.assertFalse(check_for_negation_words(negation_words, text, h_word))
    
    def test_get_hazard_info(self):
        hazard_info, hazards = get_hazard_info(self.hazard_file)
        self.assertEqual(hazards, self.test_hazards)
        for val in hazard_info:
            pd.testing.assert_frame_equal(hazard_info[val], self.hazard_info[val])
            
    def test_get_results_info(self):
        """Cases:
            - .csv
            - .xlsx
            - doc_topic_dist_field = None
            - doc_topic_dist_field != None
            - results_text_field = None
            - results_text_field != None
            - topic nums begin at 0
            - topic nums begin at -1 """
        #Case 1: .csv, doc_topic = None, begin nums = 0
        results, results_text_field, doc_topic_distribution, begin_nums = get_results_info(self.results_csv_file, self.result_text_field, self.text_field, doc_topic_dist_field=None)
        for val in results:
            pd.testing.assert_frame_equal(results[val], self.results_dict[val])
        self.assertEqual(results_text_field, self.result_text_field)
        self.assertEqual(doc_topic_distribution, None)
        self.assertEqual(begin_nums, 0)
        #Case 1.2: .csv, doc_topic = None, begin nums=1
        results, results_text_field, doc_topic_distribution, begin_nums = get_results_info(self.results_csv_file_BERT, self.result_text_field, self.text_field, doc_topic_dist_field=None)
        for val in results:
            pd.testing.assert_frame_equal(results[val], self.results_dict_BERT[val])
        self.assertEqual(results_text_field, self.result_text_field)
        self.assertEqual(doc_topic_distribution, None)
        self.assertEqual(begin_nums, 1)
        #Case 2: .csv, doc_topic_dist_field != None for input, begin nums = 0
        #expected behavior: turn doc_topic_dist_field = None since this value only works for excels
        results, results_text_field, doc_topic_distribution, begin_nums = get_results_info(self.results_csv_file, self.result_text_field, self.text_field, doc_topic_dist_field=self.doc_topic_dist_field)
        for val in results:
            pd.testing.assert_frame_equal(results[val], self.results_dict[val])
        self.assertEqual(results_text_field, self.result_text_field)
        self.assertEqual(doc_topic_distribution, None)
        self.assertEqual(begin_nums, 0)
        #Case 3: .csv, results text field is missing, results text field same as text field
        results, results_text_field, doc_topic_distribution, begin_nums = get_results_info(self.results_csv_file, None, self.text_field, doc_topic_dist_field=self.doc_topic_dist_field)
        for val in results:
            pd.testing.assert_frame_equal(results[val], self.results_dict[val])
        self.assertEqual(results_text_field, self.text_field)
        self.assertEqual(doc_topic_distribution, None)
        self.assertEqual(begin_nums, 0)
        #Case 4: .xlsx, doc_topic = None, begin nums = 0
        results, results_text_field, doc_topic_distribution, begin_nums = get_results_info(self.results_excel_file, self.result_text_field, self.text_field, doc_topic_dist_field=None)
        for val in results:
            pd.testing.assert_frame_equal(results[val], self.results_dict[val])
        self.assertEqual(results_text_field, self.result_text_field)
        self.assertEqual(doc_topic_distribution, None)
        self.assertEqual(begin_nums, 0)
        #Case 5: .xlsx doc_topic != None, begin nums = 0
        results, results_text_field, doc_topic_distribution, begin_nums = get_results_info(self.results_excel_file, self.result_text_field, self.text_field, doc_topic_dist_field=self.doc_topic_dist_field)
        for val in results:
            pd.testing.assert_frame_equal(results[val], self.results_dict[val])
        self.assertEqual(results_text_field, self.result_text_field)
        pd.testing.assert_frame_equal(doc_topic_distribution.drop(["Unnamed: 0"], axis=1), self.doc_topic_dist_df)
        self.assertEqual(begin_nums, 0)
        #Case 6: .xlsx results text field different from text field
        results, results_text_field, doc_topic_distribution, begin_nums = get_results_info(self.results_excel_file_diff, self.result_text_field_diff, self.text_field, doc_topic_dist_field=self.doc_topic_dist_field)
        for val in results:
            pd.testing.assert_frame_equal(results[val], self.results_dict_diff[val])
        self.assertEqual(results_text_field, self.result_text_field_diff)
        pd.testing.assert_frame_equal(doc_topic_distribution.drop(["Unnamed: 0"], axis=1), self.doc_topic_dist_df_diff)
        self.assertEqual(begin_nums, 0)
        #Case 7: topic_nums begin at -1
        results, results_text_field, doc_topic_distribution, begin_nums = get_results_info(self.results_excel_file_BERT, self.result_text_field, self.text_field, doc_topic_dist_field=None)
        for val in results:
            pd.testing.assert_frame_equal(results[val], self.results_dict_BERT[val])
        self.assertEqual(results_text_field, self.result_text_field)
        self.assertEqual(doc_topic_distribution, None)
        self.assertEqual(begin_nums, 1)
    
    @unittest.expectedFailure
    def test_get_results_info_failure(self):
        """Test should fail """
        #Case 1: .xlsx results text field is missing, results text field different from text field - gives error
        results, results_text_field, doc_topic_distribution, begin_nums = get_results_info(self.results_excel_file_diff, None, self.text_field, doc_topic_dist_field=self.doc_topic_dist_field)
        for val in results:
            pd.testing.assert_frame_equal(results[val], self.results_dict_diff[val])
        self.assertEqual(results_text_field, self.result_text_field_diff)
        self.assertEqual(doc_topic_distribution, self.doc_topic_dist_df_diff)
        self.assertEqual(begin_nums, 0)

    def test_set_up_docs_per_hazard_vars(self):
        true_docs = self.docs
        true_frequency = {"Break":{"2010":0, "2011":0, "2012":0},
                          "Malfunction":{"2010":0, "2011":0, "2012":0},
                          "Failure":{"2010":0, "2011":0, "2012":0}}
        true_docs_per_hazard = {"Break":{"2010":[], "2011":[], "2012":[]},
                          "Malfunction":{"2010":[], "2011":[], "2012":[]},
                          "Failure":{"2010":[], "2011":[], "2012":[]}}
        true_hazard_words_per_doc = {"Break":['none', 'none', 'none', 'none'],
                          "Malfunction":['none', 'none', 'none', 'none'],
                          "Failure":['none', 'none', 'none', 'none']}
        docs, frequency, docs_per_hazard, hazard_words_per_doc = set_up_docs_per_hazard_vars(self.preprocessed_df, self.id_field, self.test_hazards, self.time_field)
        self.assertEqual(docs, true_docs)
        self.assertDictEqual(frequency, true_frequency)
        self.assertDictEqual(docs_per_hazard, true_docs_per_hazard)
        self.assertDictEqual(hazard_words_per_doc, true_hazard_words_per_doc)
        
    def test_get_hazard_df(self):
        i = 0 #first hazard
        hazard_df, hazard_name = get_hazard_df(self.hazard_info, self.test_hazards, i)
        self.assertEqual(hazard_name, "Break")
        pd.testing.assert_frame_equal(hazard_df, self.hazard_df.loc[self.hazard_df['Hazard name']=="Break"].reset_index(drop=True))
        i = 1 #second hazard
        hazard_df, hazard_name = get_hazard_df(self.hazard_info, self.test_hazards, i)
        self.assertEqual(hazard_name, "Malfunction")
        pd.testing.assert_frame_equal(hazard_df, self.hazard_df.loc[self.hazard_df['Hazard name']=="Malfunction"].reset_index(drop=True))
        i = 2 #third hazard
        hazard_df, hazard_name = get_hazard_df(self.hazard_info, self.test_hazards, i)
        self.assertEqual(hazard_name, "Failure")
        pd.testing.assert_frame_equal(hazard_df, self.hazard_df.loc[self.hazard_df['Hazard name']=="Failure"].reset_index(drop=True))
    
    def test_get_hazard_topics(self):
        """Expected functionality:
            - gives the indices for the topics (nums) in the results df"""
        hazard_df = self.hazard_df.loc[self.hazard_df['Hazard name']=="Break"].reset_index(drop=True)
        true_nums = [1, 2, 3]
        begin_nums = 0
        nums = get_hazard_topics(hazard_df, begin_nums)
        self.assertEqual(nums, true_nums)
        hazard_df = self.hazard_df.loc[self.hazard_df['Hazard name']=="Malfunction"].reset_index(drop=True)
        true_nums = [4]
        nums = get_hazard_topics(hazard_df, begin_nums)
        self.assertEqual(nums, true_nums)
        hazard_df = self.hazard_df.loc[self.hazard_df['Hazard name']=="Failure"].reset_index(drop=True)
        true_nums = [5, 6]
        nums = get_hazard_topics(hazard_df, begin_nums)
        self.assertEqual(nums, true_nums)
        # bert, begin_nums = 1 
        hazard_df = self.hazard_df_BERT.loc[self.hazard_df['Hazard name']=="Break"].reset_index(drop=True)
        true_nums = [1, 2, 3]
        begin_nums = 1
        nums = get_hazard_topics(hazard_df, begin_nums)
        self.assertEqual(nums, true_nums)
        hazard_df = self.hazard_df_BERT.loc[self.hazard_df['Hazard name']=="Malfunction"].reset_index(drop=True)
        true_nums = [4]
        nums = get_hazard_topics(hazard_df, begin_nums)
        self.assertEqual(nums, true_nums)
        hazard_df = self.hazard_df_BERT.loc[self.hazard_df['Hazard name']=="Failure"].reset_index(drop=True)
        true_nums = [5, 6]
        nums = get_hazard_topics(hazard_df, begin_nums)
        self.assertEqual(nums, true_nums)

    def test_get_hazard_doc_ids(self):
        """Cases:
            - doc_topic_distribution is None
            - doc_topic_distribution != None
            - topic thresh == 0.0
            - topic thresh > 0.0 
            Note that all outputs should be the same unless topic thresh > 0 and doc_topic_distribution != None"""
        #case 1: doc_topic_distribution is None, topic_thres = 0
        nums = [1, 2, 3]
        temp_df, ids = get_hazard_doc_ids(nums, results=self.results_dict, results_text_field=self.result_text_field, 
                                          docs=self.docs, doc_topic_distribution=None, text_field=self.text_field, topic_thresh=0.0, 
                                          preprocessed_df=self.preprocessed_df, id_field=self.id_field)
        self.assertEqual(ids, ["US-1", "US-2", "US-3"])
        #case 2: doc_topic_distribution is None, topic_thresh>0
        nums = [1, 2, 3]
        temp_df, ids = get_hazard_doc_ids(nums, results=self.results_dict, results_text_field=self.result_text_field, 
                                          docs=self.docs, doc_topic_distribution=None, text_field=self.text_field, topic_thresh=0.1, 
                                          preprocessed_df=self.preprocessed_df, id_field=self.id_field)
        self.assertEqual(ids, ["US-1", "US-2", "US-3"])
        #case 3: doc_topic_distribution != None, topic_thresh=0
        nums = [1, 2, 3]
        temp_df, ids = get_hazard_doc_ids(nums, results=self.results_dict, results_text_field=self.result_text_field, 
                                          docs=self.docs, doc_topic_distribution=self.doc_topic_dist_df, text_field=self.text_field, topic_thresh=0.0, 
                                          preprocessed_df=self.preprocessed_df, id_field=self.id_field)
        self.assertEqual(ids, ["US-1", "US-2", "US-3"])
        #case 3: doc_topic_distribution != None, topic_thresh>0 -> filters out US-3
        nums = [1, 2, 3]
        temp_df, ids = get_hazard_doc_ids(nums, results=self.results_dict, results_text_field=self.result_text_field, 
                                          docs=self.docs, doc_topic_distribution=self.doc_topic_dist_df, text_field=self.text_field, topic_thresh=0.1, 
                                          preprocessed_df=self.preprocessed_df, id_field=self.id_field)
        self.assertEqual(ids, ["US-1", "US-2"])
        
    def test_get_topics_per_doc(self):
        topics_per_doc, hazard_topics_per_doc = get_topics_per_doc(self.docs, self.results_dict, self.result_text_field, self.test_hazards) 
        self.assertDictEqual(topics_per_doc, self.topics_per_doc)
        self.assertDictEqual(hazard_topics_per_doc, self.blank_hazard_topics_per_doc)
        
    def test_get_hazard_topics_per_doc(self):
        """Cases:
            - begin nums = 0
            - begin nums = 1"""
        final_hazard_topics_per_doc = self.blank_hazard_topics_per_doc.copy()
        for hazard in self.test_hazards:
            nums = self.hazard_nums[self.test_hazards.index(hazard)]
            ids = self.ids_per_hazard[self.test_hazards.index(hazard)]
            hazard_topics_per_doc = get_hazard_topics_per_doc(ids, self.topics_per_doc, final_hazard_topics_per_doc, hazard_name=hazard, nums=nums, begin_nums=0)
            final_hazard_topics_per_doc.update(hazard_topics_per_doc)
        self.assertDictEqual(final_hazard_topics_per_doc, self.hazard_topics_per_doc)
    
    def test_get_hazard_words(self):
        hazard_df = self.hazard_df.loc[self.hazard_df['Hazard name']=="Break"].reset_index(drop=True)
        true_hazard_words = ["broke", "break", "broken"]
        hazard_words = get_hazard_words(hazard_df)
        self.assertEqual(hazard_words, true_hazard_words)
    
    def test_get_negation_words(self):
        hazard_df = self.hazard_df.loc[self.hazard_df['Hazard name']=="Break"].reset_index(drop=True)
        negation_words = get_negation_words(hazard_df) #"Negation words":[np.nan, "fix", "mitigate, not"],
        self.assertEqual(negation_words, [])
        hazard_df = self.hazard_df.loc[self.hazard_df['Hazard name']=="Malfunction"].reset_index(drop=True)
        negation_words = get_negation_words(hazard_df) #"Negation words":[np.nan, "fix", "mitigate, not"],
        self.assertEqual(negation_words, ['fix'])
        hazard_df = self.hazard_df.loc[self.hazard_df['Hazard name']=="Failure"].reset_index(drop=True)
        negation_words = get_negation_words(hazard_df) #"Negation words":[np.nan, "fix", "mitigate, not"],
        self.assertEqual(negation_words, ['mitigate', 'not'])
    
    def test_record_hazard_doc_info(self):
        hazard_name = "Break" 
        year =  2010
        id_ = "US-1"
        docs = self.docs
        h_word = 'broke'
        frequency = {"Break":{"2010":0, "2011":0, "2012":0},
                          "Malfunction":{"2010":0, "2011":0, "2012":0},
                          "Failure":{"2010":0, "2011":0, "2012":0}}
        docs_per_hazard = {"Break":{"2010":[], "2011":[], "2012":[]},
                          "Malfunction":{"2010":[], "2011":[], "2012":[]},
                          "Failure":{"2010":[], "2011":[], "2012":[]}}
        hazard_words_per_doc = {"Break":['none', 'none', 'none', 'none'],
                          "Malfunction":['none', 'none', 'none', 'none'],
                          "Failure":['none', 'none', 'none', 'none']}
        docs_per_hazard, frequency, hazard_words_per_doc = record_hazard_doc_info(hazard_name, year, docs_per_hazard, id_, frequency, hazard_words_per_doc, docs, h_word)
        true_docs_per_hazard = {"Break":{"2010":["US-1"], "2011":[], "2012":[]},
                          "Malfunction":{"2010":[], "2011":[], "2012":[]},
                          "Failure":{"2010":[], "2011":[], "2012":[]}}
        true_frequency = {"Break":{"2010":1, "2011":0, "2012":0},
                          "Malfunction":{"2010":0, "2011":0, "2012":0},
                          "Failure":{"2010":0, "2011":0, "2012":0}}
        true_hazard_words_per_doc = {"Break":['broke', 'none', 'none', 'none'],
                           "Malfunction":['none', 'none', 'none', 'none'],
                           "Failure":['none', 'none', 'none', 'none']}
        self.assertDictEqual(docs_per_hazard, true_docs_per_hazard)
        self.assertDictEqual(frequency, true_frequency)
        self.assertDictEqual(hazard_words_per_doc,true_hazard_words_per_doc)
    
    def test_get_doc_time(self):
        id_ = "US-1"
        temp_df = self.preprocessed_df
        year = get_doc_time(id_, temp_df, self.id_field, self.time_field)
        self.assertEqual(year, 2010)
    
    def test_get_doc_text(self):
        id_ = "US-1"
        temp_df = self.preprocessed_df
        text = get_doc_text(id_, temp_df, self.id_field, self.text_field)
        self.assertEqual(text, "something broke onboard")

    def test_integration_identify_docs_per_hazard(self):
        #csv file
        frequency, docs_per_hazard, hazard_words_per_doc, topics_per_doc, hazard_topics_per_doc = identify_docs_per_hazard(self.hazard_file, self.preprocessed_df, self.results_csv_file, self.text_field, self.time_field, self.id_field, results_text_field=None, doc_topic_dist_field=None, topic_thresh=0.0)
        self.assertDictEqual(frequency, self.frequency)
        self.assertDictEqual(docs_per_hazard, self.docs_per_hazard)
        for hazard in hazard_words_per_doc:
            self.assertEqual(hazard_words_per_doc[hazard], self.hazard_words_per_doc[hazard])
        self.assertDictEqual(topics_per_doc, self.topics_per_doc)
        self.assertDictEqual(hazard_topics_per_doc, self.hazard_topics_per_doc)
        #excel file
        frequency, docs_per_hazard, hazard_words_per_doc, topics_per_doc, hazard_topics_per_doc = identify_docs_per_hazard(self.hazard_file, self.preprocessed_df, self.results_excel_file, self.text_field, self.time_field, self.id_field, results_text_field=None, doc_topic_dist_field=None, topic_thresh=0.0)
        self.assertDictEqual(frequency, self.frequency)
        self.assertDictEqual(docs_per_hazard, self.docs_per_hazard)
        for hazard in hazard_words_per_doc:
            self.assertEqual(hazard_words_per_doc[hazard], self.hazard_words_per_doc[hazard])
        self.assertDictEqual(topics_per_doc, self.topics_per_doc)
        self.assertDictEqual(hazard_topics_per_doc, self.hazard_topics_per_doc)
        
if __name__ == '__main__':
    unittest.main()