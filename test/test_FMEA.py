# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 12:39:16 2022

@author: srandrad
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))

import unittest

from mika.kd import FMEA
import pandas as pd
import numpy as np

class test_FMEA(unittest.TestCase):
    def setUp(self):
        self.test_class = FMEA()
        text_list = ['the quick brown fox jumps over the lazy dog',
                     'Hello, world!',
                     'How vexingly quick daft zebras jump!',
                     'the lazy dog slept all day',
                     'the kangaroo jumps over the bush']
        category_list = ['Animal', 'CS', 'Animal', 'Animal', 'Animal']
        date = ['2020', '2020', '2020', '2021', '2021']
        self.test_df = pd.DataFrame({"text":text_list,
                                    "category":category_list,
                                    "id":[1,2,3,4,5],
                                    "additional":[100, 100, 101, 102, 101],
                                    "date":date})
        self.test_filename = "data_test.csv"
        self.test_id_col = 'id'
        self.test_text_col = 'text'
        self.test_grouping_col = 'category'
        self.test_dates = 'date'
        self.test_additional_cols = ["additional"]
        self.test_df.to_csv(self.test_filename)
        self.test_model_checkpoint = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),"models", "FMEA-ner-model", "checkpoint-1424")
    
    def tearDown(self):
        os.remove(self.test_filename)

    def test_load_data(self):
        self.test_class.load_data(filepath=self.test_filename, formatted=False, text_col=self.test_text_col, id_col=self.test_id_col, label_col=None)
        return

    def test_load_model(self):
        self.test_class.load_model(self.test_model_checkpoint)

    def test_FMEA_integration(self):
        self.test_class.load_data(filepath=self.test_filename, formatted=False, text_col=self.test_text_col, id_col=self.test_id_col, label_col=None)
        self.test_class.load_model(self.test_model_checkpoint)
        self.test_class.predict()
        df = self.test_class.get_entities_per_doc()
        self.test_class.group_docs_with_meta(self.test_grouping_col, additional_cols=self.test_additional_cols)
        def severity_func(df):
            sev = [1 for i in range(len(df))]
            df['severity'] = sev
            return df
        self.test_class.calc_severity(severity_func, from_file=False)
        self.test_class.calc_frequency(year_col=self.test_dates)
        self.test_class.calc_risk()
        self.test_class.post_process_fmea(phase_name='additional', id_name='test', max_words=1)
        fmea_1 = self.test_class.fmea_df.copy()
        self.setUp()
        self.test_class.load_data(filepath=self.test_filename, formatted=False, text_col=self.test_text_col, id_col=self.test_id_col, label_col=None)
        self.test_class.load_model(self.test_model_checkpoint)
        self.test_class.predict()
        self.test_class.build_fmea(severity_func, group_by='meta', year_col='date', group_by_kwargs={'grouping_col':self.test_grouping_col, 'additional_cols':self.test_additional_cols}, post_process_kwargs={'phase_name':'additional', 'id_name':'test', 'max_words':1}, save=True)
        os.remove('fmea.csv')
        pd.testing.assert_frame_equal(fmea_1.drop(['test'], axis=1), self.test_class.fmea_df.drop(['test'], axis=1))
if __name__ == '__main__':
    unittest.main()