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
        self.test_df = pd.DataFrame({"text":text_list,
                                "id":[1,2,3,4,5]})
        self.test_filename = "data_test.csv"
        self.test_id_col = 'id'
        self.test_text_col = 'text'
        self.test_df.to_csv(self.test_filename)
    def tearDown(self):
        os.remove(self.test_filename)
    def test_load_data(self):
        self.test_class.load_data(self.test_filename, formatted=False, text_col=self.test_text_col, id_col=self.test_id_col, label_col=None)
        return
    def test_load_model(self):
        self.test_class.load_model()

if __name__ == '__main__':
    unittest.main()