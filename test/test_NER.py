# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 12:39:28 2022

@author: srandrad
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))

import unittest

from mika.kd.NER import *
import pandas as pd
import numpy as np

class test_NER(unittest.TestCase):
    def setUp(self):
        return
    def tearDown(self):
        return
    