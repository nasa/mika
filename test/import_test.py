# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:07:58 2022

@author: srandrad
"""

import sys
import os
from importlib import import_module
import unittest
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))

from mika.kd import FMEA
from mika.kd import Topic_Model_plus
from mika.kd.trend_analysis import *
from mika.kd.NER import *
from mika.ir import search

from mika.utils import Data
from mika.utils.SAFECOM import *
from mika.utils.SAFENET import *
from mika.utils.LLIS import *
from mika.utils.ICS import *


class test_Imports(unittest.TestCase):
    def test_util_imports(self):
        import_module('mika.utils.ICS')
        import_module('mika.utils.SAFECOM')
        import_module('mika.utils.SAFENET')
        import_module('mika.utils.LLIS')
        import_module('mika.utils')

    def test_KD_imports(self):
        import_module('mika.kd.topic_model_plus')
        import_module('mika.kd.FMEA')
        import_module('mika.kd.NER')
        import_module('mika.kd.trend_analysis')

    def test_IR_imports(self):
        import_module('mika.ir.search')

if __name__ == '__main__':
    unittest.main()