# -*- coding: utf-8 -*-
"""
@author: hswalsh
"""

import os
from module.word_intrusion import word_intrusion_class

wi = word_intrusion_class()

filepath = os.path.join('results','llis_idetc_results.csv')
save_filepath = os.path.join('results','IDETC_2021_intruded_topics.csv')
column_name = 'Lesson(s) Learned Level 1'
header = 2

shuffled_topics = wi.generate_intruded_topics(file=filepath,column_name=column_name,header=header,max_topic_size=5)
wi.save_intruded_topics(filepath=save_filepath)
