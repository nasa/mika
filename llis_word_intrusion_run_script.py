# -*- coding: utf-8 -*-
"""
@author: hswalsh
"""

import os
from module.word_intrusion import word_intrusion_class

wi = word_intrusion_class()

filepath = os.path.join('results','LLIS_topics_Oct-01-2021','lda_results.xlsx')
save_filepath = os.path.join('results','LLIS_topics_Oct-01-2021','lda_intruded_topics.xlsx')
sheet = 'Lesson(s) Learned'
column_name = 'topic words'

shuffled_topics = wi.generate_intruded_topics(file=filepath,sheet=sheet,column_name=column_name)
wi.save_intruded_topics(filepath=save_filepath,sheet=sheet)
