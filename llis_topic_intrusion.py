# -*- coding: utf-8 -*-
"""
@author: hswalsh
"""

import os
import tomotopy as tp
import pandas as pd

lesson_numbers_filename = os.path.join('results','idetc_data','hlda_topic_dist_per_doc.csv') # only used for lesson numbers
ln_df = pd.read_csv(lesson_numbers_filename)
lesson_numbers = ln_df['lesson number'].tolist()

events_filename = os.path.join('results','idetc_data','Driving Eventhlda_model_object.bin')
events = tp.HLDAModel.load(events_filename)

lessons_filename = os.path.join('results','idetc_data','Lesson(s) Learnedhlda_model_object.bin')
lessons = tp.HLDAModel.load(lessons_filename)

recommendations_filename = os.path.join('results','idetc_data','Recommendation(s)hlda_model_object.bin')
recommendations = tp.HLDAModel.load(recommendations_filename)

def get_doc_topics(mdl,lesson_numbers,lesson_id):
    desired_doc = lesson_id
    doc_topics = []
    
    for topic_id in range(mdl.k):
        if mdl.is_live_topic(topic_id) and mdl.num_docs_of_topic(topic_id=topic_id)>0:
            topic_words = [word for word, weight in mdl.get_topic_words(topic_id=topic_id)]
            
            docs_in_topics = []
            i = 0
            for doc in mdl.docs:
                if doc.path[mdl.level(topic_id)] == topic_id:
                    docs_in_topics.append(lesson_numbers[i])
                i+=1
            
            if desired_doc in docs_in_topics:
                doc_topics.append(topic_words)
     
    return doc_topics

# docs: events - 1215, 3640; lessons - 10801, 4057; recommendations - 1032, 1212
print('1215',get_doc_topics(mdl=events,lesson_numbers=lesson_numbers,lesson_id=1215))
print('3640',get_doc_topics(mdl=events,lesson_numbers=lesson_numbers,lesson_id=3640))
print('10801',get_doc_topics(mdl=lessons,lesson_numbers=lesson_numbers,lesson_id=10801))
print('4057',get_doc_topics(mdl=lessons,lesson_numbers=lesson_numbers,lesson_id=4057))
print('1032',get_doc_topics(mdl=recommendations,lesson_numbers=lesson_numbers,lesson_id=1032))
print('1259',get_doc_topics(mdl=recommendations,lesson_numbers=lesson_numbers,lesson_id=1259))

# note: intruder topics were chosen for these documents separately from this script
