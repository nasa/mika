# -*- coding: utf-8 -*-
"""
@author: hswalsh
"""

import os
from module.word_intrusion import word_intrusion_class

# LESSONS LEARNED LEVEL 1: 2 SAMPLES FOR WORD, 2 SAMPLES FOR TOPIC
wi = word_intrusion_class()

filepath = os.path.join('results','llis_idetc_results.csv')
save_filepath = os.path.join('results','IDETC_2021_intruded_topics_lesson_1.csv')
save_filepath_2 = os.path.join('results','IDETC_2021_intruded_docs_lesson_1.csv')
topic_column_name = 'Lesson(s) Learned Level 1'
doc_column_name = 'Lesson IDs for row'
header = 2

shuffled_topics = wi.generate_intruded_topics(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, header=header, max_topic_size=5,num_samples=2)
intruded_docs = wi.generate_intruded_docs(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, num_samples=2, max_num_topics=3, header=header)
wi.save_intruded_topics(filepath=save_filepath)
wi.save_intruded_docs(filepath=save_filepath_2)

# LESSONS LEARNED LEVEL 2: 1 SAMPLES FOR WORD, 1 SAMPLES FOR TOPIC
wi = word_intrusion_class()

filepath = os.path.join('results','llis_idetc_results.csv')
save_filepath = os.path.join('results','IDETC_2021_intruded_topics_lesson_2.csv')
save_filepath_2 = os.path.join('results','IDETC_2021_intruded_docs_lesson_2.csv')
topic_column_name = 'Lesson(s) Learned Level 2'
doc_column_name = 'Lesson IDs for row'
header = 2

shuffled_topics = wi.generate_intruded_topics(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, header=header, max_topic_size=5,num_samples=1)
intruded_docs = wi.generate_intruded_docs(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, num_samples=1, max_num_topics=3, header=header)
wi.save_intruded_topics(filepath=save_filepath)
wi.save_intruded_docs(filepath=save_filepath_2)

# DRIVING EVENT LEVEL 1: 1 SAMPLES FOR WORD, 1 SAMPLES FOR TOPIC
wi = word_intrusion_class()

filepath = os.path.join('results','llis_idetc_results.csv')
save_filepath = os.path.join('results','IDETC_2021_intruded_topics_event_1.csv')
save_filepath_2 = os.path.join('results','IDETC_2021_intruded_docs_event_1.csv')
topic_column_name = 'Driving Event Level 1'
doc_column_name = 'Lesson IDs for row'
header = 2

shuffled_topics = wi.generate_intruded_topics(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, header=header, max_topic_size=5,num_samples=1)
intruded_docs = wi.generate_intruded_docs(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, num_samples=1, max_num_topics=3, header=header)
wi.save_intruded_topics(filepath=save_filepath)
wi.save_intruded_docs(filepath=save_filepath_2)

# DRIVING EVENT LEVEL 2: 2 SAMPLES FOR WORD, 2 SAMPLES FOR TOPIC
wi = word_intrusion_class()

filepath = os.path.join('results','llis_idetc_results.csv')
save_filepath = os.path.join('results','IDETC_2021_intruded_topics_event_2.csv')
save_filepath_2 = os.path.join('results','IDETC_2021_intruded_docs_event_2.csv')
topic_column_name = 'Driving Event Level 2'
doc_column_name = 'Lesson IDs for row'
header = 2

shuffled_topics = wi.generate_intruded_topics(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, header=header, max_topic_size=5,num_samples=2)
intruded_docs = wi.generate_intruded_docs(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, num_samples=2, max_num_topics=3, header=header)
wi.save_intruded_topics(filepath=save_filepath)
wi.save_intruded_docs(filepath=save_filepath_2)

# RECOMMENDATIONS LEVEL 1: 2 SAMPLES FOR WORD, 2 SAMPLES FOR TOPIC
wi = word_intrusion_class()

filepath = os.path.join('results','llis_idetc_results.csv')
save_filepath = os.path.join('results','IDETC_2021_intruded_topics_rec_1.csv')
save_filepath_2 = os.path.join('results','IDETC_2021_intruded_docs_rec_1.csv')
topic_column_name = 'Recommendation(s) Level 1'
doc_column_name = 'Lesson IDs for row'
header = 2

shuffled_topics = wi.generate_intruded_topics(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, header=header, max_topic_size=5,num_samples=2)
intruded_docs = wi.generate_intruded_docs(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, num_samples=2, max_num_topics=3, header=header)
wi.save_intruded_topics(filepath=save_filepath)
wi.save_intruded_docs(filepath=save_filepath_2)

# RECOMMENDATIONS LEVEL 2: 2 SAMPLES FOR WORD, 2 SAMPLES FOR TOPIC
wi = word_intrusion_class()

filepath = os.path.join('results','llis_idetc_results.csv')
save_filepath = os.path.join('results','IDETC_2021_intruded_topics_rec_2.csv')
save_filepath_2 = os.path.join('results','IDETC_2021_intruded_docs_rec_2.csv')
topic_column_name = 'Recommendation(s) Level 2'
doc_column_name = 'Lesson IDs for row'
header = 2

shuffled_topics = wi.generate_intruded_topics(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, header=header, max_topic_size=5,num_samples=2)
intruded_docs = wi.generate_intruded_docs(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, num_samples=2, max_num_topics=3, header=header)
wi.save_intruded_topics(filepath=save_filepath)
wi.save_intruded_docs(filepath=save_filepath_2)

