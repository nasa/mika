# -*- coding: utf-8 -*-
"""
@author: hswalsh
"""

import os
import pandas as pd
import tomotopy as tp

# load data from file and separate into different parts of the survey
filepath = os.path.join('Survey_Topic_word intrusion.xlsx')
df = pd.read_excel(filepath)
q_idx = df.keys()
word_q_idx = q_idx[1:7]
topic_q_idx = q_idx[7:13]
meta_q_idx = q_idx[13]

# for calculating scores (average of kronecker delta for all participants)
def calc_scores(df,q_idx):
    num_answers = len(df[q_idx[0]])-1
    scores = []
    for q_id in q_idx:
        intruder = df[q_id][num_answers]
        answers = df[q_id][0:num_answers]
        score = 0
        for answer in answers:
            if answer == intruder:
                score += 1
        score = score/num_answers
        scores.append(score)
    return scores

# WORD INTRUSION
q_word_scores = calc_scores(df,word_q_idx)
print('-------------------------------------')
print('all words')
print(q_word_scores)
print('average')
print(sum(q_word_scores)/len(q_word_scores))

# TOPIC INTRUSION
q_topic_scores = calc_scores(df,topic_q_idx)
print('-------------------------------------')
print('all topics')
print(q_topic_scores)
print('average')
print(sum(q_topic_scores)/len(q_topic_scores))

# looking at probability mass
#print('-------------------------------------')
#print('probability mass')

# load saved hlda models
#de_mdl = tp.HLDAModel.load(os.path.join('results','idetc_data','Driving Eventhlda_model_object.bin'))
#ll_mdl = tp.HLDAModel.load(os.path.join('results','idetc_data','Lesson(s) Learnedhlda_model_object.bin'))
#r_mdl = tp.HLDAModel.load(os.path.join('results','idetc_data','Recommendation(s)hlda_model_object.bin'))
#
#topic_ids = [14, 84, 100, 104, 8, 17]
#print('-------------------------------------')
#print('driving event, lesson X')
#print(de_mdl.get_topic_words(topic_ids[0]))
#print('driving event, lesson X')
#print(de_mdl.get_topic_words(topic_ids[1]))
#print('-------------------------------------')
#print('lessons learned, lesson X')
#print(ll_mdl.get_topic_words(topic_ids[2]))
#print('lessons learned, lesson X')
#print(ll_mdl.get_topic_words(topic_ids[3]))
#print('-------------------------------------')
#print('recommendations, lesson X')
#print(r_mdl.get_topic_words(topic_ids[4]))
#print('recommendations, lesson X')
#print(r_mdl.get_topic_words(topic_ids[5]))

# NOTE: poor performance in MP does not appear to be correlated with probability mass distribution of words in topics

## get lesson numbers from saved result
#filepath = os.path.join('results','idetc_data','hlda_topic_dist_per_doc.csv')
#df_ln = pd.read_csv(filepath)
#ln = df_ln['lesson number'].to_list()
#
## get topic distribution for lessons from model
#def get_topic_dist(lesson_number,ln,mdl):
#    # get unseen doc by lesson number
#    all_docs = mdl.docs
#    doc_idx = ln.index(lesson_number)
#    unseen_doc = all_docs[doc_idx]
#
#    # get topic_dist for unseen_doc from mdl
#    doc_inst = mdl.make_doc(unseen_doc)
#    topic_dist, ll = mdl.infer(doc_inst)
#    return topic_dist
#
## need topic_dist for:
## - lesson = 1215,  mdl = driving event
## - lesson = 3640,  mdl = driving event
## - lesson = 10801, mdl = lessons learned
## - lesson = 4057,  mdl = lessons learned
## - lesson = 1032,  mdl = recommendations
## - lesson = 1259,  mdl = recommendations
#lesson_list = [1215, 3640, 10801, 4057, 1032, 1259]
#mdl_list = [de_mdl, de_mdl, ll_mdl, ll_mdl, r_mdl, r_mdl]
#
#topic_dist_list = []
#i = 0
#for lesson in lesson_list:
#    topic_dist_list.append(get_topic_dist(lesson,ln,mdl_list[i]))
#    i += 1
#
#print(topic_dist_list[0])
#
## calculate scores
#q_topic_scores = []
#for q_id in topic_q_idx:
#    intruder = df[q_id][num_answers]
#    answers = df[q_id][0:num_answers]
