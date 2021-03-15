# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:05:45 2021

@author: srandrad
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import sys
import os
#sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
sys.path.append('../')
from topic_model_plus_class import Topic_Model_plus
smart_nlp_path = os.getcwd()
smart_nlp_path = "\\".join([smart_nlp_path.split("\\")[i] for i in range(0,len(smart_nlp_path.split("\\"))-1)])

"""A test class is defined to compare the old method and the new method. Since we
want only one method in the actual Topic_model_plus class, the test class is used for replicable results."""
class test_object():
    def __init__(self, list_of_attributes, data_df, doc_id):
        self.data_df = data_df
        self.list_of_attributes = list_of_attributes
        self.doc_ids_label = doc_id
        self.doc_ids = self.data_df[self.doc_ids_label].tolist()
        
    def OLD_remove_words_in_pct_of_docs(self, pct_=0.3):
        start = time.time()
        self.old_method_df = self.data_df.copy()
        num_docs = len(self.data_df)
        pct = np.round(pct_*num_docs)
        indicies_to_drop = []
        for attr in self.list_of_attributes:
            time.sleep(0.5)
            for i in tqdm(range(0,len(self.data_df)), attr+" removing frequent words…"):
                text = self.data_df.iloc[i][attr]
                new_text = []
                for word in text:
                    in_docs = [doc for doc in self.data_df[attr] if word in doc]
                    if len(in_docs) < pct:
                        new_text.append(word)
                #if this results in the removal of all words, then just use the original full text
                #when tested, this never actually happened but is in here just in case
                if new_text == []:
                    new_text = text
                    #print("all words are frequent", i, text)
                    indicies_to_drop.append(i)
                else:
                    self.old_method_df.at[i,attr] = new_text
        indicies_to_drop = list(set(indicies_to_drop))
        self.old_method_df = self.old_method_df.drop(indicies_to_drop).reset_index(drop=True)
        self.old_doc_ids = self.old_method_df[self.doc_ids_label].tolist()
        self.old_run_time = time.time() - start
        return
    
    def NEW_remove_words_in_pct_of_docs(self, pct_=0.3):
        start = time.time()
        self.new_method_df = self.data_df.copy()
        num_docs = len(self.data_df)
        pct = np.round(pct_*num_docs)
        indicies_to_drop = []
        time.sleep(0.5)
        for attr in tqdm(self.list_of_attributes,"Removing frequent words…"):
            all_words = list(set([word for text in self.data_df[attr] for word in text]))
            good_words = []
            for word in all_words:
                count = 0
                for text in self.data_df[attr]:
                    if word in text:
                        count+=1
                if count<pct:
                    good_words.append(word)
            i = 0
            for text in self.data_df[attr]:
                text = [word for word in text if word in good_words]
                self.new_method_df.at[i,attr] = text
                if text == []:
                    indicies_to_drop.append(i)
                i+=1
        time.sleep(0.5)
        indicies_to_drop = list(set(indicies_to_drop))
        self.new_method_df = self.new_method_df.drop(indicies_to_drop).reset_index(drop=True)
        self.new_doc_ids = self.new_method_df[self.doc_ids_label].tolist()
        self.new_run_time = time.time() - start
        return
    
    def compare(self):
        print("data frames are the same: ", self.new_method_df.equals(self.old_method_df))
        #print(self.new_method_df,self.old_method_df)
        print("doc ids are the same: ", self.new_doc_ids==self.old_doc_ids)
        print("new run time: ", self.new_run_time, " seconds")
        print("old run time: ", self.old_run_time, " seconds")
        print("difference (old-new): ", self.old_run_time-self.new_run_time, " seconds")

print("======test 1======simple test======")
"""
This test is used to show the function actually removes words appearing in a percentage of the docs.
Here we can see the words "test", "is" should be removed if we want to remove words in 30% of docs.
The old method and new method have the same output
"""
doc1 = ['this', 'is', 'a', 'test']
doc2 = ['is', 'test']
doc3 = ['test']
doc4 = ['cat']
doc5 = ['black cat']
doc6 = ['python']
doc7 = ['python', 'is', 'good']
doc8 = ['is']
doc9 = ['end']

docs = [doc1,doc2,doc3,doc4,doc5,doc6,doc7,doc8,doc9]
in_df = pd.DataFrame({"docs": docs, "ids":[i for i in range(len(docs))]})
list_of_attributes=["docs"]
ids = "ids"

test = test_object(list_of_attributes=list_of_attributes, data_df=in_df, doc_id=ids)
print(test.data_df)
test.OLD_remove_words_in_pct_of_docs()
test.NEW_remove_words_in_pct_of_docs()
print(test.new_method_df)
test.compare()

print("\n")

print("======test 2======LLIS======")
"""
This test is to show the new and old functions give the same results on larger datasets.
Also shows some of the runtime difference.
"""
list_of_attributes = ['Lesson(s) Learned','Driving Event','Recommendation(s)']
document_id_col = 'Lesson ID'
csv_filename = smart_nlp_path+"/input data/train_set_expanded_H.csv"

test_df = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col, csv_file=csv_filename)
test_df.prepare_data()
df = test_df.data_df

test = test_object(list_of_attributes=list_of_attributes, data_df=df, doc_id=document_id_col)
test.OLD_remove_words_in_pct_of_docs()
test.NEW_remove_words_in_pct_of_docs()
test.compare()

print("------comparing topic_model_plus output to test output------")
"""
This secondary test demonstrates the function in the topic_model_plus class is the same
as the one in this test function, evident by having the same output and same run time.
"""
start = time.time()
test_df.remove_words_in_pct_of_docs()
runtime = time.time() - start
test.old_method_df = test_df.data_df
test.old_run_time = runtime
test.compare()

print("\n")

print("======test 3======ICS======")
"""
This test is used to show how the fucntion performs on a large datset.
We see a big difference in runtime, with the new method immensely faster, while still giving
the same output as the old method.
"""
list_of_attributes = ["REMARKS", "SIGNIF_EVENTS_SUMMARY", "MAJOR_PROBLEMS"]
document_id_col = "INCIDENT_ID"
csv_filename = smart_nlp_path+r"\input data\209-PLUS\ics209-plus-wildfire\ics209-plus-wildfire\ics209-plus-wf_sitreps_1999to2014.csv"

test_df = Topic_Model_plus(list_of_attributes=list_of_attributes, document_id_col=document_id_col, csv_file=csv_filename, combine_cols=True)
test_df.prepare_data(dtype=str)
df = test_df.data_df[:5000].reset_index(drop=True)

test = test_object(list_of_attributes=['Combined Text'], data_df=df, doc_id=document_id_col)
test.OLD_remove_words_in_pct_of_docs()
test.NEW_remove_words_in_pct_of_docs()
test.compare()

print("------comparing topic_model_plus output to test output------")
"""
This secondary test demonstrates the function in the topic_model_plus class is the same
as the one in this test function, evident by having the same output and same run time.
"""
test_df.data_df = test_df.data_df[:5000].reset_index(drop=True)
start = time.time()
test_df.remove_words_in_pct_of_docs()
runtime = time.time() - start
test.old_method_df = test_df.data_df
test.old_run_time = runtime
test.compare()