# -*- coding: utf-8 -*-
"""
@author: hswalsh
"""

import pandas as pd
import numpy as np
from collections import Counter

import random
random.seed(0)

class word_intrusion_class():
    """
    A class for automating word intrusion experimental design and analysis.
    
    Excerpt from the original paper: "Reading Tea Leaves: How Humans Interpret Topic Models" by Chang et al
    
    In the word intrusion task, the subject is presented with a randomly ordered set of six words.
    The task of the user is to find the word which is out of place or does not belong with the others,
    i.e., the intruder. When the set of words minus the intruder makes sense together, then the subject
    should easily identify the intruder.
    
    In order to construct a set to present to the subject, we first select at random a topic from the model.
    We then select the five most probable words from that topic. In addition to these words, an intruder
    word is selected at random from a pool of words with low probability in the current topic (to reduce
    the possibility that the intruder comes from the same semantic group) but high probability in some
    other topic (to ensure that the intruder is not rejected outright due solely to rarity). All six words are
    then shuffled and presented to the subject.
    
    Subjects were instructed to focus on the meanings of words, not their syntactic usage or orthography. We
    also presented subjects with the option of viewing the “correct” answer after they submitted their own
    response, to make the tasks more engaging. Here the “correct” answer was determined by the model which
    generated the data, presented as if it were the response of another user. At the same time, subjects were
    encouraged to base their responses on their own opinions, not to try to match other subjects’ (the models’)
    selections. In small experiments, we have found that this extra information did not bias subjects’ responses.
    
    Attributes
    ----------
    topics : list of lists of strings
        all topics loaded from file
    common_words : list of strings
        words commonly occurring in topics
    
    Methods
    -------
    generate_intruded_topics(file,topic_column_name,num_samples=20,max_topic_size=7,header=0)
        create intruded topics from base topics loaded from file
    
    save_intruded_topics(filepath)
        save intruded topics to file
    """

    # private attributes

    def __init__(self):
        """
        Class Constructors
        ------------------
        
        """
        
        # public attributes
        self.topics = []
        self.docs = []
        self.common_words = []
        self.intruders = []
        self.intruder_topics_idx = []
                
    def __load_topic_model(self, file, topic_column_name, doc_column_name, header=0):
        """
        Load a topic model for word intrusion analysis.
        """
        
        tm_df = pd.read_csv(file)
        tm_df = tm_df.loc[tm_df.astype(str).drop_duplicates(subset=topic_column_name, keep='last').index].reset_index(drop=True)
        self.topics = [topic.replace(' ','') for topic in tm_df[topic_column_name].tolist()]
        self.topics = [topic.split(',') for topic in self.topics]
        self.topics = self.topics[header:]
        self.docs = tm_df[doc_column_name].tolist()
        self.docs = self.docs[header:]
        self.docs = [doc.replace('[','') for doc in self.docs]
        self.docs = [doc.replace(']','') for doc in self.docs]
        self.docs = [doc.replace(' ','') for doc in self.docs]
        self.docs = [doc.split(',') for doc in self.docs]
        self.all_docs = list(set([doc for doc_in_topic in self.docs for doc in doc_in_topic]))
        
    def __get_common_words(self,perc_common=.1):
        """
        Get common words in entire topic model, for use in generating word intruders.
        """
        
        all_words = [word for topic in self.topics for word in topic]
        counts = Counter(all_words)
        num_common_words = int(np.round(len(all_words)*perc_common))
        self.common_words = [count[0] for count in counts.most_common(num_common_words)]
                
    def __select_sample_topics(self,num_samples=20):
        """
        Select sample of topics in topic model for experiment.
        """
        
        num_topics = len(self.topics)
        sample_idx = random.sample(range(num_topics),num_samples)
        selected_topics = list(map(self.topics.__getitem__,sample_idx))
        
        return selected_topics
        
    def __select_sample_docs(self,num_samples=5):
        """
        Select sample of docs in topic model for experiment.
        """
        
        num_docs = len(self.all_docs)
        sample_idx = random.sample(range(num_docs),num_samples)
        selected_docs = list(map(self.all_docs.__getitem__,sample_idx))
                
        return selected_docs, sample_idx
        
    def __get_topics_of_docs(self,top=3):
        """
        Get topics associated with docs.
        """
        
        # this doesn't grab the "top" topics though - need topic distributions to do this
        topics_of_docs = [] # this is by doc index
        for j in range(0,len(self.all_docs)): # self.all_docs: list of all docs; j is doc index
            topics_of_docs.append([])
            for i in range(0,len(self.docs)): # self.docs: list of docs per topic; i is topic index
                if self.all_docs[j] in self.docs[i]: # if current doc is in the current sublist of docs per topic
                    if len(topics_of_docs[j]) < top: # only take top n topics
                        topics_of_docs[j].append(i) # save index of topic to topics_of_docs
                    
        return topics_of_docs
                
    def __generate_word_intruder(self,topic):
        """
        For a single topic, generate word intruder.
        """
        
        candidate_intruders = list(set(self.common_words)-set(topic))
        if not candidate_intruders:
            print('Error - topic contains all uncommon words; please decrease percentage of words considered common and try again.')
            intruder = []
        else:
            intruder = candidate_intruders[random.randint(0,len(candidate_intruders)-1)]
        
        return intruder
        
    def __add_word_intruder(self,topic,intruder):
        """
        Add word intruder into single topic and shuffle.
        """
        
        topic.append(intruder)
        random.shuffle(topic)
    
        return topic
        
    def generate_intruded_topics(self, file, topic_column_name, doc_column_name, num_samples=20, max_topic_size=5, header=0):
        """
        Generate topics with word intruders for analysis.
        
        PARAMETERS
        ----------
        file : str
            file from which to load topics
        topic_column_name : str
            name of column in file from which to load topics
        doc_column_name : str
            name of column in file from which to load doc numbers
        num_samples : int
            number of intruded topics to produce
        max_topic_size : int
            limits number of words to present in a topic; 5 is the number recommended by Chang et al
        header : int
            allows for multiple header lines in file from which to load topics
            
        RETURNS
        -------
        intruded topics : list of lists of words
            intruded topics for experiment
        """
        
        self.__load_topic_model(file=file, topic_column_name=topic_column_name, doc_column_name=doc_column_name, header=header)
        self.__get_common_words()
        
        selected_topics = self.__select_sample_topics(num_samples=num_samples)
        selected_topics = [topic[:max_topic_size] for topic in selected_topics]
        
        intruded_topics = []
        for topic in selected_topics:
            intruder = self.__generate_word_intruder(topic)
            self.intruders.append(intruder)
            shuffled_topic = self.__add_word_intruder(topic,intruder)
            intruded_topics.append(shuffled_topic)
        
        self.intruded_topics = intruded_topics
        
        return intruded_topics
    
    def generate_intruded_docs(self, file, topic_column_name, doc_column_name, num_samples=5, max_num_topics=3, header=0):
        """
        Generate docs with topic intruders for analysis.
        
        PARAMETERS
        ----------
        file : str
            file from which to load topics
        topic_column_name : str
            name of column in file from which to load topics
        doc_column_name : str
            name of column in file from which to load doc numbers
        num_samples : int
            number of intruded docs to produce
        max_num_topics : int
            limits number of topics to present for a doc
        header : int
            allows for multiple header lines in file from which to load topics
            
        RETURNS
        -------
        intruded docs : list of lists of words
            intruded docs for experiment
        """
        
        self.__load_topic_model(file=file, topic_column_name=topic_column_name, doc_column_name=doc_column_name, header=header)
        
        self.selected_docs, selected_docs_idx = self.__select_sample_docs(num_samples=num_samples)
        doc_topics = self.__get_topics_of_docs(top=max_num_topics)
        selected_doc_topics = list(map(doc_topics.__getitem__,selected_docs_idx))
        
        all_topic_idx = range(0,len(self.topics))
        
        intruded_docs = []
        for doc_topic in selected_doc_topics:
            intruded_docs.append(doc_topic)
            sample_idx_options = set(all_topic_idx) - set(doc_topic)
            sample_topic_idx = random.sample(sample_idx_options,1)[0]
            intruded_docs[-1].append(sample_topic_idx)
            random.shuffle(intruded_docs[-1])
            self.intruder_topics_idx.append(sample_topic_idx)
            
        self.intruded_docs = intruded_docs
            
        return intruded_docs
        
    
    def save_intruded_topics(self,filepath):
        """
        Saves intruded topics.
        
        PARAMETERS
        ----------
        filepath : str
            path to file to which the intruded topics will be saved
        """
                
        it_df = pd.DataFrame({'Topic Words':self.intruded_topics,'Intruder':self.intruders})
        it_df.to_csv(filepath)
    
    def save_intruded_docs(self,filepath):
        """
        Saves intruded docs.
        
        PARAMETERS
        ----------
        filepath : str
            path to file to which the intruded docs will be saved
        """
        
        doc_intruded_topics = []
        for intruded_doc in self.intruded_docs:
            doc_intruded_topics.append([])
            for topic in intruded_doc:
                doc_intruded_topics[-1].append(self.topics[topic])
        
        intruder_topics = []
        for intruder_topic_idx in self.intruder_topics_idx:
            intruder_topics.append(self.topics[intruder_topic_idx])
        
        it_df = pd.DataFrame({'Lesson Number':self.selected_docs,'Topics':doc_intruded_topics,'Intruder':intruder_topics})
        it_df.to_csv(filepath)
