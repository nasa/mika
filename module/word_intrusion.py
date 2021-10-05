# -*- coding: utf-8 -*-
"""
@author: hswalsh
"""

import pandas as pd
import numpy as np
import random
from collections import Counter

class word_intrusion_class():
    """
    A class for automating word intrusion experimental design and analysis.
    
    Attributes
    ----------
    
    Methods
    -------
    
    """

    # private attributes

    def __init__(self):
        """
        Class Constructors
        ------------------
        
        """
        
        # public attributes
        self.topics = []
        self.selected_topics = []
        self.common_words = []
        self.shuffled_topics = []
        self.intruded_topics = []
                
    def __load_topic_model(self,file,column_name,header=0):
        """
        Load a topic model for word intrusion analysis.
        """
        
        tm_df = pd.read_csv(file)
        self.topics = [topic.split(',') for topic in tm_df[column_name].tolist()]
        self.topics = self.topics[header:]
        
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
        self.selected_topics = list(map(self.topics.__getitem__,sample_idx))
                
    def __generate_word_intruder(self,topic):
        """
        For a single topic, generate word intruder.
        """
        
        candidate_intruders = list(set(self.common_words)-set(topic))
        if not candidate_intruders:
            print('Error - topic does not contain any uncommon words; please decrease percentage of words considered common and try again.')
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
        
    def generate_intruded_topics(self,file,column_name,num_samples=20,max_topic_size=7,header=0):
        """
        Generate topics with word intruders for analysis.
        """
        
        self.__load_topic_model(file=file,column_name=column_name,header=header)
        self.__get_common_words()
        
        self.topics = [topic[:max_topic_size] for topic in self.topics]
        
        self.__select_sample_topics(num_samples=20)
        self.intruded_topics = []
        for topic in self.selected_topics:
            intruder = self.__generate_word_intruder(topic)
            shuffled_topic = self.__add_word_intruder(topic,intruder)
            self.intruded_topics.append(shuffled_topic)
        
        return self.shuffled_topics
    
    def save_intruded_topics(self,filepath):
        """
        Saves intruded topics.
        """
        
        intruded_topics_str = [', '.join(topic) for topic in self.intruded_topics]
        
        it_df = pd.DataFrame(intruded_topics_str,columns = ['Topic Words'])
        it_df.to_csv(filepath)
