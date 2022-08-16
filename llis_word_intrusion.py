# -*- coding: utf-8 -*-
"""
@author: hswalsh
"""

import os

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
                
    def __load_topic_model(self, file, topic_column_name, doc_column_name, header=0):
        """
        Load a topic model for word intrusion analysis.
        """
        
        tm_df = pd.read_csv(file)
        topics_with_duplicates = [topic.replace(' ','') for topic in tm_df[topic_column_name].tolist()]
        topics_with_duplicates = [topic.split(',') for topic in topics_with_duplicates]
        topics_with_duplicates = topics_with_duplicates[header:]
        self.topics = []
        for topic in topics_with_duplicates:
            if topic not in self.topics:
                self.topics.append(topic) # self.topics does not have duplicates
                
        topic_docs_not_consolidated = tm_df[doc_column_name].tolist()
        topic_docs_not_consolidated = topic_docs_not_consolidated[header:]
        topic_docs_not_consolidated = [doc.replace('[','') for doc in topic_docs_not_consolidated]
        topic_docs_not_consolidated = [doc.replace(']','') for doc in topic_docs_not_consolidated]
        topic_docs_not_consolidated = [doc.replace(' ','') for doc in topic_docs_not_consolidated]
        topic_docs_not_consolidated = [doc.split(',') for doc in topic_docs_not_consolidated]
        self.docs = []
        for topic in self.topics:
            self.docs.append([])
            for i in range(0,len(topic_docs_not_consolidated)):
                if topics_with_duplicates[i] == topic:
                    for doc in topic_docs_not_consolidated[i]:
                        self.docs[-1].append(doc) # list of docs for consolidated topics

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
    
    def save_intruded_topics(self,filepath):
        """
        Saves intruded topics.
        
        PARAMETERS
        ----------
        filepath : str
            path to file to which the intruded topics will be saved
        """
        
        intruded_topics_str = []
        for topic in self.intruded_topics:
            intruded_topics_str.append(', '.join(topic))
        
        it_df = pd.DataFrame({'Topic Words':intruded_topics_str,'Intruder':self.intruders})
        it_df.to_csv(filepath)


# LESSONS LEARNED LEVEL 1: 1 SAMPLES FOR WORD
wi = word_intrusion_class()

filepath = os.path.join('examples','Taxonomy','IDETC_2021','llis_idetc_results.csv')
save_filepath = os.path.join('results','IDETC_2021_intruded_topics_lesson_1.csv')
topic_column_name = 'Lesson(s) Learned Level 1'
doc_column_name = 'Lesson IDs for row'
header = 2

shuffled_topics = wi.generate_intruded_topics(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, header=header, max_topic_size=5,num_samples=1)
wi.save_intruded_topics(filepath=save_filepath)

# LESSONS LEARNED LEVEL 2: 1 SAMPLES FOR WORD
wi = word_intrusion_class()

filepath = os.path.join('examples','Taxonomy','IDETC_2021','llis_idetc_results.csv')
save_filepath = os.path.join('results','IDETC_2021_intruded_topics_lesson_2.csv')
topic_column_name = 'Lesson(s) Learned Level 2'
doc_column_name = 'Lesson IDs for row'
header = 2

shuffled_topics = wi.generate_intruded_topics(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, header=header, max_topic_size=5,num_samples=1)
wi.save_intruded_topics(filepath=save_filepath)

# DRIVING EVENT LEVEL 1: 1 SAMPLES FOR WORD
wi = word_intrusion_class()

filepath = os.path.join('examples','Taxonomy','IDETC_2021','llis_idetc_results.csv')
save_filepath = os.path.join('results','IDETC_2021_intruded_topics_event_1.csv')
topic_column_name = 'Driving Event Level 1'
doc_column_name = 'Lesson IDs for row'
header = 2

shuffled_topics = wi.generate_intruded_topics(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, header=header, max_topic_size=5,num_samples=1)
wi.save_intruded_topics(filepath=save_filepath)

# DRIVING EVENT LEVEL 2: 1 SAMPLES FOR WORD
wi = word_intrusion_class()

filepath = os.path.join('examples','Taxonomy','IDETC_2021','llis_idetc_results.csv')
save_filepath = os.path.join('results','IDETC_2021_intruded_topics_event_2.csv')
topic_column_name = 'Driving Event Level 2'
doc_column_name = 'Lesson IDs for row'
header = 2

shuffled_topics = wi.generate_intruded_topics(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, header=header, max_topic_size=5,num_samples=1)
wi.save_intruded_topics(filepath=save_filepath)

# RECOMMENDATIONS LEVEL 1: 1 SAMPLES FOR WORD
wi = word_intrusion_class()

filepath = os.path.join('examples','Taxonomy','IDETC_2021','llis_idetc_results.csv')
save_filepath = os.path.join('results','IDETC_2021_intruded_topics_rec_1.csv')
topic_column_name = 'Recommendation(s) Level 1'
doc_column_name = 'Lesson IDs for row'
header = 2

shuffled_topics = wi.generate_intruded_topics(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, header=header, max_topic_size=5,num_samples=1)
wi.save_intruded_topics(filepath=save_filepath)

# RECOMMENDATIONS LEVEL 2: 1 SAMPLES FOR WORD
wi = word_intrusion_class()

filepath = os.path.join('examples','Taxonomy','IDETC_2021','llis_idetc_results.csv')
save_filepath = os.path.join('results','IDETC_2021_intruded_topics_rec_2.csv')
topic_column_name = 'Recommendation(s) Level 2'
doc_column_name = 'Lesson IDs for row'
header = 2

shuffled_topics = wi.generate_intruded_topics(file=filepath, topic_column_name=topic_column_name, doc_column_name=doc_column_name, header=header, max_topic_size=5,num_samples=1)
wi.save_intruded_topics(filepath=save_filepath)
