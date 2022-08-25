# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 09:10:57 2020
Functions for lesson selection processing
@author: srandrad
"""

#importing packages
import pandas as pd
import re
import nltk
#nltk.download('wordnet')
#nltk.download('stopwords')
from nltk.corpus import stopwords
#nltk.download('averaged_perceptron_tagger')
#nltk.download('sentiwordnet')
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
from nltk.stem import PorterStemmer 
ps = PorterStemmer() 
lemmatizer = WordNetLemmatizer()

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

if sys.platform == "win32":
    smart_nlp_path = os.getcwd()
    smart_nlp_path = "\\".join([smart_nlp_path.split("\\")[i] for i in range(0,4)])
    smart_nlp_path = ""

lex_path = smart_nlp_path + "Lexicons\\"

def drop_topics(df):
    to_remove = ['Acquisition / procurement strategy and planning','Business processes','Center distribution of programs and management activities',
    'Communications between different offices and contractor personnel','Contractor relationships','Cross Agency coordination',
    'Education and public engagement','International partner coordination','Organizational Planning','Program level review processes',
    'Program Management','Program planning / development and management','Review systems and boards','Logistics','Procurement',
    'Small Business and Industrial Relations']
    for i in range(0,len(df)):
        #print(df['Topics'][i])
        topics = [t.strip("[]'' ") for t in str(df['Topics'][i]).split(',') if str(df['Topics'][i])!="nan"]
        topics_tmp = topics
        for j in range(0,len(to_remove)):
            if (to_remove[j] in topics):
                topics_tmp.remove(to_remove[j])
        if not topics_tmp:
            df = df.drop([i])
    return df

#text cleaning NEEDS WORK
def text_cleaning (text, lemmatize = True):
    #make lowercase
    text = text.lower()
    #split into sentences
    text = text.replace('"', "").replace("<strong>", "").replace("</strong>", ".").replace("["," ").replace("]", " ").replace("/", " ")
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        #remove punctuation, this is okay since we already have a list of sentences
        sentence = sentence.replace(r'[\W]', '').replace 
        words = nltk.word_tokenize(str(sentence))
        #removing stopwords
        filtered_words = [word for word in words if word not in stopwords.words('english')]
        sentence = filtered_words
        #lemmatization and stemming, not necessary for all types of analysis
        if lemmatize == True: 
            lem = WordNetLemmatizer()
            words = [lem.lemmatize(w) for w in sentence]
            sentence = words
    text = sentences
    #removal of numbers
    return text

#combine columns
def combine_columns(df):
    #removes unnecessary columns; maybe save lesson number for future reference
    to_drop = [ 'Submitter 1', 'Submitter 2', 'Submitter 3', 'Submitter 4',
                           'Submitter 5', 'Pont of Contact 1','Pont of Contact 2',
                           'Pont of Contact 3','Pont of Contact 4','Pont of Contact 5',
                           'Contributor 1','Contributor 2','Contributor 3','Contributor 4',
                           'Contributor 5','Organization', 'Recommendation(s)',
                           'Date Lesson Occurred','Evidence','Project / Program',
                           'The related NASA policy(s), standard(s), handbook(s), procedure(s) or other rules',	
                           'NASA Mission Directorate(s)','Sensitivity',
                           'From what phase of the program or project was this lesson learned captured?',
                          'Where (other lessons, presentations, publications, etc.)?',
                          'Publish Date','Topics']
    df = df.drop(to_drop, axis=1)
    #print("after dropped columns",df.iloc[193])
    #cleans columns
    columns = ['Title','Abstract','Lesson(s) Learned', 'Driving Event']#, 'Recommendation(s)']
    #for i in range(0, len(df)):
    #    for col in columns: #something funky about the split, combines words together
    #        df.at[i, col] = [t.strip("[]'' ") for t in df[col][i].split('.')]
    #combines remaining text into one column
    text = []
    for i in range(0, len(df)):
        df_text = str(df.iloc[i]['Title'])+" "+str(df.iloc[i]['Abstract'])+" "+str(df.iloc[i]['Driving Event'])+" "+str(df.iloc[i]['Lesson(s) Learned'])#+" "+str(df.iloc[i]['Recommendation(s)'])
        text.append(df_text)
    #df['text']= df['Title'] + df['Abstract']+df['Driving Event']+df['Lesson(s) Learned']
    df['text']=text
    #print("after adding text",df.iloc[193]['text'])
    df = df.drop(labels = columns, axis = 1)
    
    return df

#preprocessing
def preprocessing(file_name):
    #from nlp import get_data, drop_topics
    #gets data from file
    #df = get_data(file_name)
    df = pd.read_csv(smart_nlp_path+file_name)
    #drops documents with only irrelevant topics
    #df = drop_topics(df)
    #combines subject, abstract, driving event, and lessons learned
    #print("before dropped columns",df.iloc[193])
    df = combine_columns(df)
   # print(df.iloc[0]['text'],type(df.iloc[0]['text']))
    #print(df.iloc[193]['text'])
    ind = 0
    for i in range(0, len(df)):
        text = str(df.iloc[i]['text']).split(" ")
        cleaned_text = ""
        for t in text:
            t = text_cleaning(t)
            cleaned_text += (" " + str(t).strip("[]'' ").replace(".', '", ". "))
        df.at[ind,'text'] = cleaned_text
        #print(ind)
        ind+=1
    return df
    
#calculate sentiment score (+,-,=)

def penn_to_wn(tag):
    """Convert between the PennTreebank tags to simple Wordnet tags"""
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith("I") or tag.startswith("PRP") or tag.startswith("FW") or tag.startswith("C") or tag.startswith("DT") or tag.startswith("MD"):
        return wn.NOUN
    else:
        return wn.NOUN
    return None

def get_sentiment(word,tag):
    """ returns list of pos neg and objective score. But returns empty list if not present in senti wordnet. """
    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []
    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return []
    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []
    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())
    return [swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]
def aggregat_senti(senti_value):
    import numpy as np
    pos_values = []
    neg_values = []
    objective_values = []
    for senti_list in senti_value:
        if senti_list == []: senti_list = [0,0,0]
        pos_values.append(senti_list[0])
        neg_values.append(senti_list[1])
        objective_values.append(senti_list[2])
    pos = float(np.average(pos_values))
    neg = float(np.average(neg_values))
    obj = float(np.average(objective_values))
    return(pos,neg,obj)

def sentiment_score(df):
    pos = []; neg = []; obj = []
    #i = 0
    for i in range(0, len(df)):
        doc = df.iloc[i]['text']
        words_data = nltk.word_tokenize(doc)
        POS_val = nltk.pos_tag(words_data)
        senti_val = [get_sentiment(x,y) for (x,y) in POS_val]
        pos_val, neg_val, obj_val = aggregat_senti(senti_val)
        pos.append(pos_val); neg.append(neg_val); obj.append(obj_val)
    return pos, neg, obj
        
#calculate engineering (topic) relevance score
def engineering_score(df):
    import pandas as pd
    eng_lex = lemmatize_lexicon(pd.read_csv(lex_path+"_general_engineering_lexicon.csv"))
    #eng_lex = pd.read_csv("_general_engineering_lexicon.csv")
    engineering_scores = []
    terms = list(set([t for t in eng_lex['term']]))
    for doc in df['text']:
        total = 0; eng = 0
        words = nltk.word_tokenize(doc)
        POS_val = nltk.pos_tag(words)
        lem_words = [lem.lemmatize(word,penn_to_wn(tag)) for (word,tag) in POS_val]
        for w in lem_words: 
            if w in terms: eng+=1
            total +=1
        engineering_scores.append(float(float(eng)/float(total)))
    return engineering_scores
    
#calculate failure and success score
def fail_success_scores(df):
    fail_lex = lemmatize_lexicon(pd.read_csv(lex_path+"failure_lexicon.csv"))
    success_lex = lemmatize_lexicon(pd.read_csv(lex_path+"success_lexicon.csv"))
    degree_lex = pd.read_csv(lex_path+"degree_modifier_lexicon.csv")
    negation_lex = pd.read_csv(lex_path+"negation_lexicon.csv")
    #fail_lex = pd.read_csv("failure_lexicon.csv")
    fail_terms = list(set([t for t in fail_lex['term']]))
    #success_lex = pd.read_csv("success_lexicon.csv")
    success_terms = [t for t in success_lex['term']]
    #degree_lex = pd.read_csv("degree_modifier_lexicon.csv")
    degree_terms = [t for t in degree_lex['term']]
    #negation_lex = pd.read_csv("negation_lexicon.csv")
    negation_terms = [t for t in negation_lex['term']]
    fail_scores = []; success_scores = []
    for doc in df['text']:
        
        fail = 0; suc = 0; total = 0
        words = nltk.word_tokenize(doc)
        POS_val = nltk.pos_tag(words)
        words = [lem.lemmatize(word,penn_to_wn(tag)) for (word,tag) in POS_val]
        index = len(words)
        for i in range(0, index):
            success_present = False; fail_present = False
            #print(fail_lex["term"])
            if words[i] in fail_terms: 
                fail_present = True; val = 1
                additional_words = [words[max(i-1,0)], words[max(i-2,0)], words[max(i-3,0)]]
                additional_words = set(additional_words)
                for word in additional_words:
                    if word in negation_terms:
                        fail_present = False; success_present = True
                    elif word in degree_terms:
                        ind = degree_lex[degree_lex['term']== word].index.values
                        multiplier = float(degree_lex.loc[ind, 'effect'])
                        val = val*multiplier
            elif words[i] in success_terms:
                success_present = True; val = 1
                additional_words = [words[max(i-1,0)], words[max(i-2,0)], words[max(i-3,0)]]
                additional_words = set(additional_words)
                for word in additional_words:
                    if word in negation_terms:
                        fail_present = True; success_present = False
                    elif word in degree_terms:
                        ind = degree_lex[degree_lex['term']== word].index.values
                        multiplier = float(degree_lex.loc[ind, 'effect'])
                        val = val*multiplier
            if success_present == True: suc+=val
            elif fail_present == True: fail+=val 
            total+=1
        fail_scores.append(float(float(fail)/float(total))); success_scores.append(float(float(suc)/float(total)))
    return fail_scores, success_scores

#calculate design relevance score
def design_score(df):
    design_lex = lemmatize_lexicon(pd.read_csv(lex_path+"design_lexicon.csv"))
    #design_lex = pd.read_csv("design_lexicon.csv")
    design_terms = [t for t in design_lex['term']]
    design_terms = list(set(design_terms))
    design_scores = []
    for doc in df['text']:
        design = 0; total = 0
        words = nltk.word_tokenize(doc)
        POS_val = nltk.pos_tag(words)
        lem_words = [lem.lemmatize(word,penn_to_wn(tag)) for (word,tag) in POS_val]
        for word in lem_words:
            if word in design_terms: design+=1
            total+=1
        design_scores.append(float(float(design)/float(total)))
    return design_scores

#calculate managment/personnel relevance score
def management_score(df):
    management_lex = lemmatize_lexicon(pd.read_csv(lex_path+"management lexicon.csv"))
    #design_lex = pd.read_csv("design_lexicon.csv")
    #check_repeated_words(management_lex)
    manage_terms = [str(t) for t in management_lex['term']]
    manage_terms = list(set(manage_terms))
    manage_scores = []
    for doc in df['text']:
        manage = 0; total = 0
        words = nltk.word_tokenize(doc)
        POS_val = nltk.pos_tag(words)
        lem_words = [lem.lemmatize(word,penn_to_wn(tag)) for (word,tag) in POS_val]
        #words = [ps.stem(word) for word in lem_words]
        for word in lem_words:
            if word in manage_terms: manage+=1
            total+=1
        manage_scores.append(float(float(manage)/float(total)))
    return manage_scores
#calculate length of text

#lemmatize lexicons
def lemmatize_lexicon(df):
    #print(df['term'])
    lem = WordNetLemmatizer()
    #from nltk.stem import PorterStemmer 
    #ps = PorterStemmer() 
    terms = [term for term in df['term']]
    POS_val = nltk.pos_tag(terms)
    #print(POS_val)
    #wn_tag =[penn_to_wn(tag) for tag in POS_val]
    index = 0
    for (word,tag) in POS_val:
        #print(word,tag)
        wn_tag = penn_to_wn(tag)
        new_term = lem.lemmatize(word,wn_tag)
        df.at[index,'term']=new_term 
        index+=1
    """
    for term in df['term']:
        #term = lem.lemmatize(term)
        index = df[df['term']== term].index.values
        #print(term)
        if type(term)==str:
            new_term = lem.lemmatize(term)
            #print(term,new_term)
            #new_term = ps.stem(new_term)
            df.at[index,'term']=new_term 
    #print(df['term'])
            """
    return df

def length (df):
    lengths = []
    for i in range(0,len(df)):
        text_length = len(df.iloc[i]['text'])
        lengths.append(text_length)
    return lengths

#adds the scores to the data frame, creates a list of the scores for the ML model
def process_data(df):
    pos_senti_scores, neg_senti_scores, obj_senti_scores = sentiment_score(df)
    engineer_scores = engineering_score(df)
    fail_scores, success_scores = fail_success_scores(df)
    design_scores = design_score(df)
    management_scores = management_score(df)
    lengths = length(df)
    df['positive sentiment'] = pos_senti_scores
    df["negative sentiment"] = neg_senti_scores
    df["objective sentiment"] = obj_senti_scores
    df["engineering relevance"] = engineer_scores
    df["failure relevance"] = fail_scores
    df["success relevance"] = success_scores
    df["design relevance"] = design_scores
    df['management relevance'] = management_scores
    df['length'] = lengths
    data = []
    for i in range (0, len(df)):
        document_data = [pos_senti_scores[i], neg_senti_scores[i], obj_senti_scores[i],
                         engineer_scores[i], fail_scores[i], success_scores[i],
                         design_scores[i], lengths[i]]
        data.append(document_data)
    return df, data

def check_repeated_words(file):
    print(file)
    df = lemmatize_lexicon(pd.read_csv(file))
    terms_set =list(set([t for t in df['term']]))
    terms = []
    for i in range(0, len(df)):
        if (df.iloc[i]["term"] not in terms): terms.append(df.iloc[i]["term"])
        else: print("repeat word: ", df.iloc[i]["term"], i)
    if len(terms_set)==len(terms): print("no repeat terms")

#files = ["_general_engineering_lexicon.csv", "failure_lexicon.csv", "success_lexicon.csv",
        # "design_lexicon.csv","management lexicon.csv"]
#for file in files: check_repeated_words(file)
#df = preprocessing("NLP_useable_LL.csv")
#print(df.iloc[193]['text'])
#df,data = process_data(df)
#print(data[0])
#senti = sentiment_score(df)
#engineer = engineering_score(df)
#fail_suc = fail_success_scores(df)
#design = design_score(df)
#print("sentiment:",senti,"engineer:", engineer,"fail/success:", fail_suc,"design:", design)