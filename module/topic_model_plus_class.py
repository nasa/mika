# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:10:28 2021
TopicModel+ class definition
@author: srandrad, hswalsh
"""
import pandas as pd
import tomotopy as tp
import numpy as np
from time import time,sleep
from tqdm import tqdm
import os
import datetime
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from gensim.utils import simple_tokenize
from gensim.models import Phrases
import pyLDAvis
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from symspellpy import SymSpell, Verbosity
import pkg_resources
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

class Topic_Model_plus():
    """
    A class for topic modeling for aviation safety
    
    Attributes
    ----------
    data_csv : str
        defines the input data file
    doc_ids_label : str
        defines the document id column label name
    list_of_attributes : list of str
        defines various attributes within a single dataset
    extra_cols : list of str
        a list of strings defining any extra columns in the database
    folder_path : str
        defines path to folder where output files are stored
    database_name : str
        defines output file names
    combine_cols : boolean
        defines whether to combine attributes
    correction_list : list of str
        if spellchecker or segmentation is active, contains corrections made
    
    Methods
    -------
    prepare_data(self, **kwargs)
        removes incomplete rows or combines columns as defined by user
    preprocess_data(self, domain_stopwords=[], ngrams=True, ngram_range=3, threshold=15, min_count=5,quot_correction=False,spellcheck=False,segmentation=False)
        performs data preprocessing steps as defined by user
    save_preprocessed_data(self)
        saves preprocessed data to a file
    extract_preprocessed_data(self, file_name)
        uses previously saved preprocessed data
    coherence_scores(self, mdl, lda_or_hlda, measure='c_v'):
        computes and returns coherence scores
    lda(self, num_topics={}, training_iterations=1000, iteration_step=10, remove_pct=0.3, **kwargs)
        performs lda topic modeling
    save_lda_results(self):
        saves the taxonomy, coherence, and document topic distribution in one excel file
    save_lda_models(self)
        saves lda models to file
    save_lda_document_topic_distribution(self)
        saves lda document topic distribution to file
    save_lda_coherence(self)
        saves lda coherence to file
    save_lda_taxonomy(self)
        saves lda taxonomy to file
    lda_extract_models(self, file_path)
        gets lda models from file
    lda_visual(self, attr)
        saves pyLDAvis output from lda to file
    hlda(self, levels=3, training_iterations=1000, iteration_step=10, remove_pct=0.3, **kwargs)
        performs hlda topic modeling
    save_hlda_results(self):
        saves the taxonomy, level 1 taxonomy, raw topics coherence, and document topic distribution in one excel file
    save_hlda_document_topic_distribution(self)
        saves hlda document topic distribution to file
    save_hlda_models(self)
        saves hlda models to file
    save_hlda_topics(self)
        saves hlda topics to file
    save_hlda_coherence(self)
        saves hlda coherence to file
    save_hlda_taxonomy(self)
        saves hlda taxonomy to file
    save_hlda_level_n_taxonomy(self, lev=1)
        saves hlda taxonomy at level n
    hlda_extract_models(self, file_path)
        gets hlda models from file
    hlda_display(self, attr, num_words = 5, display_options={"level 1": 1, "level 2": 6}, colors='bupu')
        saves graphviz visualization of hlda tree structure
    """

#   TO DO:
#   add hyper parameter tuning for lda (alpha and beta and k/number of topics) and hlda (eta, alpha, gamma)
#   some of the attributes are ambiguously named - can we make these clearer? e.g. name, combine_cols
#   some docstring short descriptions may be improved
    
    # private attributes
    __english_vocab = set([w.lower() for w in words.words()])

    def __init__(self, document_id_col="", csv_file="", list_of_attributes=[], extra_cols = [], database_name='', combine_cols=False, create_ids=False, min_word_len=2, max_word_len=15):
        """
        CLASS CONSTRUCTORS
        ------------------
        data_csv : str
            defines the input data file
        doc_ids_label : str
            defines the document id column label name
        list_of_attributes : list of str
            defines various attributes within a single database
        extra_cols : list of str
            a list of strings defining any extra columns in the database
        folder_path : str
            defines path to folder where output files are stored
        name : str
            defines output file names
        combine_cols : boolean
            defines whether to combine attributes
        min_word_len : int
            minimum word length during tokenization
        max_word_len : int
            maxumum word length during tokenization
        """
        
        # public attributes
        self.data_csv = csv_file
        if document_id_col == "":
            self.doc_ids_label = "index"
        else:
            self.doc_ids_label = document_id_col
        self.list_of_attributes = list_of_attributes
        self.extra_cols = extra_cols
        self.folder_path = ""
        self.database_name = database_name
        self.min_word_len = min_word_len # maybe these should be defined only in the tokenization/preprocessing function - unless they are needed for other functions, do not need to be object attributes can be regular variables
        self.max_word_len = max_word_len # maybe these should be defined only in the tokenization/preprocessing function - unless they are needed for other functions, do not need to be object attributes can be regular variables
        self.correction_list = [] # maybe this should be defined only in the correction function - unless they are needed for other functions, do not need to be object attributes can be regular variables
        if combine_cols == True: 
            self.database_name = os.path.join(self.database_name+'_combined')
        self.combine_cols = combine_cols
        self.hlda_models = {}
        self.lda_models = {}
        self.create_ids = create_ids
        
    def __load_data(self, csv=True, sheet_name=None, **kwargs):
        if csv: self.data_df = pd.read_csv(open(self.data_csv,encoding='utf8',errors='ignore'), **kwargs)
        else: self.data_df = pd.read_excel(self.data_csv, sheet_name=sheet_name, **kwargs)
        if self.doc_ids_label == "index":
            self.data_df['index'] = self.data_df.index
        self.doc_ids = self.data_df[self.doc_ids_label].tolist()
     
    def __combine_columns(self):
        columns_to_drop = [cols for cols in  self.data_df.columns if cols not in [self.doc_ids_label]+self.extra_cols]
        rows_to_drop = []
        combined_text = []
        sleep(0.5)
        for i in tqdm(range(0, len(self.data_df)), "Combining Columns…"):
            text = ""
            for attr in self.list_of_attributes:
                if not(str(self.data_df.iloc[i][attr]).strip("()").lower().startswith("see") or str(self.data_df.iloc[i][attr]).strip("()").lower().startswith("same") or str(self.data_df.iloc[i][attr])=="" or isinstance(self.data_df.iloc[i][attr],float) or str(self.data_df.iloc[i][attr]).lower().startswith("none")):
                    text += str(self.data_df.iloc[i][attr])
            if text == "":
                rows_to_drop.append(i)
            combined_text.append(text)
        sleep(0.5)
        self.data_df["Combined Text"] = combined_text
        self.list_of_attributes = ["Combined Text"]
        self.data_df = self.data_df.drop(columns_to_drop, axis=1)
        self.data_df = self.data_df.drop(rows_to_drop).reset_index(drop=True)
    
    def __remove_incomplete_rows(self):
        columns_to_drop = [cols for cols in  self.data_df.columns if cols not in self.list_of_attributes+[self.doc_ids_label]+self.extra_cols]
        rows_to_drop = []
        sleep(0.5)
        for i in tqdm(range(0, len(self.data_df)), "Preparing data…"):
            for attr in self.list_of_attributes:
                if str(self.data_df.iloc[i][attr]).strip("()").lower().startswith("see") or str(self.data_df.iloc[i][attr]).strip("()").lower().startswith("same") or str(self.data_df.iloc[i][attr])=="" or isinstance(self.data_df.iloc[i][attr],float) or str(self.data_df.iloc[i][attr]).lower().startswith("none"):
                    rows_to_drop.append(i)
        sleep(0.5)
        self.data_df = self.data_df.drop(columns_to_drop, axis=1)
        self.data_df = self.data_df.drop(list(set(rows_to_drop)), axis=0)
        self.data_df = self.data_df.reset_index(drop=True)
        self.doc_ids = self.data_df[self.doc_ids_label].tolist()
    
    def __create_unique_ids(self):
        unique_ids = []
        prev_id = None
        j = 0
        for i in tqdm(range(len(self.data_df)), "Creating Unique IDs…"):
            current_id = self.data_df.iloc[i][self.doc_ids_label]
            if current_id == prev_id:
                j+=1
            else:
                j=0
            unique_ids.append(current_id+"_"+str(j))
            prev_id = current_id
        self.data_df["Unique IDs"] = unique_ids
        self.doc_ids_label = "Unique IDs"
        self.doc_ids = self.data_df["Unique IDs"].tolist()
         
    def prepare_data(self, **kwargs):
        """
        loads and prepares data by removing and combining rows/cols as defined by user
        """
        
        start_time = time()
        self.__load_data(**kwargs)
        if self.combine_cols == False: 
            self.__remove_incomplete_rows()
        if self.combine_cols == True:
            self.__combine_columns()
        if self.create_ids == True:
            self.__create_unique_ids()
        print("data preparation: ", (time()-start_time)/60,"minutes \n")
        
    def __tokenize_texts(self,texts):
        texts = [simple_tokenize(text) for text in texts if not isinstance(text,float)]
        texts = [[word for word in text if len(word)>self.min_word_len and len(word)<self.max_word_len] for text in texts if not isinstance(text,float)]
        return texts
        
    def __lowercase_texts(self,texts):
        texts = [[word.lower() for word in text] for text in texts]
        return texts
        
    def __lemmatize_texts(self,texts): #why is this so slow?
        def get_wordnet_pos(word):
            tag = pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,"N": wordnet.NOUN,"V": wordnet.VERB,"R": wordnet.ADV}
            if tag not in ['I','D','M','T','C','P']:return tag_dict.get(tag,wordnet.NOUN)
            else: return "unnecessary word"
        lemmatizer = WordNetLemmatizer()
        texts = [[lemmatizer.lemmatize(w,get_wordnet_pos(w)) for w in text if get_wordnet_pos(w)!="unnecessary word"] for text in texts if not isinstance(text,float)]
        return texts
        
    def __remove_stopwords(self,texts,domain_stopwords):
        all_stopwords = stopwords.words('english')+domain_stopwords
        all_stopwords = [word.lower() for word in all_stopwords]
        texts = [[w for w in text if not w in all_stopwords] for text in texts if not isinstance(text,float)]
        texts = [[w for w in text if len(w)>=3] for text in texts if not isinstance(text,float)]
        return texts
    
    def __quot_normalize(self,texts):
        def quot_replace(word):
            if word not in self.__english_vocab:
                w_tmp = word.replace('quot','')
                if w_tmp in self.__english_vocab:
                    word = w_tmp
            return word
        texts = [[quot_replace(word) for word in text] for text in texts if not isinstance(text,float)]
        return texts
    
    def __spellchecker(self,texts):
        def spelling_replace(word):
            if word not in self.__english_vocab and not word.isupper() and not sum(1 for c in word if c.isupper()) > 1:
                suggestions = sym_spell.lookup(word,Verbosity.CLOSEST,           max_edit_distance=2,include_unknown=True,transfer_casing=True)
                correction = suggestions[0].term
                self.correction_list.append(word+' --> '+correction)
                word = correction
            return word
        sym_spell = SymSpell()
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        texts = [[spelling_replace(word) for word in text] for text in texts if not isinstance(text,float)]
        return texts
    
    def __segment_text(self,texts):
        def segment_replace(text):
            for word in text:
                if word not in self.__english_vocab and not word.isupper():
                    segmented_word = sym_spell.word_segmentation(word).corrected_string
                    if len(segmented_word.split())>1:
                        text_str = ' '.join(text)
                        text_str = text_str.replace(word,segmented_word)
                        text = text_str.split()
                        self.correction_list.append(word+' --> '+segmented_word)
            return text
        sym_spell = SymSpell()
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        texts = [segment_replace(text) for text in texts if not isinstance(text,float)]
        return texts
        
    def __trigram_texts(self, texts, ngram_range, threshold, min_count):
        #NEEDS WORK
        # changes word order!
        ngrams = []
        ngram_models = {}
        
        for n in range(2, ngram_range+1):
            if n == 2:
                text_input = texts
            elif n > 2:
                text_input = ngram_models[str(n-1)+"gram"][texts]
            ngram_models[str(n)+"gram"]=Phrases(text_input, min_count=min_count, delimiter=b' ', threshold=threshold)
        
        for text in texts:
            ngrams_={}
            for n in range(2, ngram_range+1):
                if n == 2:
                    model = ngram_models[str(n)+"gram"][text]
                if n > 2:
                    model = ngram_models[str(n)+"gram"][ngram_models[str(n-1)+"gram"][text]]
                ngrams_[str(n)+"gram"] = [b for b in model if b.count(' ') == (n-1)]
            single_words = []
            model_ngrams = [ngram for key in ngrams_ for ngram in ngrams_[key]]
            for word in text: #only adds words that do not appear in trigrams or bigrams from the diven document
                for ngram in model_ngrams:
                    if word not in ngrams:
                        single_words.append(word)
                if model_ngrams == []:
                    single_words = text
            ngram = list(set(model_ngrams+single_words))
            #ngram = (bigrams_+trigrams_+words)
            ngrams.append(ngram)
                
        return ngrams
    
    def preprocess_data(self, domain_stopwords=[], ngrams=True, ngram_range=3, threshold=15, min_count=5,quot_correction=False,spellcheck=False,segmentation=False,drop_short_docs_thres=3, percent=0.3, drop_na=False, save_words=[], drop_dups=False):
        """
        performs data preprocessing steps as defined by user
        
        ARGUMENTS
        ---------
        domain_stopwords : list of str
            any domain specific stopwords added for preprocessing
        ngrams : boolean
            determines whether to use ngrams
        threshold : int
            threshold for ngrams
        min_count : int
            min count for ngrams
        quot_correction : boolean
            option for using quot correction feature, designed for use with LLIS
        spellcheck : boolean
            option for using spellchecker
        segmentation : boolean
            option for using segmentation spellchecker
        """
        
        if ngrams == True:
            self.ngrams = "custom"
        else: 
            self.ngrams = "tp"
        start = time()
        sleep(0.5)
                
        for attr in self.list_of_attributes:
            pbar = tqdm(total=100, desc="Preprocessing "+attr+"…")
            self.data_df[attr] = self.__tokenize_texts(self.data_df[attr])
            pbar.update(10)
            if quot_correction == True:
                self.data_df[attr] = self.__quot_normalize(self.data_df[attr])
            pbar.update(10)
            if spellcheck == True:
                self.data_df[attr] = self.__spellchecker(self.data_df[attr])
            pbar.update(10)
            if segmentation == True:
                self.data_df[attr] = self.__segment_text(self.data_df[attr])
            pbar.update(10)
            self.data_df[attr] = self.__lowercase_texts(self.data_df[attr])
            pbar.update(10)
            self.data_df[attr] = self.__lemmatize_texts(self.data_df[attr])
            pbar.update(20)
            self.data_df[attr] = self.__remove_stopwords(self.data_df[attr],domain_stopwords)
            pbar.update(20)
            if ngrams == True:
                self.data_df[attr] = self.__trigram_texts(self.data_df[attr], ngram_range,threshold,min_count)
            pbar.update(10)
            sleep(0.5)
            pbar.close()
                
        if drop_dups: self.__drop_duplicate_docs(self.list_of_attributes)
        self.__drop_short_docs(thres=drop_short_docs_thres) # need to drop short docs, else we'll get NaNs because of the next two lines; but we can set the thres so this works for the test case
        
        cols = self.data_df.columns.difference([self.doc_ids_label]+self.extra_cols)
        self.data_df[cols] = self.data_df[cols].applymap(lambda y: np.nan if (type(y)==int or len(y)==0) else y)
        if drop_na: self.data_df = self.data_df.dropna(how="any").reset_index(drop=True)
                          
        self.data_df = self.__remove_words_in_pct_of_docs(self.data_df, self.list_of_attributes, pct_=percent, save_words=save_words)

        self.doc_ids = self.data_df[self.doc_ids_label].tolist()
        print("Processing time: ", (time()-start)/60, " minutes")
        sleep(0.5)
                
    def __remove_words_in_pct_of_docs(self, data_df, list_of_attributes, pct_=0.3, save_words=[]):
            num_docs = len(data_df)
            pct = np.round(pct_*num_docs)
            indicies_to_drop = []
            sleep(0.5)
            for attr in tqdm(list_of_attributes,"Removing frequent words…"):
                all_words = list(set([word for text in data_df[attr] for word in text]))
                good_words = save_words
                for word in all_words:
                    count = 0
                    for text in data_df[attr]:
                        if word in text:
                            count+=1
                    if count<pct:
                        good_words.append(word)
                i = 0
                for text in data_df[attr]:
                    text = [word for word in text if word in good_words]
                    data_df.at[i,attr] = text
                    if text == []:
                        indicies_to_drop.append(i)
                    i+=1
            sleep(0.5)
            indicies_to_drop = list(set(indicies_to_drop))
            self.data_df = data_df.drop(indicies_to_drop).reset_index(drop=True)
            #print(self.data_df)
            #self.doc_ids = self.data_df[self.doc_ids_label].tolist()
            return self.data_df
        
    def __create_folder(self, itr=""): #itr is an optional argument to pass in a number for multiple runs on same day
        if self.folder_path == "":
            #path = os.getcwd()
            today_str = datetime.date.today().strftime("%b-%d-%Y")
            if itr != "":
                itr = "-"+str(itr)
            if os.getcwd().split("//")[-1] == 'smart_nlp':
                filename = os.path.join('results',self.database_name+'_topics_'+today_str+str(itr))
            if os.path.dirname(os.getcwd()).split("\\")[-1] == 'smart_nlp':
                filename = os.path.join(os.path.dirname(os.getcwd()),'results',self.database_name+'_topics_'+today_str+str(itr))
            else:
                filename = os.path.join('results',self.database_name+'_topics_'+today_str+str(itr))
            self.folder_path = filename#path+"/"+filename
            os.makedirs(self.folder_path, exist_ok = True)
            #print("folder created")
        else:
            return
        
    def __drop_duplicate_docs(self, cols):
        self.data_df = self.data_df.loc[self.data_df.astype(str).drop_duplicates(subset=cols, keep='last').index].reset_index(drop=True)
        
    def __drop_short_docs(self, thres=3):
        indx_to_drop = []
        for i in range(len(self.data_df)):
            for attr in self.list_of_attributes:
                if len(self.data_df.iloc[i][attr])<thres:
                    indx_to_drop.append(i)
        self.data_df = self.data_df.drop(indx_to_drop).reset_index(drop=True)
    
    def save_preprocessed_data(self):
        """
        saves preprocessed data to a file
        """
        
        self.__create_folder()
        name = "preprocessed_data.csv"
        if self.combine_cols == True:
            name = "preprocessed_data_combined_text.csv"
        self.data_df.to_csv(os.path.join(self.folder_path,name), index=False)
        #print("Preprocessed data saves to: ", self.folder_path+name)
        
    
    def extract_preprocessed_data(self, file_name, drop_short_docs=True, drop_short_docs_thres=3, drop_duplicates=True):
        """
        uses previously saved preprocessed data
        
        ARGUMENTS
        ---------
        file_name : str
            file name for extracting data
        """
        def check_for_ngrams():
            for i in range(len(self.data_df)):
                for attr in self.list_of_attributes:
                    for word in self.data_df.iloc[i][attr]:
                        words = word.split(" ")
                        if len(words)>1:
                            self.ngrams = "custom"
                            return
            self.ngrams = "tp"
            return
            
        def remove_quote_marks(word_list):
            word_list = word_list.strip("[]").split(", ")
            word_list = [w.replace("'","") for w in word_list]
            return word_list
        self.data_df = pd.read_csv(file_name)
        cols = self.list_of_attributes
        self.data_df[cols] = self.data_df[cols].applymap(lambda y: remove_quote_marks(y))
        if drop_duplicates == True:
            self.__drop_duplicate_docs(cols)
        if drop_short_docs == True:
            self.__drop_short_docs(thres=drop_short_docs_thres)
        self.doc_ids = self.data_df[self.doc_ids_label].tolist()
        check_for_ngrams()
        #print("Preprocessed data extracted from: ", file_name)
    
    def get_bert_coherence(self, coh_method='u_mass'):
        self.BERT_coherence = {}
        for attr in self.list_of_attributes:
            docs = self.data_df[attr].tolist()
            topics = self.BERT_model_topics_per_doc[attr]
            topic_model = self.BERT_models[attr]
            self.BERT_coherence[attr] = self.calc_bert_coherence(docs, topics, topic_model, method=coh_method)
            
    def calc_bert_coherence(self, docs, topics, topic_model, method = 'u_mass'):
        # Preprocess Documents
        documents = pd.DataFrame({"Document": docs,
                                  "ID": range(len(docs)),
                                  "Topic": topics})
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

        # Extract vectorizer and analyzer from BERTopic
        vectorizer = topic_model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        # Extract features for Topic Coherence evaluation
        words = vectorizer.get_feature_names()
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topic_words = [[words for words, _ in topic_model.get_topic(topic)]
                       for topic in range(len(set(topics))-1)]

        # Evaluate
        coherence_model = CoherenceModel(topics=topic_words,
                                         texts=tokens,
                                         corpus=corpus,
                                         dictionary=dictionary,
                                         coherence=method)
        coherence_per_topic = coherence_model.get_coherence_per_topic()
        return coherence_per_topic
    
    def save_bert_coherence(self, return_df=False, coh_method='u_mass'):
        self.get_bert_coherence(coh_method)
        self.__create_folder()
        max_topics = max([len(self.BERT_models[attr].topics) for attr in self.BERT_models])
        coherence_score = {"topic numbers": ["average score"]+['std dev']+[i for i in range(0,max_topics)]}
        for attr in self.list_of_attributes:
            coherence_score[attr] = []
            c_scores = self.BERT_coherence[attr]
            average_coherence = np.average(c_scores)
            coherence_score[attr].append(average_coherence)
            std_coherence = np.std(c_scores)
            coherence_score[attr].append(std_coherence)
            coherence_per_topic = c_scores
            for i in range(0, (max_topics-len(coherence_per_topic))):
                coherence_per_topic.append("n/a")
            coherence_score[attr] += coherence_per_topic
        coherence_df = pd.DataFrame(coherence_score)
        if return_df == True:
            return coherence_df
        coherence_df.to_csv(os.path.join(self.folder_path,"BERT_coherence.csv"))
        
        
    def bert_topic(self, sentence_transformer_model='all-MiniLM-L6-v2', umap=None, count_vectorizor=None, ngram_range=(1,3), BERTkwargs={}):
        self.sentence_models = {}; self.embeddings = {}; self.BERT_models = {}
        self.BERT_model_topics_per_doc = {}; self.BERT_model_probs={}
        for attr in self.list_of_attributes:
            sentence_model = SentenceTransformer(sentence_transformer_model)
            corpus = self.data_df[attr]
            embeddings = sentence_model.encode(corpus, show_progress_bar=False)
            topic_model = BERTopic(umap_model=umap, vectorizer_model=count_vectorizor, 
                                   verbose=True, n_gram_range=ngram_range, **BERTkwargs)
            topics, probs = topic_model.fit_transform(corpus, embeddings)
            self.sentence_models[attr] = sentence_model
            self.embeddings[attr] = embeddings
            self.BERT_models[attr] = topic_model
            self.BERT_model_topics_per_doc[attr] = topics
            self.BERT_model_probs[attr] = probs
            self.reduced = False
    
    def reduce_bert_topics(self, num=30):
        self.reduced = True
        for attr in self.list_of_attributes:
            corpus = self.data_df[attr]
            topic_model = self.BERT_models[attr]
            topics, probs = topic_model.reduce_topics(corpus, self.BERT_model_topics_per_doc[attr], 
                                                      self.BERT_model_probs[attr] , nr_topics=num)
            self.BERT_models[attr] = topic_model
            self.BERT_model_topics_per_doc[attr] = topics
            self.BERT_model_probs[attr] = probs
            
    def save_bert_topics(self, return_df=False, p_thres=0.001, coherence=False, coh_method='u_mass'):
        """
        saves bert topics to file
        """
        #saving raw topics with coherence
        self.__create_folder()
        dfs = {}
        for attr in self.list_of_attributes:
            mdl = self.BERT_models[attr]
            mdl_info = mdl.get_topic_info()
            doc_text = self.data_df[attr].to_list()
            topics_data = {"topic number": [],           
                           "number of documents in topic": [],
                           "topic words": [],
                           "number of words": [],
                           "best documents": [],
                           #"coherence": [],
                           "documents": []}
            if coherence == True: 
                try:
                    topics_data["coherence"] = self.BERT_coherence[attr]
                except:
                    self.get_bert_coherence(coh_method)
                    topics_data["coherence"] = self.BERT_coherence[attr]
            for k in mdl.topics:
                topics_data["topic number"].append(k)
                topics_data["number of words"].append(len(mdl.get_topic(k)))
                topics_data["topic words"].append(", ".join([word[0] for word in mdl.get_topic(k) if word[1]>p_thres]))
                topics_data["number of documents in topic"].append(mdl_info.loc[mdl_info['Topic']==k].reset_index(drop=True).at[0,'Count'])
                if k!=-1: 
                    best_docs = [self.doc_ids[doc_text.index(text)] for text in mdl.get_representative_docs(k)]
                else:
                    best_docs = "n/a"
                topics_data["best documents"].append(best_docs)
                docs = [self.doc_ids[i] for i in range(len(self.BERT_model_topics_per_doc[attr])) if self.BERT_model_topics_per_doc[attr][i] == k]
                topics_data["documents"].append(docs)
            df = pd.DataFrame(topics_data)
            dfs[attr] = df
            if return_df == False:
                if self.reduced:
                    file = os.path.join(self.folder_path,attr+"_reduced_BERT_topics.csv")
                else: 
                    file = os.path.join(self.folder_path,attr+"_BERT_topics.csv")
                df.to_csv(file)
        if return_df == True:
            return dfs
    
    def save_bert_taxonomy(self, return_df=False, p_thres=0.0001):
        self.__create_folder()
        taxonomy_data = {attr:[] for attr in self.list_of_attributes}
        for attr in self.list_of_attributes:
            mdl = self.BERT_models[attr]
            for doc in self.BERT_model_topics_per_doc[attr]: 
                topic_num = doc
                words =  ", ".join([word[0] for word in mdl.get_topic(topic_num) if word[1]>p_thres])
                taxonomy_data[attr].append(words)
        taxonomy_df = pd.DataFrame(taxonomy_data)
        taxonomy_df = taxonomy_df.drop_duplicates()
        lesson_nums_per_row = []
        num_lessons_per_row = []
        for i in range(len(taxonomy_df)):
            lesson_nums = []
            tax_row  = "\n".join([taxonomy_df.iloc[i][key] for key in taxonomy_data])
            for j in range(len(self.doc_ids)):
                doc_row = "\n".join([taxonomy_data[key][j] for key in taxonomy_data])
                if doc_row == tax_row:                      
                    lesson_nums.append(self.doc_ids[j])
            lesson_nums_per_row.append(lesson_nums)
            num_lessons_per_row.append(len(lesson_nums))
        taxonomy_df["document IDs for row"] = lesson_nums_per_row
        taxonomy_df["number of documents for row"] = num_lessons_per_row
        taxonomy_df = taxonomy_df.sort_values(by=[key for key in taxonomy_data])
        taxonomy_df = taxonomy_df.reset_index(drop=True)
        self.bert_taxonomy_df = taxonomy_df
        if return_df == True:
            return taxonomy_df
        if self.reduced:
            file = os.path.join(self.folder_path,attr+"_reduced_bert_taxonomy.csv")
        else: 
            file = os.path.join(self.folder_path,'bert_taxonomy.csv')
        taxonomy_df.to_csv(file)
    
    def save_bert_document_topic_distribution(self, return_df=False):
        self.__create_folder()
        doc_data = {attr: [] for attr in self.list_of_attributes}
        doc_data['document number'] = self.doc_ids
        for attr in self.list_of_attributes:
            doc_data[attr] = self.BERT_model_topics_per_doc[attr]
        doc_df = pd.DataFrame({key:pd.Series(value) for key, value in doc_data.items()})
        if return_df == True:
            return doc_df
        if self.reduced:
            file = os.path.join(self.folder_path,attr+"_reduced_bert_topic_dist_per_doc.csv")
        else: 
            file = os.path.join(self.folder_path,"bert_topic_dist_per_doc.csv")
        doc_df.to_csv(file)
        
    def save_bert_results(self, coherence=False, coh_method='u_mass'):
        """
        saves the taxonomy, coherence, and document topic distribution in one excel file
        """
        
        self.__create_folder()
        data = {}
        topics_dict = self.save_bert_topics(return_df=True, coherence=coherence, coh_method=coh_method)
        data.update(topics_dict)
        data["taxonomy"] = self.save_bert_taxonomy(return_df=True)
        if coherence == True: data["coherence"] = self.save_bert_coherence(return_df=True, coh_method=coh_method)
        data["document topic distribution"] = self.save_bert_document_topic_distribution(return_df=True)
        if self.reduced:
            file = os.path.join(self.folder_path,"Reduced_BERTopic_results.xlsx")
        else: 
            file = os.path.join(self.folder_path,'BERTopic_results.xlsx')
        with pd.ExcelWriter(file) as writer2:
            for results in data:
                if len(results) >31:
                    result_label = results[:31]
                else:
                    result_label = results
                data[results].to_excel(writer2, sheet_name = result_label, index = False)
    
    def save_bert_vis(self):
        self.__create_folder()
        if self.reduced:
            file = os.path.join(self.folder_path, 'Reduced')
        else: 
            file = os.path.join(self.folder_path,"")
        for attr in self.list_of_attributes:
            topic_model = self.BERT_models[attr]
            fig = topic_model.visualize_topics()
            fig.write_html(file+'bertopics_viz.html')
            hfig = topic_model.visualize_hierarchy()
            hfig.write_html(file+'bertopics_hierarchy_viz.html')
    
    def coherence_scores(self, mdl, lda_or_hlda, measure='c_v'):
        """
        computes and returns coherence scores
        
        ARGUMENTS
        ---------
        mdl : lda or hlda model object
            topic model object created previously
        lda_or_hlda : str
            denotes whether coherence is being calculated for lda or hlda
        measure : str
            denotes which coherence metric to compute
            
        RETURNS
        -------
        scores : dict
            coherence scores, averages, and std dev
        """
        
        scores = {}
        coh = tp.coherence.Coherence(mdl, coherence= measure)
        if lda_or_hlda == "hlda":
            scores["per topic"] = [coh.get_score(topic_id=k) for k in range(mdl.k) if (mdl.is_live_topic(k) and mdl.num_docs_of_topic(k)>0)]
            for level in range(1, self.levels):
                level_scores = []
                for k in range(mdl.k):
                    if int(mdl.level(k)) == level:
                        level_scores.append(coh.get_score(topic_id=k))
                scores["level "+str(level)+" average"] = np.average(level_scores)
                scores["level "+str(level)+" std dev"] = np.std(level_scores)
        elif lda_or_hlda == "lda":
            scores["per topic"] = [coh.get_score(topic_id=k) for k in range(mdl.k)]
        scores["average"] = np.average(scores["per topic"])
        scores['std dev'] = np.std(scores["per topic"])
        return scores
    
    def __create_corpus_of_ngrams(self, texts):
        corpus = tp.utils.Corpus()
        for text in texts:
            corpus.add_doc(text)
        #identifies n_grams
        cands = corpus.extract_ngrams(min_cf=1, min_df=1, max_len=3)
        #transforms corpus to contain n_grams
        corpus.concat_ngrams(cands, delimiter=' ')
        return corpus
    
    def __find_optimized_lda_topic_num(self, attr, max_topics, training_iterations=1000, iteration_step=10, thres = 0.005, **kwargs):
        coherence = []
        LL = []
        perplexity = []
        topic_num = [i for i in range(1, max_topics+1)]
        ##need to address this specifically what percentage is removed
        texts = self.data_df[attr].tolist()
        sleep(0.5)
        for num in tqdm(topic_num, attr+" LDA optimization…"):
            if self.ngrams == "tp":
                corpus = self.__create_corpus_of_ngrams(texts)
                lda = tp.LDAModel(k=num, tw = tp.TermWeight.IDF, corpus=corpus, **kwargs)
            else:
                lda = tp.LDAModel(k=num, tw = tp.TermWeight.IDF, **kwargs)
                for text in texts:
                    lda.add_doc(text)
            sleep(0.5)
            for i in range(0, training_iterations, iteration_step):
                lda.train(iteration_step)
            coherence.append(self.coherence_scores(lda, "lda")["average"])
            LL.append(lda.ll_per_word)
            perplexity.append(lda.perplexity)
        #print(coherence, perplexity)
        coherence = normalize(np.array([coherence,np.zeros(len(coherence))]))[0]
        perplexity = normalize(np.array([perplexity,np.zeros(len(perplexity))]))[0]
        #plots optomization graph
        plt.figure()
        plt.xlabel("Number of Topics")
        plt.ylabel("Normalized Score")
        plt.title("LDA optimization for "+attr)
        plt.plot(topic_num, coherence, label="Coherence (c_v)", color="purple")
        plt.plot(topic_num, perplexity, label="Perplexity", color="green")
        plt.legend()
        plt.show()
        self.__create_folder()
        plt.savefig(os.path.join(self.folder_path,"LDA_optimization_"+attr+"_.png"))
        plt.close()
#        plt.figure()
#        plt.xlabel("Number of Topics")
#        plt.ylabel("Perplexity")
#        plt.title("LDA optimization for "+attr)
#        plt.plot(topic_num, perplexity, marker='o', color="green")
#        plt.show()
#        self.__create_folder()
#        plt.savefig(self.folder_path+"/LDA_optimization_P_"+attr+"_.png")
#
#        plt.close()
#        plt.figure()
#        plt.xlabel("Number of Topics")
#        plt.ylabel("Loglikelihood")
#        plt.title("LDA optimization for "+attr)
#        plt.plot(topic_num, LL, marker='o', color="blue")
#        plt.show()
#        self.__create_folder()
#        plt.savefig(self.folder_path+"/LDA_optimization_LL_"+attr+"_.png")
        #want to minimize perplexity, maximize coherence, look for max difference between the two
        #diff = [coherence[i]-perplexity[i] for i in range(len(topic_num))]
        #change_in_diff = [abs(diff[i]-diff[i+1])-abs(diff[i+1]-diff[i+2]) for i in range(0, len(diff)-2)]
        
        best_index = 0
        diffs = [abs(coherence[i]-coherence[i-1]) for i in range(1, len(coherence))]
        for diff in diffs:
            if diff<thres:
                best_index = diffs.index(diff)
                break
        best_num_of_topics = topic_num[best_index]
        self.lda_num_topics[attr] = best_num_of_topics
        
    def __lda_optimization(self, max_topics=200,training_iterations=1000, iteration_step=10, thres = 0.005, **kwargs):
        #needs work
        start = time()
        self.lda_num_topics = {}
        for attr in self.list_of_attributes:
            self.__find_optimized_lda_topic_num(attr, max_topics, training_iterations=1000, iteration_step=10, thres = 0.005, **kwargs)
            #print(self.lda_num_topics[attr], " topics for ", attr)
        print("LDA topic optomization: ", (time()-start)/60, " minutes")
    
    def lda(self, num_topics={}, training_iterations=1000, iteration_step=10, max_topics=0, **kwargs):
        # TO DO: the function of the num_topics var is not easy to understand - nd to make clearer and revise corresponding argument description in docstring
        """
        performs lda topic modeling
        
        ARGUMENTS
        ---------
        num_topics : dict
            keys are values in list_of_attributes, values are the number of topics lda forms
            optional - if omitted, lda optimization is run and produces the num_topics
        training_iterations : int
            number of training iterations
        iteration_step : int
            iteration step size for training
        **kwargs:
            any key-word arguments that can be passed into the tp lda model (i.e. hyperparaters alpha, beta, eta)
        """
        
        start = time()
        self.lda_models = {}
        self.lda_coherence = {}
        if num_topics == {}:
            if max_topics == 0:
                max_topics = 200
            self.__lda_optimization(max_topics=max_topics, **kwargs)
        else:
            self.lda_num_topics = num_topics
        for attr in self.list_of_attributes:
            texts = self.data_df[attr].tolist()
            if self.ngrams == "tp":
                corpus = self.__create_corpus_of_ngrams(texts)
                lda = tp.LDAModel(k=self.lda_num_topics[attr], tw = tp.TermWeight.IDF, corpus=corpus, **kwargs)
            else:
                lda = tp.LDAModel(k=self.lda_num_topics[attr], tw = tp.TermWeight.IDF, **kwargs)
                for text in texts:
                    lda.add_doc(text)
            sleep(0.5)
            for i in tqdm(range(0, training_iterations, iteration_step), attr+" LDA…"):
                lda.train(iteration_step)
            self.lda_models[attr] = lda
            self.lda_coherence[attr] = self.coherence_scores(lda, "lda")
        print("LDA: ", (time()-start)/60, " minutes")
        
    def save_lda_models(self):
        """
        saves lda models to file
        """
        self.__create_folder()
        for attr in self.list_of_attributes:
            mdl = self.lda_models[attr]
            mdl.save(os.path.join(self.folder_path,attr+"_lda_model_object.bin"))
        self.save_preprocessed_data()
    
    def save_lda_document_topic_distribution(self, return_df=False):
        """
        saves lda document topic distribution to file or returns the dataframe to another function
        """
        
        #identical to hlda function except for lda tag
        self.__create_folder()
        doc_data = {attr: [] for attr in self.list_of_attributes}
        doc_data['document number'] = self.doc_ids
        for attr in self.list_of_attributes:
            mdl = self.lda_models[attr]
            for doc in mdl.docs:
                doc_data[attr].append(doc.get_topic_dist())
        doc_df = pd.DataFrame(doc_data)
        if return_df == True:
            return doc_df
        doc_df.to_csv(os.path.join(self.folder_path,"lda_topic_dist_per_doc.csv"))
        #print("LDA topic distribution per document saved to: ",self.folder_path+"/lda_topic_dist_per_doc.csv")
    
    def save_lda_coherence(self, return_df=False):
        """
        saves lda coherence to file or returns the dataframe to another function
        """
        
        self.__create_folder()
        max_topics = max([value for value in self.lda_num_topics.values()])
        coherence_score = {"topic numbers": ["average score"]+['std dev']+[i for i in range(0,max_topics)]}
        for attr in self.list_of_attributes:
            coherence_score[attr] = []
            c_scores = self.lda_coherence[attr]
            average_coherence = c_scores['average']
            coherence_score[attr].append(average_coherence)
            std_coherence = c_scores['std dev']
            coherence_score[attr].append(std_coherence)
            coherence_per_topic = c_scores['per topic']
            for i in range(0, (max_topics-len(coherence_per_topic))):
                coherence_per_topic.append("n/a")
            coherence_score[attr] += coherence_per_topic
        coherence_df = pd.DataFrame(coherence_score)
        if return_df == True:
            return coherence_df
        coherence_df.to_csv(os.path.join(self.folder_path,"lda_coherence.csv"))
    
    def save_lda_topics(self, return_df=False, p_thres=0.001):
        """
        saves lda topics to file
        """
        
        #saving raw topics with coherence
        self.__create_folder()
        dfs = {}
        for attr in self.list_of_attributes:
            mdl = self.lda_models[attr]
            topics_data = {"topic number": [],           
                           "number of documents in topic": [],
                           "topic words": [],
                           "total number of words": [],
                           "number of words": [],
                           "best document": [],
                           "coherence": [],
                           "documents": []}
            topics_data["coherence"] = self.lda_coherence[attr]["per topic"]
            for k in range(mdl.k):
                topics_data["topic number"].append(k)
                topics_data["total number of words"].append(mdl.get_count_by_topics()[k])
                probs = mdl.get_topic_word_dist(k)
                probs = [p for p in probs if p>p_thres]
                topics_data["number of words"].append(len(probs))
                topics_data["topic words"].append(", ".join([word[0] for word in mdl.get_topic_words(k, top_n=len(probs))]))
            docs_in_topic ={k:[] for k in range(mdl.k)}
            probs = {k:[] for k in range(mdl.k)}
            i = 0
            for doc in mdl.docs:
                for topic, weight in doc.get_topics(top_n=5):
                    docs_in_topic[topic].append(self.doc_ids[i])
                    probs[topic].append(weight)
                i+=1
            #print(probs)
            for k in docs_in_topic:
                topics_data["best document"].append(docs_in_topic[k][probs[k].index(max(probs[k]))])
                topics_data["number of documents in topic"].append(len(docs_in_topic[k]))
                topics_data["documents"].append(docs_in_topic[k])
            df = pd.DataFrame(topics_data)
            dfs[attr] = df
            if return_df == False:
                df.to_csv(os.path.join(self.folder_path,attr+"_lda_topics.csv"))
                #print("LDA topics for "+attr+" saved to: ",self.folder_path+"/"+attr+"_lda_topics.csv")
        if return_df == True:
            return dfs
    
    def save_lda_taxonomy(self, return_df=False, use_labels=False, num_words=10):
        """
        saves lda taxonomy to file or returns the dataframe to another function
        """
        
        self.__create_folder()
        taxonomy_data = {attr:[] for attr in self.list_of_attributes}
        for attr in self.list_of_attributes:
            mdl = self.lda_models[attr]
            for doc in mdl.docs: 
                topic_num = int(doc.get_topics(top_n=1)[0][0])
                if use_labels == False:
                    num_words = min(mdl.get_count_by_topics()[topic_num], num_words)
                    words =  ", ".join([word[0] for word in mdl.get_topic_words(topic_num, top_n=num_words)])
                else:
                    words = ", ".join(self.lda_labels[attr][topic_num])
                #if len(words) > 35000:
                #    words = words[0:words.rfind(", ")]
                taxonomy_data[attr].append(words)
        taxonomy_df = pd.DataFrame(taxonomy_data)
        taxonomy_df = taxonomy_df.drop_duplicates()
        lesson_nums_per_row = []
        num_lessons_per_row = []
        for i in range(len(taxonomy_df)):
            lesson_nums = []
            tax_row  = "\n".join([taxonomy_df.iloc[i][key] for key in taxonomy_data])
            for j in range(len(self.doc_ids)):
                doc_row = "\n".join([taxonomy_data[key][j] for key in taxonomy_data])
                if doc_row == tax_row:                      
                    lesson_nums.append(self.doc_ids[j])
            lesson_nums_per_row.append(lesson_nums)
            num_lessons_per_row.append(len(lesson_nums))
        taxonomy_df["document IDs for row"] = lesson_nums_per_row
        taxonomy_df["number of documents for row"] = num_lessons_per_row
        taxonomy_df = taxonomy_df.sort_values(by=[key for key in taxonomy_data])
        taxonomy_df = taxonomy_df.reset_index(drop=True)
        self.lda_taxonomy_df = taxonomy_df
        if return_df == True:
            return taxonomy_df
        taxonomy_df.to_csv(os.path.join(self.folder_path,'lda_taxonomy.csv'))
        #print("LDA taxonomy saved to: ", os.path.join(self.folder_path,'lda_taxonomy.csv'))
    
    def save_lda_results(self):
        """
        saves the taxonomy, coherence, and document topic distribution in one excel file
        """
        
        self.__create_folder()
        data = {}
        topics_dict = self.save_lda_topics(return_df=True)
        data.update(topics_dict)
        data["taxonomy"] = self.save_lda_taxonomy(return_df=True)
        data["coherence"] = self.save_lda_coherence(return_df=True)
        data["document topic distribution"] = self.save_lda_document_topic_distribution(return_df=True)
        with pd.ExcelWriter(os.path.join(self.folder_path,'lda_results.xlsx')) as writer2:
            for results in data:
                data[results].to_excel(writer2, sheet_name = results, index = False)
        #print("LDA results saved to: ", os.path.join(self.folder_path,'lda_results.xlsx'))
        
    def lda_extract_models(self, file_path):
        """
        gets lda models from file
        
        ARGUMENTS
        ---------
        file_path : str
            path to file
        """
        self.lda_num_topics = {}
        self.lda_coherence = {}
        self.lda_models = {}
        for attr in self.list_of_attributes:
            self.lda_models[attr] = tp.LDAModel.load(os.path.join(file_path,attr+"_lda_model_object.bin"))
            self.lda_coherence[attr] = self.coherence_scores(self.lda_models[attr], "lda")
            self.lda_num_topics[attr] = self.lda_models[attr].k
        #print("LDA models extracted from: ", file_path)
        preprocessed_filepath = os.path.join(file_path,"preprocessed_data")
        #if self.list_of_attributes == ['Combined Text']:
        #    self.combine_cols = True
        #    preprocessed_filepath += "_combined_text"
        self.extract_preprocessed_data(preprocessed_filepath+".csv")
        self.folder_path = file_path
        
    def lda_visual(self, attr):
        """
        saves pyLDAvis output from lda to file
        
        ARGUMENTS
        ---------
        attr : str
            reference to attribute of interest
        """
        
        self.__create_folder()
        mdl = self.lda_models[attr]
        topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
        doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
        doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
        vocab = list(mdl.used_vocabs)
        term_frequency = mdl.used_vocab_freq
        prepared_data = pyLDAvis.prepare(
            topic_term_dists, 
            doc_topic_dists, 
            doc_lengths, 
            vocab, 
            term_frequency
        )
        pyLDAvis.save_html(prepared_data, os.path.join(self.folder_path,attr+'_ldavis.html'))
        #print("LDA Visualization for "+attr+" saved to: "+self.folder_path+'/'+attr+'_ldavis.html')
    
    def hlda_visual(self, attr):
        """
        saves pyLDAvis output from hlda to file
        
        ARGUMENTS
        ---------
        attr : str
            reference to attribute of interest
        """
        self.__create_folder()
        mdl = self.hlda_models[attr]
        topics = [k for k in range(mdl.k) if mdl.is_live_topic(k)]
        
        topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k) if mdl.is_live_topic(k)])
        doc_topic_dists_pre_stack = []
        for doc in mdl.docs:
            doc_topics = []
            for k in topics:
                if k in doc.path:
                    doc_topics.append(list(doc.path).index(k))
                else:
                    doc_topics.append(0)
            doc_topic_dists_pre_stack.append(doc_topics)
        doc_topic_dists = np.stack(doc_topic_dists_pre_stack)
        doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
        vocab = list(mdl.used_vocabs)
        term_frequency = mdl.used_vocab_freq
        prepared_data = pyLDAvis.prepare(
            topic_term_dists, 
            doc_topic_dists, 
            doc_lengths, 
            vocab, 
            term_frequency
        )
        pyLDAvis.save_html(prepared_data, os.path.join(self.folder_path,attr+'_hldavis.html'))
        #print("hLDA Visualization for "+attr+" saved to: "+self.folder_path+'/'+attr+'_hldavis.html')
    
    def label_lda_topics(self, extractor_min_cf=5, extractor_min_df=3, extractor_max_len=5, extractor_max_cand=5000, labeler_min_df=5, labeler_smoothing=1e-2, labeler_mu=0.25, label_top_n=3):
        """
        Uses tomotopy's auto topic labeling tool to label topics. Stores labels in class; after running this function, a flag can be used to use labels or not in taxonomy saving functions.
        
        ARGUMENTS
        ---------
        extractor_min_cf : int
            from tomotopy docs: "minimum collection frequency of collocations. Collocations with a smaller collection frequency than min_cf are excluded from the candidates. Set this value large if the corpus is big"
        extractor_min_df : int
            from tomotopy docs: "minimum document frequency of collocations. Collocations with a smaller document frequency than min_df are excluded from the candidates. Set this value large if the corpus is big"
        extractor_max_len : int
            from tomotopy docs: "maximum length of collocations"
        extractor_max_cand : int
            from tomotopy docs: "maximum number of candidates to extract"
        labeler_min_df : int
            from tomotopy docs: "minimum document frequency of collocations. Collocations with a smaller document frequency than min_df are excluded from the candidates. Set this value large if the corpus is big"
        labeler_smoothing : float
            from tomotopy docs: "a small value greater than 0 for Laplace smoothing"
        labeler_mu : float
            from tomotopy docs: "a discriminative coefficient. Candidates with high score on a specific topic and with low score on other topics get the higher final score when this value is the larger."
        label_top_n : int
            from tomotopy docs: "the number of labels"
        """
        
        extractor = tp.label.PMIExtractor(min_cf=extractor_min_cf, min_df=extractor_min_df, max_len=extractor_max_len, max_cand=extractor_max_cand)
        
        self.lda_labels = {}
        for attr in self.list_of_attributes:
            cands = extractor.extract(self.lda_models[attr])
            labeler = tp.label.FoRelevance(self.lda_models[attr], cands, min_df=labeler_min_df, smoothing=labeler_smoothing, mu=labeler_mu)
            self.lda_labels[attr] = []
            for k in range(self.lda_models[attr].k):
                label_w_probs = labeler.get_topic_labels(k,top_n=label_top_n)
                label = [word for word,prob in label_w_probs]
                self.lda_labels[attr].append(label)
        
    def label_hlda_topics(self, extractor_min_cf=5, extractor_min_df=3, extractor_max_len=5, extractor_max_cand=5000, labeler_min_df=5, labeler_smoothing=1e-2, labeler_mu=0.25, label_top_n=3):
        """
        Uses tomotopy's auto topic labeling tool to label topics. Stores labels in class; after running this function, a flag can be used to use labels or not in taxonomy saving functions.
        
        ARGUMENTS
        ---------
        extractor_min_cf : int
            from tomotopy docs: "minimum collection frequency of collocations. Collocations with a smaller collection frequency than min_cf are excluded from the candidates. Set this value large if the corpus is big"
        extractor_min_df : int
            from tomotopy docs: "minimum document frequency of collocations. Collocations with a smaller document frequency than min_df are excluded from the candidates. Set this value large if the corpus is big"
        extractor_max_len : int
            from tomotopy docs: "maximum length of collocations"
        extractor_max_cand : int
            from tomotopy docs: "maximum number of candidates to extract"
        labeler_min_df : int
            from tomotopy docs: "minimum document frequency of collocations. Collocations with a smaller document frequency than min_df are excluded from the candidates. Set this value large if the corpus is big"
        labeler_smoothing : float
            from tomotopy docs: "a small value greater than 0 for Laplace smoothing"
        labeler_mu : float
            from tomotopy docs: "a discriminative coefficient. Candidates with high score on a specific topic and with low score on other topics get the higher final score when this value is the larger."
        label_top_n : int
            from tomotopy docs: "the number of labels"
        """
        
        extractor = tp.label.PMIExtractor(min_cf=extractor_min_cf, min_df=extractor_min_df, max_len=extractor_max_len, max_cand=extractor_max_cand)
        
        self.hlda_labels = {}
        for attr in self.list_of_attributes:
            cands = extractor.extract(self.hlda_models[attr])
            labeler = tp.label.FoRelevance(self.lda_models[attr], cands, min_df=labeler_min_df, smoothing=labeler_smoothing, mu=labeler_mu)
            self.hlda_labels[attr] = []
            for k in range(self.hlda_models[attr].k):
                label_w_probs = labeler.get_topic_labels(k,top_n=label_top_n)
                label = [word for word,prob in label_w_probs]
                self.hlda_labels[attr].append(label)
    
    def save_mixed_taxonomy(self,use_labels=False):
        """
        A custom mixed lda/hlda model taxonomy. Must run lda and hlda with desired parameters first.
        
        ARGUMENTS
        ---------
        
        """
        
        attr_for_lda = [self.list_of_attributes[i] for i in [0,2]]
        attr_for_hlda = self.list_of_attributes[1]
        
        self.__create_folder()
        taxonomy_data = {'Lesson(s) Learned':[],'Driving Event Level 1':[],'Driving Event Level 2':[],'Recommendation(s)':[]}

        # first attribute; lda
        attr = self.list_of_attributes[0]
        mdl = self.lda_models[attr]
        for doc in mdl.docs:
            topic_num = int(doc.get_topics(top_n=1)[0][0])
            if use_labels == False:
                num_words = min(mdl.get_count_by_topics()[topic_num], 100)
                words =  ", ".join([word[0] for word in mdl.get_topic_words(topic_num, top_n=num_words)])
            else:
                words = ", ".join(self.lda_labels[attr][topic_num])
            taxonomy_data[attr].append(words)
        
        # second attribute; hlda
        attr = self.list_of_attributes[1]
        mdl = self.hlda_models[attr]
        for doc in mdl.docs:
            topic_nums = doc.path
            for level in range(1, self.levels):
                if use_labels == False:
                    words =  ", ".join([word[0] for word in mdl.get_topic_words(topic_nums[level], top_n=500)])
                else:
                    words = ", ".join(self.hlda_labels[attr][topic_nums[level]])
                taxonomy_data[attr+" Level "+str(level)].append(words)
                
        # third attribute; lda
        attr = self.list_of_attributes[2]
        mdl = self.lda_models[attr]
        for doc in mdl.docs:
            topic_num = int(doc.get_topics(top_n=1)[0][0])
            if use_labels == False:
                num_words = min(mdl.get_count_by_topics()[topic_num], 100)
                words =  ", ".join([word[0] for word in mdl.get_topic_words(topic_num, top_n=num_words)])
            else:
                words = ", ".join(self.lda_labels[attr][topic_num])
            taxonomy_data[attr].append(words)
        
        self.taxonomy_data = taxonomy_data
        taxonomy_df = pd.DataFrame(taxonomy_data)
        taxonomy_df = taxonomy_df.drop_duplicates()
        lesson_nums_per_row = []
        num_lessons_per_row = []
        for i in range(len(taxonomy_df)):
            lesson_nums = []
            tax_row  = "\n".join([taxonomy_df.iloc[i][key] for key in taxonomy_data])
            for j in range(len(self.doc_ids)):
                doc_row = "\n".join([taxonomy_data[key][j] for key in taxonomy_data])
                if doc_row == tax_row:
                    lesson_nums.append(self.doc_ids[j])
            lesson_nums_per_row.append(lesson_nums)
            num_lessons_per_row.append(len(lesson_nums))
        taxonomy_df["document IDs for row"] = lesson_nums_per_row
        taxonomy_df["number of documents for row"] = num_lessons_per_row
        taxonomy_df = taxonomy_df.sort_values(by=[key for key in taxonomy_data])
        taxonomy_df = taxonomy_df.reset_index(drop=True)
        self.taxonomy_df = taxonomy_df
        taxonomy_df.to_csv(os.path.join(self.folder_path,'mixed_taxonomy.csv'))
    
    def hlda(self, levels=3, training_iterations=1000, iteration_step=10, **kwargs):
        """
        performs hlda topic modeling
        
        ARGUMENTS
        ---------
        levels : int
            number of hierarchical levels
        training_iterations : int
            number of training iterations
        iteration_step : int
            iteration step size for training
        **kwargs:
            any key-word arguments that can be passed into the tp lda model (i.e. hyperparaters alpha, gamma, eta)
        """
        
        start = time()
        self.hlda_models = {}
        self.hlda_coherence = {}
        self.levels = levels
        for attr in self.list_of_attributes:
            texts = self.data_df[attr].tolist()
            if self.ngrams == "tp":
                corpus = self.__create_corpus_of_ngrams(texts)
                mdl = tp.HLDAModel(depth=levels, tw = tp.TermWeight.IDF, corpus=corpus, **kwargs)
            else: 
                mdl = tp.HLDAModel(depth=levels, tw = tp.TermWeight.IDF, **kwargs)
                for text in texts:
                    mdl.add_doc(text)
            sleep(0.5)
            for i in tqdm(range(0, training_iterations, iteration_step), attr+" hLDA…"):
                mdl.train(iteration_step)
                self.hlda_models[attr]=mdl
                sleep(0.5)
            self.hlda_coherence[attr] = self.coherence_scores(mdl, "hlda")
            sleep(0.5)
        print("hLDA: ", (time()-start)/60, " minutes")
        return
    
    def save_hlda_document_topic_distribution(self, return_df=False):
        """
        saves hlda document topic distribution to file
        """
        
        self.__create_folder()
        doc_data = {attr: [] for attr in self.list_of_attributes}
        doc_data['document number']=self.doc_ids
        for attr in self.list_of_attributes:
            mdl = self.hlda_models[attr]
            for doc in mdl.docs:
                doc_data[attr].append(doc.get_topic_dist())
        doc_df = pd.DataFrame(doc_data)
        if return_df == True:
            return doc_df
        doc_df.to_csv(os.path.join(self.folder_path,'hlda_topic_dist_per_doc.csv'))
        #print("hLDA topic distribution per document saved to: ",self.folder_path+"hlda_topic_dist_per_doc.csv")
    
    def save_hlda_models(self):
        """
        saves hlda models to file
        """
        self.__create_folder()
        for attr in self.list_of_attributes:
            mdl = self.hlda_models[attr]
            mdl.save(os.path.join(self.folder_path,attr+"_hlda_model_object.bin"))
            #print("hLDA model for "+attr+" saved to: ", (self.folder_path+"/"+attr+"_hlda_model_object.bin"))
        self.save_preprocessed_data()
        
    def save_hlda_topics(self, return_df=False, p_thres=0.001):
        """
        saves hlda topics to file
        """
        #saving raw topics with coherence
        self.__create_folder()
        dfs = {}
        for attr in self.list_of_attributes:
            mdl = self.hlda_models[attr]
            topics_data = {"topic level": [],
                "topic number": [],
                "parent": [],
                "number of documents in topic": [],
                "topic words": [],
                "total number of words": [],
                "number of words": [],
                "best document": [],
                "coherence": [],
                "documents": []}
            topics_data["coherence"] = self.hlda_coherence[attr]["per topic"]
            for k in range(mdl.k):
                if not mdl.is_live_topic(k) or mdl.num_docs_of_topic(k)<0:
                    continue
                topics_data["parent"].append(mdl.parent_topic(k))
                topics_data["topic level"].append(mdl.level(k))
                topics_data["number of documents in topic"].append(mdl.num_docs_of_topic(k))
                topics_data["topic number"].append(k)
                probs = mdl.get_topic_word_dist(k)
                probs = [p for p in probs if p>p_thres]
                topics_data["number of words"].append(len(probs))
                topics_data["total number of words"].append(mdl.get_count_by_topics()[k])
                topics_data["topic words"].append(", ".join([word[0] for word in mdl.get_topic_words(k, top_n=len(probs))]))
                i = 0
                docs_in_topic = []
                probs = []
                for doc in mdl.docs:
                    if doc.path[mdl.level(k)] == k:
                        prob = doc.get_topic_dist()[mdl.level(k)]
                        docs_in_topic.append(self.doc_ids[i])
                        probs.append(prob)
                    i += 1
                topics_data["best document"].append(docs_in_topic[probs.index(max(probs))])
                #print(k, docs_in_topic)
                topics_data["documents"].append(docs_in_topic)
            df = pd.DataFrame(topics_data)
            dfs[attr] = df
            if return_df == False:
                df.to_csv(os.path.join(self.folder_path,attr+"_hlda_topics.csv"))
                #print("hLDA topics for "+attr+" saved to: ",self.folder_path+"/"+attr+"_hlda_topics.csv")
        if return_df == True:
            return dfs
            
    def save_hlda_coherence(self, return_df=False):
        """
        saves hlda coherence to file
        """
        self.__create_folder()
        coherence_data = {}
        for attr in self.list_of_attributes:
            coherence_data[attr+" average"]=[]; coherence_data[attr+" std dev"]=[]
            for level in range(self.levels):
                if level == 0:
                    coherence_data[attr+" average"].append(self.hlda_coherence[attr]["average"])
                    coherence_data[attr+" std dev"].append(self.hlda_coherence[attr]["std dev"])
                else:
                    coherence_data[attr+" std dev"].append(self.hlda_coherence[attr]["level "+str(level)+" std dev"])
                    coherence_data[attr+" average"].append(self.hlda_coherence[attr]["level "+str(level)+" average"])
        index = ["total"]+["level "+str(i) for i in range(1, self.levels)]
        coherence_df = pd.DataFrame(coherence_data, index=index)
        if return_df == True:
            return coherence_df
        coherence_df.to_csv(os.path.join(self.folder_path,"hlda_coherence.csv"))
        #print("hLDA coherence scores saved to: ",self.folder_path+"/"+"hlda_coherence.csv")
    
    def save_hlda_taxonomy(self, return_df=False, use_labels=False, num_words=10):
        """
        saves hlda taxonomy to file
        """
        
        self.__create_folder()
        taxonomy_data = {attr+" Level "+str(level):[] for attr in self.list_of_attributes for level in range(1,self.levels)}
        for attr in self.list_of_attributes:
            mdl = self.hlda_models[attr]
            for doc in mdl.docs: 
                topic_nums = doc.path
                for level in range(1, self.levels):
                    if use_labels == False:
                        words = ", ".join([word[0] for word in mdl.get_topic_words(topic_nums[level], top_n=num_words)])
                    else:
                        words = ", ".join(self.hlda_labels[attr][topic_nums[level]])
                    taxonomy_data[attr+" Level "+str(level)].append(words)
        self.taxonomy_data = taxonomy_data
        taxonomy_df = pd.DataFrame(taxonomy_data)
        taxonomy_df = taxonomy_df.drop_duplicates()
        lesson_nums_per_row = []
        num_lessons_per_row = []
        for i in range(len(taxonomy_df)):
            lesson_nums = []
            tax_row  = "\n".join([taxonomy_df.iloc[i][key] for key in taxonomy_data])
            for j in range(len(self.doc_ids)):
                doc_row = "\n".join([taxonomy_data[key][j] for key in taxonomy_data])
                if doc_row == tax_row:                      
                    lesson_nums.append(self.doc_ids[j])
            lesson_nums_per_row.append(lesson_nums)
            num_lessons_per_row.append(len(lesson_nums))
        taxonomy_df["document IDs for row"] = lesson_nums_per_row
        taxonomy_df["number of documents for row"] = num_lessons_per_row
        taxonomy_df = taxonomy_df.sort_values(by=[key for key in taxonomy_data])
        taxonomy_df = taxonomy_df.reset_index(drop=True)
        self.taxonomy_df = taxonomy_df
        if return_df == True:
            return taxonomy_df
        taxonomy_df.to_csv(os.path.join(self.folder_path,'hlda_taxonomy.csv'))
        #print("hLDA taxonomy saved to: ", self.folder_path+"/hlda_taxonomy.csv")
    
    def save_hlda_level_n_taxonomy(self, lev=1, return_df=False):
        """
        saves hlda taxonomy at level n
        
        ARGUMENTS
        ---------
        lev : int
            level number to save
        """
        
        self.__create_folder()
        try:
            pd.read_csv(os.path.join(self.folder_path,'hlda_taxonomy.csv'))
        except:
            self.save_hlda_taxonomy(return_df = True)
        taxonomy_level_data = {attr+" Level "+str(lev): self.taxonomy_data[attr+" Level "+str(lev)] for attr in self.list_of_attributes}
        taxonomy_level_df = pd.DataFrame(taxonomy_level_data)
        taxonomy_level_df = taxonomy_level_df.drop_duplicates()
        lesson_nums_per_row = []
        num_lessons_per_row = []
        for i in range(len(taxonomy_level_df)):
            lesson_nums = []
            tax_row = "\n".join([taxonomy_level_df.iloc[i][key] for key in taxonomy_level_data])
            for j in range(len(self.doc_ids)):
                doc_row = "\n".join([taxonomy_level_data[key][j] for key in taxonomy_level_data])
                if doc_row == tax_row:                      
                    lesson_nums.append(self.doc_ids[j])
            lesson_nums_per_row.append(lesson_nums)
            num_lessons_per_row.append(len(lesson_nums))
        taxonomy_level_df["document IDs for row"] = lesson_nums_per_row
        taxonomy_level_df["number of documents for row"] = num_lessons_per_row
        taxonomy_level_df = taxonomy_level_df.sort_values(by=[key for key in taxonomy_level_data])
        taxonomy_level_df = taxonomy_level_df.reset_index(drop=True)
        if return_df == True:
            return taxonomy_level_df
        taxonomy_level_df.to_csv(os.path.join(self.folder_path,"hlda_level"+str(lev)+"_taxonomy.csv"))
        #print("hLDA level "+str(lev)+" taxonomy saved to: ", self.folder_path+"/hlda_level"+str(lev)+"_taxonomy.csv")
    
    def save_hlda_results(self):
        """
        saves the taxonomy, level 1 taxonomy, raw topics coherence, and document topic distribution in one excel file
        """
        
        self.__create_folder()
        data = {}
        data["taxonomy"] = self.save_hlda_taxonomy(return_df=True)
        data["level 1 taxonomy"] = self.save_hlda_level_n_taxonomy(lev=1, return_df=True)
        topics_dict = self.save_hlda_topics(return_df=True)
        data.update(topics_dict)
        data["coherence"] = self.save_hlda_coherence(return_df=True)
        data["document topic distribution"] = self.save_hlda_document_topic_distribution(return_df=True)
        with pd.ExcelWriter(os.path.join(self.folder_path,"hlda_results.xlsx")) as writer2:
            for results in data:
                data[results].to_excel(writer2, sheet_name = results, index = False)
        #print("hLDA results saved to: ", self.folder_path+"/hlda_results.xlsx")
    
    def hlda_extract_models(self, file_path):
        """
        gets hlda models from file
        
        ARGUMENTS
        ---------
        file_path : str
            path to file
        """
        
        #TO DO: add extract preprocessed data, use existing folder
        self.hlda_models = {}
        self.hlda_coherence = {}
        for attr in self.list_of_attributes:
            self.hlda_models[attr]=tp.HLDAModel.load(os.path.join(file_path,attr+"_hlda_model_object.bin"))
            self.levels = self.hlda_models[attr].depth
            self.hlda_coherence[attr] = self.coherence_scores(self.hlda_models[attr], "hlda")
        #print("hLDA models extracted from: ", file_path)
        preprocessed_filepath = os.path.join(file_path,"preprocessed_data")
        if self.list_of_attributes == ['Combined Text']:
            self.combine_cols = True
            preprocessed_filepath += "_combined_text"
        self.extract_preprocessed_data(preprocessed_filepath+".csv")
        self.folder_path = file_path
        
    def hlda_display(self, attr, num_words = 5, display_options={"level 1": 1, "level 2": 6}, colors='bupu', filename=''):
        # TO DO: levels/level/lev are used inconsistently as params throughout this class
        """
        saves graphviz visualization of hlda tree structure
        
        ARGUMENTS
        ---------
        attr : str
            attribute of interest
        num_words : int
            number of words per node
        display_options : dict, nested
            keys are levels, values are max nodes
            {"level 1": n} n is the max number over level 1 nodes
        colors: str
            brewer colorscheme used, default is blue-purple
            see http://graphviz.org/doc/info/colors.html#brewer for options
        filename: str
            can input a filename for where the topics are stored in order to make display 
            after hlda; must be an ouput from "save_hlda_topics()" or hlda.bin object
        
        """
        
        try:
            from graphviz import Digraph
        except ImportError as error:
            # Output expected ImportErrors.
            print(error.__class__.__name__ + ": " + error.message)
            print("GraphViz not installed. Please see:\n https://pypi.org/project/graphviz/ \n https://www.graphviz.org/download/")
            return
        if filename != '':
            #handles saved topic inputs, bin inputs
            paths = filename.split("\\")
            self.folder_path = "\\".join([paths[i] for i in range(len(paths)-1)])
            if self.hlda_models == {}:
                self.hlda_extract_models(self.folder_path+"\\")
        try:
            df = pd.read_csv(os.path.join(self.folder_path,attr+"_hlda_topics.csv"))
        except:
            try:
                df = pd.read_excel(os.path.join(self.folder_path,"hlda_results.xlsx"),sheet_name=attr)
            except:
                self.save_hlda_topics()
                df = pd.read_csv(os.path.join(self.folder_path,attr+"_hlda_topics.csv"))
        dot = Digraph(comment="hLDA topic network")
        color_scheme = '/'+colors+str(max(3,len(display_options)+1))+"/"
        nodes = {key:[] for key in display_options}
        for i in range(len(df)):
            if int(df.iloc[i]["topic level"]) == 0 and int(df.iloc[i]["number of documents in topic"]) > 0:
                root_words = df.iloc[i]["topic words"].split(", ")
                root_words = "\\n".join([root_words[i] for i in range(0,min(num_words,int(df.iloc[i]["number of words"])))])
                dot.node(str(df.iloc[i]["topic number"]), root_words, style="filled", fillcolor=color_scheme+str(1))
            elif int(df.iloc[i]["number of documents in topic"])>0 and str(df.iloc[i]["topic level"]) != '0':
                if (len(nodes["level "+str(df.iloc[i]["topic level"])]) <= display_options["level "+str(df.iloc[i]["topic level"])]) and not isinstance(df.iloc[i]["topic words"],float):
                    words = df.iloc[i]["topic words"].split(", ")
                    words = "\\n".join([words[i] for i in range(0,min(num_words,int(df.iloc[i]["number of words"])))])
                    topic_id = df.iloc[i]["topic number"]
                    parent_id = df.iloc[i]["parent"]
                    level = df.iloc[i]['topic level']
                    if int(level)>1 and parent_id not in nodes["level "+str(level-1)]: 
                        continue
                    else:
                        dot.node(str(topic_id), words, style="filled", fillcolor=color_scheme+str(level+1))
                        dot.edge(str(parent_id),str(topic_id))
                        nodes["level "+str(level)].append(topic_id)

        dot.attr(layout='twopi')
        dot.attr(overlap="voronoi")
        dot.render(filename = os.path.join(self.folder_path,attr+"_hlda_network"), format = 'png')
