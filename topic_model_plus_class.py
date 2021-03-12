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
from gensim.utils import simple_preprocess
from gensim.models import Phrases
import pyLDAvis
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from symspellpy import SymSpell, Verbosity
import pkg_resources

"""
- Note: pass in doman stopwords when we call the processing function, pass in optional parameter

- to perform preprocessing:
    -prepare data: print run time
    -preprocess data: include progress bar bc large data sets; print run time; include all parameters as optional so they are easy to change:
        -ngrams: min count, threshold, n-gram range; use of tp ngrams or custom ngrams
- to perform topic modeling:
    -hlda or lda->calls % removal function: include progress bar bc large data sets, based on training;print run time:
        -params: % to be removed, min_cf, seed, level, eta, gamma, alpha, TW, training iterations and steps,
- saving options:
    - save preprocessed data
    - save tm object .bin
    - save topics for each attr
    - save topic coherence
    - save taxonomy of varying levels
    - print run time
 - display options:
     -hlda tree display-graphviz required
     -ldaviz for lda objects
     -print run time
 - to run: 
     -identify file name, columns, and document id column
     -create object
     -run prepare data
     -run process data
     -run topic modeling
     -run save
     -run display
     ~10 lines of code for a simple dataset
  - future improvements:
      -when extracting preprocessed data or bins, save results to existing folder rather than new one
      -add hyper parameter tuning for lda (alpha and beta) and hlda (eta, alpha, gamma)"""

class Topic_Model_plus():
    def __init__(self, document_id_col="", csv_file="", list_of_attributes=[], extra_cols = [], name="output data/", combine_cols=False,quot_correction=False,spellcheck=False,segmentation=False):
        self.data_csv = csv_file
        self.doc_ids_label = document_id_col
        self.list_of_attributes = list_of_attributes
        self.extra_cols = extra_cols
        self.folder_path = ""
        self.name = name
        if combine_cols == True: 
            self.name += "_combined"
        self.combine_cols = combine_cols
        self.quot_correction = quot_correction
        self.spellcheck = spellcheck
        self.segmentation = segmentation
        self.correction_list = []
        
    def load_data(self, **kwargs):
            self.data_df = pd.read_csv(open(self.data_csv,encoding='utf8',errors='ignore'), **kwargs)
            self.doc_ids = self.data_df[self.doc_ids_label].tolist()
     
    def combine_columns(self):
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
    
    def remove_incomplete_rows(self):
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
         
    def prepare_data(self, **kwargs):
        start_time = time()
        self.load_data(**kwargs)
        if self.combine_cols == False: 
            self.remove_incomplete_rows()
        if self.combine_cols == True:
            self.combine_columns()
        print("data preparation: ", (time()-start_time)/60,"minutess \n")
        
    def preprocess_data(self, domain_stopwords=[], ngrams=True, ngram_range=3, threshold=15, min_count=5):
        english_vocab = set([w.lower() for w in words.words()])
        if ngrams == True:
            self.ngrams = "custom"
        else: 
            self.ngrams = "tp"
        def clean_texts(texts):
            def clean_text(text):
                if not isinstance(text,float):
                    text = simple_preprocess(text)
                return text
            texts = [clean_text(text) for text in texts]
            return texts
        def lemmatize_texts(texts):
            def get_wordnet_pos(word):
                tag = pos_tag([word])[0][1][0].upper()
                tag_dict = {"J": wordnet.ADJ,"N": wordnet.NOUN,"V": wordnet.VERB,"R": wordnet.ADV}
                if tag not in ['I','D','M','T','C','P']:return tag_dict.get(tag,wordnet.NOUN)
                else: return "unnecessary word"
            def lemmatize_text(text):
                if not isinstance(text,float):
                    lemmatizer = WordNetLemmatizer()
                    text = [lemmatizer.lemmatize(w,get_wordnet_pos(w)) for w in text if get_wordnet_pos(w)!="unnecessary word"]
                return text
            lemmatized_texts = [lemmatize_text(text) for text in texts]
            return lemmatized_texts
        def remove_stopwords(texts,domain_stopwords):
            def rm_stopwords(text):
                all_stopwords = stopwords.words('english')+domain_stopwords
                all_stopwords = [word.lower() for word in all_stopwords]
                if not isinstance(text,float):
                    text = [w for w in text if not w in all_stopwords]
                    text = [w for w in text if len(w)>3]
                return text
            texts = [rm_stopwords(text) for text in texts]
            return texts
        def canonize_texts(self,texts):
            sym_spell = SymSpell()
            dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
            sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
            def quot_normalize(text):
                for word in text:
                    if word not in english_vocab:
                        w_tmp = word.replace('quot','')
                        if w_tmp in english_vocab:
                            text = list(map(lambda x: x if x != word else w_tmp,text))
                return text
            def spellchecker(text): # TODO: a list of words not to correct; way of detecting acronyms that should not be corrected
                for word in text:
                    if word not in english_vocab:
                        suggestions = sym_spell.lookup(word,Verbosity.CLOSEST,           max_edit_distance=2,include_unknown=True)
                        correction = suggestions[0].term
                        if correction != word:
                            text = list(map(lambda x: x if x != word else correction,text))
                            self.correction_list.append(word+' --> '+correction)
                return text
            def segment_text(text):
                for word in text:
                    if word not in english_vocab:
                        segmented_word = sym_spell.word_segmentation(word).corrected_string
                        if segmented_word.split()[0] != word and len(segmented_word.split())>1:
                            text_str = ' '.join(text)
                            text_str = text_str.replace(word,segmented_word)
                            text = text_str.split()
                            self.correction_list.append(word+' --> '+segmented_word)
                return text
            for text in texts:
                if not isinstance(text,float):
                    if self.quot_correction == True:
                        text = quot_normalize(text)
                    if self.spellcheck == True:
                        text = spellchecker(text)
                    if self.segmentation == True:
                        text = segment_text(text)
            return texts
        def trigram_texts(texts, ngram_range, threshold, min_count):
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
        def preprocess(texts,domain_stopwords=[], ngrams=True, ngram_range=3, threshold=15, min_count=5,LLIS=0,spellcheck=1):
            texts = clean_texts(texts)
            texts = canonize_texts(self,texts)
            texts = lemmatize_texts(texts)
            texts = remove_stopwords(texts,domain_stopwords)
            if ngrams == True:
                texts = trigram_texts(texts, ngram_range,threshold, min_count)
            return texts
        start = time()
        texts = {}
        sleep(0.5)
        for attr in tqdm(self.list_of_attributes,desc="Preprocessing data…"):
            texts[attr] = preprocess(self.data_df[attr], domain_stopwords, ngrams, ngram_range, threshold, min_count)
            self.data_df[attr]=texts[attr]
        cols = self.data_df.columns.difference([self.doc_ids_label]+self.extra_cols)
        self.data_df[cols] = self.data_df[cols].applymap(lambda y: np.nan if (type(y)==int or len(y)==0) else y)#.dropna(how="any")
        self.data_df = self.data_df.dropna(how="any").reset_index(drop=True)
        self.doc_ids = self.data_df[self.doc_ids_label].tolist()
        print("Processing time: ", (time()-start)/60, " minutes")
        sleep(0.5)
        
    def create_folder(self, itr=""): #itr is an optional argument to pass in a number for multiple runs on same day
        if self.folder_path == "":
            path = os.getcwd()
            today_str = datetime.date.today().strftime("%b-%d-%Y")
            if itr != "":
                itr = "-"+str(itr)
            filename = self.name+'topics-'+today_str+str(itr)
            self.folder_path = path+"/"+filename
            os.makedirs(self.folder_path, exist_ok = True)
            print("folder created")
        else:
            return
        
    def save_preprocessed_data(self):
        self.create_folder()
        name = "/preprocessed_data.csv"
        if self.combine_columns == True:
            name = "/preprocessed_data_combined_text.csv"
        self.data_df.to_csv(self.folder_path+name, index=False)
        print("Preprocessed data saves to: ", self.folder_path+name)
    
    def extract_preprocessed_data(self, file_name):
        def remove_quote_marks(word_list):
            word_list = word_list.strip("[]").split(", ")
            word_list = [w.replace("'","") for w in word_list]
            return word_list
        self.data_df = pd.read_csv(file_name)
        cols = self.list_of_attributes
        self.data_df[cols] = self.data_df[cols].applymap(lambda y: remove_quote_marks(y))
        self.doc_ids = self.data_df[self.doc_ids_label].tolist()
        print("Preprocessed data extracted from: ", file_name)
        
    def coherence_scores(self, mdl, lda_or_hlda, measure='c_v'):
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
    
    def remove_words_in_pct_of_docs(self, pct_=0.3):
        num_docs = len(self.data_df)
        pct = np.round(pct_*num_docs)
        indicies_to_drop = []
        for attr in self.list_of_attributes:
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
                    print("all words are frequent", i, text)
                    indicies_to_drop.append(i)
                else:
                    self.data_df.at[i,attr] = new_text
        indicies_to_drop = list(set(indicies_to_drop))
        self.data_df = self.data_df.drop(indicies_to_drop).reset_index(drop=True)
        self.doc_ids = self.data_df[self.doc_ids_label].tolist()
        return
    
    def create_corpus_of_ngrams(self, texts):
        corpus = tp.utils.Corpus()
        for text in texts:
            corpus.add_doc(text)
        #identifies n_grams
        cands = corpus.extract_ngrams(min_cf=5, min_df=1, max_len=3)
        #transforms corpus to contain n_grams
        corpus.concat_ngrams(cands, delimiter=' ')
        return corpus
    
    def find_optimized_lda_topic_num(self, attr, max_topics, training_iterations=1000, iteration_step=10, remove_pct=0.3, **kwargs):
        coherence = []
        LL = []
        perplexity = []
        topic_num = [i for i in range(1, max_topics+1)]
        ##need to address this specifically what percentage is removed
        texts = self.data_df[attr].tolist()
        sleep(0.5)
        for num in tqdm(topic_num, attr+" LDA optimization…"):
            if self.ngrams == "tp":
                corpus = self.create_corpus_of_ngrams(texts)
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
        print(coherence, perplexity)
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
        self.create_folder()
        plt.savefig(self.folder_path+"/LDA_optimization_"+attr+"_.png")
        plt.close()
        """
        plt.figure()
        plt.xlabel("Number of Topics")
        plt.ylabel("Perplexity")
        plt.title("LDA optimization for "+attr)
        plt.plot(topic_num, perplexity, marker='o', color="green")
        plt.show()
        self.create_folder()
        plt.savefig(self.folder_path+"/LDA_optimization_P_"+attr+"_.png")
        
        plt.close()
        plt.figure()
        plt.xlabel("Number of Topics")
        plt.ylabel("Loglikelihood")
        plt.title("LDA optimization for "+attr)
        plt.plot(topic_num, LL, marker='o', color="blue")
        plt.show()
        self.create_folder()
        plt.savefig(self.folder_path+"/LDA_optimization_LL_"+attr+"_.png")
        """
        #want to minimize perplexity, maximize coherence, look for max difference between the two
        diff = [coherence[i]-perplexity[i] for i in range(len(topic_num))]
        change_in_diff = [abs(diff[i]-diff[i+1])-abs(diff[i+1]-diff[i+2]) for i in range(0, len(diff)-2)]
        index_best_num_of_topics = np.argmax(change_in_diff) + 1
        #index_best_num_of_topics = np.argmax(diff)
        best_num_of_topics = topic_num[index_best_num_of_topics]
        self.lda_num_topics[attr] = best_num_of_topics
        
    def lda_optimization(self, max_topics=50,training_iterations=1000, iteration_step=10, remove_pct=0.3, **kwargs):
        #needs work
        start = time()
        self.lda_num_topics = {}
        self.remove_words_in_pct_of_docs()
        for attr in self.list_of_attributes:
            self.find_optimized_lda_topic_num(attr, max_topics, training_iterations=1000, iteration_step=10, remove_pct=0.3, **kwargs)
            print(self.lda_num_topics[attr], " topics for ", attr)
        print("LDA topic optomization: ", (time()-start)/60, " minutes")
    
    def lda(self, num_topics={}, training_iterations=1000, iteration_step=10, remove_pct=0.3, **kwargs):
        start = time()
        self.lda_models = {}
        self.lda_coherence = {}
        self.remove_words_in_pct_of_docs(pct_=remove_pct)
        if num_topics == {}:
            self.lda_optimization(**kwargs)
        else:
            self.lda_num_topics = num_topics
        for attr in self.list_of_attributes:
            texts = self.data_df[attr].tolist()
            if self.ngrams == "tp":
                corpus = self.create_corpus_of_ngrams(texts)
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
        self.create_folder()
        for attr in self.list_of_attributes:
            mdl = self.lda_models[attr]
            mdl.save(self.folder_path+"/"+attr+"_lda_model_object.bin")
    
    def save_lda_document_topic_distribution(self):
        #identical to hlda function except for lda tag
        self.create_folder()
        doc_data = {attr: [] for attr in self.list_of_attributes}
        doc_data['document number'] = self.doc_ids
        for attr in self.list_of_attributes:
            mdl = self.lda_models[attr]
            for doc in mdl.docs:
                doc_data[attr].append(doc.get_topic_dist())
        doc_df = pd.DataFrame(doc_data)
        doc_df.to_csv(self.folder_path+"/lda_topic_dist_per_doc.csv")
        print("LDA topic distribution per document saved to: ",self.folder_path+"/lda_topic_dist_per_doc.csv")
    
    def save_lda_coherence(self):
        self.create_folder()
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
        coherence_df.to_csv(self.folder_path+"/lda_coherence_scores.csv")
    
    def save_lda_taxonomy(self):
        self.create_folder()
        taxonomy_data = {attr:[] for attr in self.list_of_attributes}
        for attr in self.list_of_attributes:
            mdl = self.lda_models[attr]
            for doc in mdl.docs: 
                topic_num = int(doc.get_topics(top_n=1)[0][0])
                num_words = min(mdl.get_count_by_topics()[topic_num], 100)
                words =  ", ".join([word[0] for word in mdl.get_topic_words(topic_num, top_n=num_words)])
                #if len(words) > 35000:
                #    words = words[0:words.rfind(", ")]
                taxonomy_data[attr].append(words)
        taxonomy_df = pd.DataFrame(taxonomy_data)
        #print(taxonomy_df, taxonomy_data)
        taxonomy_df = taxonomy_df.drop_duplicates()
        taxonomy_df.to_csv(self.folder_path+"/lda_taxonomy_test.csv")
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
        taxonomy_df.to_csv(self.folder_path+"/lda_taxonomy.csv")
        print("LDA taxonomy saved to: ", self.folder_path+"/lda_taxonomy.csv")
        
    def lda_extract_models(self, file_path):
        self.lda_models = {}
        for attr in self.list_of_attributes:
            self.lda_models[attr] = tp.LDAModel.load(file_path+attr+"_lda_model_object.bin")
        print("LDA models extracted from: ", file_path)
        
    def lda_visual(self, attr):
        self.create_folder()
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
        pyLDAvis.save_html(prepared_data, self.folder_path+'/'+attr+'_ldavis.html')
        print("LDA Visualization for "+attr+" saved to: "+self.folder_path+'/'+attr+'_ldavis.html')
    
    def hlda (self, levels=3, training_iterations=1000, iteration_step=10, remove_pct=0.3, **kwargs):
        start = time()
        self.hlda_models = {}
        self.hlda_coherence = {}
        self.levels = levels
        self.remove_words_in_pct_of_docs(pct_=remove_pct)
        for attr in self.list_of_attributes:
            texts = self.data_df[attr].tolist()
            if self.ngrams == "tp":
                corpus = self.create_corpus_of_ngrams(texts)
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
    
    def save_hlda_document_topic_distribution(self):
        self.create_folder()
        doc_data = {attr: [] for attr in self.list_of_attributes}
        doc_data['document number']=self.doc_ids
        for attr in self.list_of_attributes:
            mdl = self.hlda_models[attr]
            for doc in mdl.docs:
                doc_data[attr].append(doc.get_topic_dist())
        doc_df = pd.DataFrame(doc_data)
        doc_df.to_csv(self.folder_path+"/hlda_topic_dist_per_doc.csv")
        print("hLDA topic distribution per document saved to: ",self.folder_path+"/hlda_topic_dist_per_doc.csv")
    
    def save_hlda_models(self):
        self.create_folder()
        for attr in self.list_of_attributes:
            mdl = self.hlda_models[attr]
            mdl.save(self.folder_path+"/"+attr+"_hlda_model_object.bin")
            print("hLDA model for "+attr+" saved to: ", (self.folder_path+"/"+attr+"_hlda_model_object.bin"))
            
    def save_hlda_topics(self):
        #saving raw topics with coherence
        self.create_folder()
        for attr in self.list_of_attributes:
            mdl = self.hlda_models[attr]
            topics_data = {"topic level": [],
                "topic number": [],
                "parent": [],
                "number of documents in topic": [],
                "topic words": [],
                "number of words": [],
                "best document": [],
                "coherence": []}
            topics_data["coherence"] = self.hlda_coherence[attr]["per topic"]
            for k in range(mdl.k):
                if not mdl.is_live_topic(k) or mdl.num_docs_of_topic(k)<0:
                    continue
                topics_data["parent"].append(mdl.parent_topic(k))
                topics_data["topic level"].append(mdl.level(k))
                topics_data["number of documents in topic"].append(mdl.num_docs_of_topic(k))
                topics_data["topic number"].append(k)
                topics_data["number of words"].append(mdl.get_count_by_topics()[k])
                topics_data["topic words"].append(", ".join([word[0] for word in mdl.get_topic_words(k, top_n=mdl.get_count_by_topics()[k])]))
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
            df = pd.DataFrame(topics_data)
            df.to_csv(self.folder_path+"/"+attr+"_hlda_topics.csv")
            print("hLDA topics for "+attr+" saved to: ",self.folder_path+"/"+attr+"_hlda_topics.csv")
            
    def save_hlda_coherence(self):
        #saving coherence values
        self.create_folder()
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
        coherence_df.to_csv(self.folder_path+"/"+"hlda_coherence.csv")
        print("hLDA coherence scores saved to: ",self.folder_path+"/"+"hlda_coherence.csv")
    
    def save_hlda_taxonomy(self):
        self.create_folder()
        taxonomy_data = {attr+" Level "+str(level):[] for attr in self.list_of_attributes for level in range(1,self.levels)}
        for attr in self.list_of_attributes:
            mdl = self.hlda_models[attr]
            for doc in mdl.docs: 
                topic_nums = doc.path
                for level in range(1, self.levels):
                    taxonomy_data[attr+" Level "+str(level)].append( ", ".join([word[0] for word in mdl.get_topic_words(topic_nums[level], top_n=500)]))
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
        taxonomy_df.to_csv(self.folder_path+"/hlda_taxonomy.csv")
        print("hLDA taxonomy saved to: ", self.folder_path+"/hlda_taxonomy.csv")
    
    def save_hlda_level_n_taxonomy(self, lev=1):
        self.create_folder()
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
        taxonomy_level_df.to_csv(self.folder_path+"/hlda_level1_taxonomy.csv")
        print("hLDA level "+str(lev)+" taxonomy saved to: ", self.folder_path+"/hlda_level1_taxonomy.csv")
        
    def hlda_extract_models(self, file_path):
        self.hlda_models = {}
        for attr in self.list_of_attributes:
            self.hlda_models[attr]=tp.HLDAModel.load(file_path+attr+"_hlda_model_object.bin")
        print("hLDA models extracted from: ", file_path)
        
    def hlda_display_tp(self, attr, levels = 3, num_words = 5, max_level_1_nodes = 1, max_level_2_nodes = 6):
        #adapt to general number of levels
        try:
            from graphviz import Digraph
        except ImportError as error:
            # Output expected ImportErrors.
            print(error.__class__.__name__ + ": " + error.message)
            print("GraphViz not installed. Please see:\n https://pypi.org/project/graphviz/ \n https://www.graphviz.org/download/")
            return
        df = pd.read_csv(self.folder_path+"/"+attr+"_hlda_topics.csv")
        dot = Digraph(comment="hLDA topic network")
        colors = {'1':"paleturquoise1", '2':"turquoise"}
        level_1_nodes = []; level_2_nodes = 0
        for i in range(2, len(df)):
            if int(df.iloc[i]["topic level"]) == 0 and int(df.iloc[i]["number of documents in topic"]) > 0:
                root_words = df.iloc[i]["topic words"].split(", ")
                root_words = "\\n".join([root_words[i] for i in range(0,min(num_words,int(df.iloc[i]["number of words"])))])
                dot.node(str(df.iloc[i]["topic number"]), root_words)
            elif int(df.iloc[i]["number of documents in topic"])>0 and str(df.iloc[i]["topic level"]) != '0':
                if (len(level_1_nodes) <= max_level_1_nodes and int(df.iloc[i]['topic level']) == 1) or (level_2_nodes <= max_level_2_nodes and int(df.iloc[i]['topic level']) == 2):
                    words = df.iloc[i]["topic words"].split(", ")
                    words = "\\n".join([words[i] for i in range(0,min(num_words,int(df.iloc[i]["number of words"])))])
                    topic_id = df.iloc[i]["topic number"]
                    parent_id = df.iloc[i]["parent"]
                    level = df.iloc[i]['topic level']
                    if int(level) == 2 and parent_id not in level_1_nodes: 
                        continue
                    else:
                        dot.node(str(topic_id), words,color=colors[str(level)], style="filled", fillcolor=colors[str(level)])
                        dot.edge(str(parent_id),str(topic_id))
                        if int(level) == 1 : level_1_nodes.append(topic_id)
                        elif int(level) == 2: level_2_nodes+=1
        dot.attr(layout='twopi')
        dot.attr(overlap="voronoi")
        dot.render(filename = self.folder_path+"/"+attr+"_hlda_network", format = 'png')
