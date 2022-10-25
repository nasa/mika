# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:30:38 2022
To do: 
    - remove correction list variable? 
    - speed up functions: flatten nested functions
@author: srandrad
"""
import pandas as pd
import numpy as np
from time import time, sleep
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from gensim.utils import simple_tokenize
from gensim.models import Phrases
from symspellpy import SymSpell, Verbosity
import pkg_resources
import re

class Data():
    def __init__(self, name=""):
        """
        initializes a data object

        Parameters
        ----------
        name : string, optional
            name of the dataset. The default is "".

        Returns
        -------
        None.

        """
        self.name = name
        self.text_columns = []
        self.data_df = None
        self.doc_ids = []
        self.id_col = None
        self.__english_vocab = set([w.lower() for w in words.words()])
    
    def __update_ids(self):
        """
        updates the document ids variable

        Returns
        -------
        None.

        """
        self.doc_ids = self.data_df[self.id_col].tolist()
    
    def __set_id_col_to_index(self):
        """
        sets the document id column to the dataframe index

        Returns
        -------
        None.

        """
        self.id_col = 'index'
        self.data_df['index'] = self.data_df.index
        
    def __load_preprocessed(self, filename, drop_short_docs=True, drop_short_docs_thres=3, drop_duplicates=True, id_in_dups=True, tokenized=True):
        """
        uses previously saved preprocessed data

        Parameters
        ----------
        filename : string
            File name for extracting preprocessed daya.
        drop_short_docs : Boolean, optional
            True to drop documents shorter than drop_short_docs_thres. The default is True.
        drop_short_docs_thres : int, optional
            Threshold length of document used to drop short docs. The default is 3.
        drop_duplicates : Boolean, optional
            True to drop duplicate documents/rows. The default is True.
        id_in_dups : Boolean, optional
            True to consider the document id when droping duplicates. The default is True.
        tokenized : Boolean, optional
            True if the preprocessed data is tokenized. The default is True.

        Returns
        -------
        None.

        """
        
        self.data_df = pd.read_csv(filename)
        self.data_df = self.data_df.fillna('')
        if tokenized == True:
            self.data_df[self.text_columns] = self.data_df[self.text_columns].applymap(lambda y: self.__remove_quote_marks(y))
        if drop_duplicates == True:
            if id_in_dups: cols = self.text_columns + [self.id_col]
            else: cols = self.text_columns
            self.__drop_duplicate_docs(cols)
        if drop_short_docs == True:
            self.__drop_short_docs(thres=drop_short_docs_thres)
        self.__update_ids()
        cols_to_drop = [col for col in self.data_df.columns if "Unnamed" in col]
        self.data_df = self.data_df.drop(cols_to_drop, axis=1)
        #print("Preprocessed data extracted from: ", file_name)
    
    def __load_raw(self, filename, kwargs):
        """
        loads data from a raw files (.csv or .xlsx)

        Parameters
        ----------
        filename : string
            filename for where the data is stored.
        kwargs : dict
            any kwargs for reading in the file.

        Returns
        -------
        None.

        """
        if ".csv" in filename: 
            self.data_df = pd.read_csv(open(filename,encoding='utf8',errors='ignore'), **kwargs)
        else: 
            self.data_df = pd.read_excel(filename, **kwargs)
        cols_to_drop = [col for col in self.data_df.columns if "Unnamed" in col]
        self.data_df = self.data_df.drop(cols_to_drop, axis=1)
        self.data_df = self.data_df.fillna('')
    
    def load(self, filename, preprocessed=False, id_col=None, text_columns=[], name='', load_kwargs={}, preprocessed_kwargs={}):
        """
        loads in data, either preprocessed or raw

        Parameters
        ----------
        filename : string
            filename for where the data is stored.
        preprocessed : Boolean, optional
            true if the data is preprocessed. The default is False.
        id_col : string, optional
            the column in the dataset where the document ids are stored. The default is None.
        text_columns : list, optional
            list of oclumns in the dataset that contain text. The default is [].
        name : string, optional
            name of the dataset. The default is ''.
        load_kwargs : dict, optional
            dictionary of kwargs for loading raw data. The default is {}.
        preprocessed_kwargs : dict, optional
            dictionary of kwargs for loading preprocessed data. The default is {}.

        Returns
        -------
        None.

        """
        self.text_columns = text_columns
        self.name = name
        self.id_col = id_col
        if preprocessed == True:
            self.__load_preprocessed(filename, **preprocessed_kwargs)
        else:
            self.__load_raw(filename, load_kwargs)
        if id_col == None:
            self.__set_id_col_to_index()
        self.__update_ids()
    
    def save(self, results_path=""):
        """
        saves preprocessed data

        Parameters
        ----------
        results_path : string, optional
            path to save the data to. The default is "".

        Returns
        -------
        None.

        """
        if results_path == "":
            results_path = "preprocessed_data.csv"
        if ".csv" not in results_path:
            results_path+=".csv"
        self.data_df.to_csv(results_path, index=False)
        #print("Preprocessed data saves to: ", results_path)
    
    def __combine_columns(self, combine_columns):
        """
        combines text columns into one combined text field

        Parameters
        ----------
        combine_columns : list
            list of columns to combine.

        Returns
        -------
        None.

        """
        combined_text = []
        for i in tqdm(range(0, len(self.data_df)), "Combining Columns…"):
            text = ""
            for col in combine_columns:
                if text != "":
                    text += ". " 
                text += str(self.data_df.iloc[i][col])
            combined_text.append(text)
        self.data_df["Combined Text"] = combined_text
        self.combined_text_col = ["Combined Text"]
        self.text_columns += self.combined_text_col
        
    def __remove_incomplete_rows(self):
        """
        removes incomplete rows

        Returns
        -------
        None.

        """
        rows_to_drop = []
        for i in tqdm(range(0, len(self.data_df)), "Removing Incomplete Rows…"):
            for col in self.text_columns:
                if str(self.data_df.iloc[i][col])=="" or isinstance(self.data_df.iloc[i][col],float):
                    rows_to_drop.append(i)
        self.data_df = self.data_df.drop(list(set(rows_to_drop)), axis=0)
        self.data_df = self.data_df.reset_index(drop=True)
    
    def __create_unique_ids(self):
        """
        creates unique ids for datasets that may have multiple documents under the same id

        Returns
        -------
        None.

        """
        unique_ids = []
        prev_id = None
        j = 0
        for i in tqdm(range(len(self.data_df)), "Creating Unique IDs…"):
            current_id = self.data_df.iloc[i][self.id_col]
            if type(current_id) is not str:
                current_id = str(current_id)
            if current_id == prev_id:
                j+=1
            else:
                j=0
            unique_ids.append(current_id+"_"+str(j))
            prev_id = current_id
        self.data_df["Unique IDs"] = unique_ids
        self.id_col= "Unique IDs"
        self.doc_ids = self.data_df["Unique IDs"].tolist()
        
    def prepare_data(self, combine_columns=[], remove_incomplete_rows=True, create_ids=False):
        """
        Prepares data by creating unique ids, removing and combining rows/cols as defined by user

        Parameters
        ----------
        combine_columns : list, optional
            list of oclumns to combine. The default is [].
        remove_incomplete_rows : boolean, optional
            true to remove incomplete rows. The default is True.
        create_ids : boolean, optional
            true to create unique ids. The default is False.

        Returns
        -------
        None.

        """
        start_time = time()
        if combine_columns != []: 
            self.__combine_columns(combine_columns)
        if remove_incomplete_rows == True:
            self.__remove_incomplete_rows()
        if create_ids == True:
            self.__create_unique_ids()
        self.__update_ids()
        print("data preparation: ", round((time()-start_time)/60,2),"minutes \n")
    
    def sentence_tokenization(self):
        """
        tokenizes each document in the dataset into sentences. 
        creates an updated data_df where each sentence has a separate row

        Returns
        -------
        None.

        """
        dfs = []
        for i in range(len(self.data_df)):
            sentences_for_doc = {col:[] for col in self.text_columns}
            for col in self.text_columns:
                text = self.data_df.at[i, col]
                punctuation = re.findall('[.!?]', text)
                sentences_for_doc[col] = list(filter(None, re.split('[.!?]', text)))
                sentences_for_doc[col] = [sentences_for_doc[col][i].strip(" ")+punctuation[i] for i in range(len(sentences_for_doc[col]))]
                #sentences_for_doc[col] = #text.split(".")
            num_rows = max([len(sentences_for_doc[col]) for col in self.text_columns])
            for col in self.text_columns:
                while len(sentences_for_doc[col])<num_rows:
                    sentences_for_doc[col].append("")
            doc_df = pd.concat([self.data_df.loc[i:i][:] for j in range(num_rows)]).reset_index(drop=True)
            for col in self.text_columns:
                doc_df[col+' Sentences'] = sentences_for_doc[col]
            dfs.append(doc_df)
        self.dfs = dfs
        self.data_df = pd.concat(dfs).reset_index(drop=True)#, ignore_index=True)
        self.text_columns_sentences = [col+' Sentences' for col in self.text_columns]
        self.__update_ids()
        return
    
    def __remove_quote_marks(self, word_list):
        """
        removes quotation marks and brackets from a string

        Parameters
        ----------
        word_list : string
            list of words in string format.

        Returns
        -------
        word_list : list
            list of words in list format.

        """
        word_list = word_list.strip("[]").split(", ")
        word_list = [w.replace("'","") for w in word_list]
        return word_list

    def __drop_duplicate_docs(self, cols):
        """
        drops duplicate rows from a dataset

        Parameters
        ----------
        cols : list
            list of columns used for determining duplicate reports.

        Returns
        -------
        None.

        """
        self.data_df = self.data_df.iloc[self.data_df.astype(str).drop_duplicates(subset=cols, keep='first').index,:].reset_index(drop=True)
        
    def __drop_short_docs(self, thres=3):
        """
        drops documents with len<thresh from the dataset

        Parameters
        ----------
        thres : int, optional
            threshold for length of a document. The default is 3.

        Returns
        -------
        None.

        """
        indx_to_drop = []
        for i in range(len(self.data_df)):
            for col in self.text_columns:
                text = self.data_df.at[i, col]
                if type(text) is str:
                    text = text.split(" ")
                if len(text)<thres:
                    indx_to_drop.append(i)
        self.data_df = self.data_df.drop(indx_to_drop).reset_index(drop=True)
    
    def __tokenize_texts(self, texts, min_word_len, max_word_len):
        """
        tokenizes texts

        Parameters
        ----------
        texts : list
            list of doucment texts.
        min_word_len : int
            minimum word length.
        max_word_len : int
            maximum word length.

        Returns
        -------
        texts : list
            list of tokenized doucment texts.

        """
        texts = [simple_tokenize(text) for text in texts if not isinstance(text,float)]
        texts = [[word for word in text if len(word)>min_word_len and len(word)<max_word_len] for text in texts if not isinstance(text,float)]
        return texts
        
    def __lowercase_texts(self,texts):
        """
        converts all text to lowercase

        Parameters
        ----------
        texts : list
            list of doucment texts.

        Returns
        -------
        texts : list
            list of doucment texts with all letters lowercase.

        """
        texts = [[word.lower() for word in text] for text in texts]
        return texts
    
    def __get_wordnet_pos(self, word):
        """
        returns the wordnet pos for a word

        Parameters
        ----------
        word : string
            a word.

        Returns
        -------
        None.

        """
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,"N": wordnet.NOUN,"V": wordnet.VERB,"R": wordnet.ADV}
        if tag not in ['I','D','M','T','C','P']:return tag_dict.get(tag,wordnet.NOUN)
        else: return "unnecessary word"
        
    def __lemmatize_texts(self,texts): 
        """
        lemmatizes texts

        Parameters
        ----------
        texts : list
            list of doucment texts.

        Returns
        -------
        texts : list
            list of lemmatized doucment texts.

        """
        #why is this so slow?
        lemmatizer = WordNetLemmatizer()
        texts = [[lemmatizer.lemmatize(w, self.__get_wordnet_pos(w)) for w in text if self.__get_wordnet_pos(w)!="unnecessary word"] for text in texts if not isinstance(text,float)]
        return texts
        
    def __remove_stopwords(self, texts, domain_stopwords):
        """
        removes stop words from texts

        Parameters
        ----------
        texts : list
            list of doucment texts.
        domain_stopwords : list
            list of domain specific stopwords.

        Returns
        -------
        texts : list
            list of doucment texts with stopwords removed.

        """
        all_stopwords = stopwords.words('english') + domain_stopwords
        all_stopwords = [word.lower() for word in all_stopwords]
        texts = [[w for w in text if not w in all_stopwords] for text in texts if not isinstance(text,float)]
        texts = [[w for w in text if len(w)>=3] for text in texts if not isinstance(text,float)]
        return texts
    
    def __quot_replace(self, word):
        """
        replaces 'quot' in words

        Parameters
        ----------
        word : string
            word to be fixed.

        Returns
        -------
        word : string
            word with 'quot' removed.

        """
        if word not in self.__english_vocab:
            w_tmp = word.replace('quot','')
            if w_tmp in self.__english_vocab:
                word = w_tmp
        return word
    
    def __quot_normalize(self, texts):
        """
        replaces 'quot' in words in a set of texts

        Parameters
        ----------
        texts : list
            list of doucment texts.

        Returns
        -------
        texts : list
            list of doucment texts with 'quot' removed from words.

        """
        texts = [[self.__quot_replace(word) for word in text] for text in texts if not isinstance(text,float)]
        return texts
    
    def __spelling_replace(self, word, sym_spell, correction_list):
        """
        replaces a mispelled word with a corrected word

        Parameters
        ----------
        word : string
            word checkd for mispelling.
        sym_spell : symspell object
            object used for spell checking.
        correction_list : list
            list of mispelled words and their corrections.

        Returns
        -------
        word : TYPE
            DESCRIPTION.

        """
        if word not in self.__english_vocab and not word.isupper() and not sum(1 for c in word if c.isupper()) > 1:
            suggestions = sym_spell.lookup(word,Verbosity.CLOSEST,           max_edit_distance=2,include_unknown=True,transfer_casing=True)
            correction = suggestions[0].term
            correction_list.append(word+' --> '+correction)
            word = correction
        return word
    
    def __spellchecker(self, texts, correction_list):
        """
        replaces mispelled words with correct words in a collection of texts

        Parameters
        ----------
        texts : list
            list of doucment texts.
        correction_list : list
            list of mispelled words and their corrections.

        Returns
        -------
        texts : list
            list of doucment texts with mispelled words corrected.

        """
        sym_spell = SymSpell()
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        texts = [[self.__spelling_replace(word, sym_spell, correction_list) for word in text] for text in texts if not isinstance(text,float)]
        return texts
    
    def __segment_replace(self, text, sym_spell, correction_list):
        """
        replaces mispelled words joined together with corrected words in a text

        Parameters
        ----------
        text : list
            list of tokens in a text.
        sym_spell : symspell object
            object used for spell checking
        correction_list : list
            list of mispelled words and their corrections.

        Returns
        -------
        text : list
            list of tokens in a text with words segmented.

        """
        for word in text:
            if word not in self.__english_vocab and not word.isupper():
                segmented_word = sym_spell.word_segmentation(word).corrected_string
                if len(segmented_word.split())>1:
                    text_str = ' '.join(text)
                    text_str = text_str.replace(word,segmented_word)
                    text = text_str.split()
                    correction_list.append(word+' --> '+segmented_word)
        return text
    
    def __segment_text(self, texts, correction_list):
        """
        performs word segmentation on a collection of texts

        Parameters
        ----------
        texts : list
            list of doucment texts.
        correction_list : list
            list of mispelled words and their corrections.

        Returns
        -------
        texts : list
            list of doucment texts with words segmented.

        """
        sym_spell = SymSpell()
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        texts = [self.__segment_replace(text, sym_spell, correction_list) for text in texts if not isinstance(text,float)]
        return texts
        
    def __trigram_texts(self, texts, ngram_range, threshold, min_count):
        """
        generates ngrams from a corpus of texts

        Parameters
        ----------
        texts : list
            list of doucment texts.
        ngram_range : int
            highest level for ngrams, i.e. most number of words to be considered
        threshold : int
            threshold used in gensim phrases
        min_count : int
            minimum count an ngram must occur in the corpus to be considered an ngram

        Returns
        -------
        ngrams : list
            list of document texts as ngrams.

        """
        #NEEDS WORK - could probably replace with a BERT tokenizer
        #very slow
        # changes word order!
        ngrams = []
        ngram_models = {}
        
        for n in range(2, ngram_range+1):
            if n == 2:
                text_input = texts
            elif n > 2:
                text_input = ngram_models[str(n-1)+"gram"][texts]
            ngram_models[str(n)+"gram"]=Phrases(text_input, min_count=min_count, delimiter=' ', threshold=threshold)
        
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
    
    def preprocess_data(self, domain_stopwords=[], ngrams=True, ngram_range=3, threshold=15, 
                        min_count=5, quot_correction=False, spellcheck=False, segmentation=False, 
                        drop_short_docs_thres=3, percent=0.3, drop_na=False, save_words=[], drop_dups=False,
                        min_word_len=2, max_word_len=15):
        """
        performs data preprocessing steps as defined by user

        Parameters
        ----------
        domain_stopwords : list, optional
            list of domain specific stopwords. The default is [].
        ngrams : boolean, optional
            true to generate ngrams. The default is True.
        ngram_range : int, optional
            highest level for ngrams, i.e. most number of words to be considered. The default is 3.
        threshold : int, optional
            threshold used in gensim phrases. The default is 15.
        min_count : int, optional
            minimum count an ngram must occur in the corpus to be considered an ngram. The default is 5.
        quot_correction : boolean, optional
            true to perform 'quot' normalization. The default is False.
        spellcheck : boolean, optional
            true to use symspell spellchecker. The default is False.
        segmentation : boolean, optional
            true to use word segmentation spell checker. The default is False.
        drop_short_docs_thres : int, optional
           Threshold length of document used to drop short docs.. The default is 3.
        percent : float, optional
            removes words in greater than or equal to this percent of documents. The default is 0.3.
        drop_na : boolean, optional
            true to drop rows with nan values. The default is False.
        save_words : list, optional
            list of words to save from frequent word removal. The default is [].
        drop_dups : boolean, optional
            true to drop duplicate rows. The default is False.
        min_word_len : int, optional
           minimum word length.. The default is 2.
        max_word_len : int, optional
            maximum word length. The default is 15.

        Returns
        -------
        correction_list : list
            list of mispelled words and their corrections if spelling correcting is used.

        """
        
        if ngrams == True:
            self.ngrams = "custom"
        else: 
            self.ngrams = "tp"
        start = time()
        correction_list = []
        for col in self.text_columns:
            pbar = tqdm(total=100, desc="Preprocessing "+col+"…")
            self.data_df[col] = self.__tokenize_texts(self.data_df[col], min_word_len, max_word_len)
            pbar.update(10)
            if quot_correction == True:
                self.data_df[col] = self.__quot_normalize(self.data_df[col])
            pbar.update(10)
            if spellcheck == True:
                self.data_df[col] = self.__spellchecker(self.data_df[col], correction_list)
            pbar.update(10)
            if segmentation == True:
                self.data_df[col] = self.__segment_text(self.data_df[col], correction_list)
            pbar.update(10)
            self.data_df[col] = self.__lowercase_texts(self.data_df[col])
            pbar.update(10)
            self.data_df[col] = self.__lemmatize_texts(self.data_df[col])
            pbar.update(20)
            self.data_df[col] = self.__remove_stopwords(self.data_df[col], domain_stopwords)
            pbar.update(20)
            if ngrams == True:
                self.data_df[col] = self.__trigram_texts(self.data_df[col], ngram_range, threshold, min_count)
            pbar.update(10)
            sleep(0.5)
            pbar.close()
                
        if drop_dups: 
            self.__drop_duplicate_docs(self.text_columns)
        if drop_short_docs_thres>0:
            self.__drop_short_docs(thres=drop_short_docs_thres) # need to drop short docs, else we'll get NaNs because of the next two lines; but we can set the thres so this works for the test case
        #replaces empty text or numbers in text cols with nans
        cols = self.text_columns #self.data_df.columns.difference([self.id_col]+self.extra_cols)
        self.data_df[cols] = self.data_df[cols].applymap(lambda y: np.nan if (type(y)==int or type(y)==float or len(y)==0) else y) 
        if drop_na: 
            self.data_df = self.data_df.dropna(how="any").reset_index(drop=True)
        self.data_df = self.__remove_words_in_pct_of_docs(self.data_df, pct_=percent, save_words=save_words) #also slow
        self.__update_ids()
        print("Processing time: ", round((time()-start)/60,2), " minutes")
        if correction_list != []:
            return correction_list
                
    def __remove_words_in_pct_of_docs(self, data_df, pct_=0.3, save_words=[]):
        """
        

        Parameters
        ----------
        data_df : pandas dataframe
            dataframe of the dataset
        pct_ : float, optional
            removes words in greater than or equal to this percent of documents. The default is 0.3.
        save_words : list, optional
            list of words to save from frequent word removal. The default is [].

        Returns
        -------
        pandas dataframe
            self.data_df updated with frequent words removed.

        """
        num_docs = len(data_df)
        pct = np.round(pct_*num_docs)
        indicies_to_drop = []
        for col in tqdm(self.text_columns,"Removing frequent words…"):
            all_words = list(set([word for text in data_df[col] for word in text]))
            good_words = save_words
            for word in all_words:
                count = 0
                for text in data_df[col]:
                    if word in text:
                        count+=1
                if count<pct:
                    good_words.append(word)
            i = 0
            for text in data_df[col]:
                text = [word for word in text if word in good_words]
                data_df.at[i, col] = text
                if text == []:
                    indicies_to_drop.append(i)
                i+=1
        indicies_to_drop = list(set(indicies_to_drop))
        self.data_df = data_df.drop(indicies_to_drop).reset_index(drop=True)
        return self.data_df
