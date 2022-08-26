"""
@author: hswalsh
"""

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from gensim.utils import simple_tokenize
from symspellpy import SymSpell, Verbosity
import pkg_resources

class preprocess_df():
    
    __english_vocab = set([w.lower() for w in words.words()])
    
    def __init__(self, df, columns, domain_stopwords = ['']):
        self.df = df
        self.columns = columns
        self.min_word_len = 2
        self.max_word_len = 15
        self.correction_list = []
        self.df = self.preprocess(domain_stopwords = domain_stopwords)
    
    def tokenize_texts(self, texts):
        new_texts = []
        for text in texts:
            if isinstance(text,float):
                text = []
            else:
                text = simple_tokenize(text)
                text = [word for word in text if len(word)>self.min_word_len and len(word)<self.max_word_len]
            new_texts.append(text)
        return new_texts
        
    def lowercase_texts(self, texts):
        new_texts = []
        for text in texts:
            if isinstance(text,float):
                text = []
            else:
                text = [word.lower() for word in text]
            new_texts.append(text)
        return new_texts
        
    def lemmatize_texts(self, texts):
        def get_wordnet_pos(word):
            tag = pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,"N": wordnet.NOUN,"V": wordnet.VERB,"R": wordnet.ADV}
            if tag not in ['I','D','M','T','C','P']:return tag_dict.get(tag,wordnet.NOUN)
            else: return "unnecessary word"
        lemmatizer = WordNetLemmatizer()
        new_texts = []
        for text in texts:
            if isinstance(text,float):
                text = []
            else:
                text = [lemmatizer.lemmatize(w,get_wordnet_pos(w)) for w in text if get_wordnet_pos(w)!="unnecessary word"]
            new_texts.append(text)
        return new_texts
        
    def remove_stopwords(self, texts, domain_stopwords):
        all_stopwords = stopwords.words('english')+domain_stopwords
        all_stopwords = [word.lower() for word in all_stopwords]
        new_texts = []
        for text in texts:
            if isinstance(text,float):
                text = []
            else:
                text = [w for w in text if not w in all_stopwords]
                text = [w for w in text if len(w)>=3]
            new_texts.append(text)
        return new_texts

    def quot_normalize(self, texts):
        def quot_replace(word):
            if word not in self.__english_vocab:
                w_tmp = word.replace('quot','')
                if w_tmp in self.__english_vocab:
                    word = w_tmp
            return word
        new_texts = []
        for text in texts:
            if isinstance(text,float):
                text = []
            else:
                text = [quot_replace(word) for word in text]
            new_texts.append(text)
        return new_texts

    def spellchecker(self, texts):
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
        new_texts = []
        for text in texts:
            if isinstance(text,float):
                text = []
            else:
                text = [spelling_replace(word) for word in text]
            new_texts.append(text)
        return new_texts

    def segment_text(self, texts):
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
        new_texts = []
        for text in texts:
            if isinstance(text,float):
                text = []
            else:
                text = segment_replace(text)
            new_texts.append(text)
        return new_texts
    
    def preprocess(self, domain_stopwords = ['']):
        for column in self.columns:
            texts = self.df[column].to_list()
            texts = self.tokenize_texts(texts)
            texts = self.lowercase_texts(texts)
            texts = self.lemmatize_texts(texts)
            texts = self.remove_stopwords(texts, domain_stopwords)
            texts = self.quot_normalize(texts)
            texts = self.spellchecker(texts)
            texts = self.segment_text(texts)
            self.df[column] = texts
        return self.df
            
