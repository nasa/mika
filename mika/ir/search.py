# hswalsh
#from mika.utils.remove_nans import remove_nans
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
import torch

class search():
    """
    Class to perform information retrieval using semantic search.
    
    PARAMETERS
    ----------
    data : Data() object
        Data object loaded using mika.utils.Data
    model : str
        Either a path to a custom model or a name of a pretrained model.
    """

    def __init__(self, data=None, model=''):
        self.cols = data.text_columns
        self.sbert_model = SentenceTransformer(model)
        
        self.corpus = [data.data_df[col].tolist() for col in self.cols]
        self.corpus = [item for sublist in self.corpus for item in sublist]
        #self.corpus = remove_nans(self.corpus)
        self.__make_sentence_corpus()
        return
    
    def __make_sentence_corpus(self):
        self.sentence_corpus = []
        self.sentence_doc_idx = []
        i = 0
        for doc in self.corpus:
            sentences = sent_tokenize(doc)
            for sentence in sentences:
                self.sentence_corpus.append(sentence)
                self.sentence_doc_idx.append(i)
            i = i + 1
    
    def get_sentence_embeddings(self, savepath):
        """
        Compute sentence embeddings for the corpus.
        
        PARAMETERS
        ----------
        savepath : str
            Filepath to save sentence embeddings
        
        RETURNS
        -------
        None
        """
        
        self.__make_sentence_corpus()
        self.sentence_corpus_embeddings = self.sbert_model.encode(self.sentence_corpus, convert_to_tensor=True)
        embeddings_as_numpy = self.sentence_corpus_embeddings.numpy()
        np.save(savepath, embeddings_as_numpy)
        
    def load_sentence_embeddings(self, filepath):
        """
        Load previously computed sentence embeddings.
        
        PARAMETERS
        ----------
        filepath : str
            Path to saved sentence embeddings
            
        RETURNS
        -------
        None
        """
        embeddings_as_numpy = np.load(filepath)
        self.sentence_corpus_embeddings = torch.from_numpy(embeddings_as_numpy)
    
    def run_search(self, query, return_k=1):
        """
        Run the search using a query and loaded corpus embeddings.
        
        PARAMETERS
        ----------
        query : str
            Query to search
        return_k : int
            Number of results to return
        
        RETURNS
        -------
        top_hits : DataFrame
            Index, scores, and text of top k hits, structured in a DataFrame
        """
        
        k_range = range(0,return_k)
        query_embedding = self.sbert_model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.sentence_corpus_embeddings, top_k=return_k)
        top_hit_ids = [hits[0][k]['corpus_id'] for k in k_range]
        top_hit_idx = [self.sentence_doc_idx[id_k] for id_k in top_hit_ids]
        top_hit_scores = [hits[0][k]['score'] for k in k_range]
        top_hit_text = [self.corpus[self.sentence_doc_idx[id_k]] for id_k in top_hit_ids]
        top_hits = pd.DataFrame({'top_hit_idx': top_hit_idx, 'top_hit_scores': top_hit_scores, 'top_hit_text': top_hit_text})
        return top_hits
