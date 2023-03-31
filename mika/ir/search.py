# hswalsh
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
    column_to_search : str
        name of column to search - use "Combined Text" if it is desired to search all text columns (must first be defined in Data object)
    model :
        
    """

    def __init__(self, column_to_search, data=None, model=''):
        self.sbert_model = model
        self.doc_ids = data.data_df[data.id_col]
        
        self.corpus = data.data_df[column_to_search].tolist()
        self.corpus_ids = self.doc_ids.tolist()
        self.__make_sentence_corpus()
        self.__make_passage_corpus()
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

    def __make_passage_corpus(self):
        self.passage_corpus = []
        self.passage_doc_idx = []
        passage_length = 3
        i = 0
        for doc in self.corpus:
            sentences = sent_tokenize(doc)
            num_sentences = len(sentences)
            for start_idx in range(0, num_sentences, passage_length):
                end_idx = min(start_idx+passage_length, num_sentences)
                self.passage_corpus.append(" ".join(sentences[start_idx:end_idx]))
                self.passage_doc_idx.append(i)
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
        embeddings_as_numpy = self.sentence_corpus_embeddings.cpu().numpy()
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
            Doc ID, scores, and text of top k hits, structured in a DataFrame
        """
        
        # semantic search

        k_range = range(0,return_k)
        query_embedding = self.sbert_model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.sentence_corpus_embeddings, top_k=return_k)
        top_hit_ids = [hits[0][k]['corpus_id'] for k in k_range]
        top_hit_idx = [self.sentence_doc_idx[id_k] for id_k in top_hit_ids]
        top_hit_scores = [hits[0][k]['score'] for k in k_range]
        top_hit_text = [self.corpus[self.sentence_doc_idx[id_k]] for id_k in top_hit_ids]
        top_hit_doc = [self.corpus_ids[top_hit_idx[id_j]] for id_j in range(0,len(top_hit_idx))]
        return_data = {'top_hit_doc': top_hit_doc, 'top_hit_scores': top_hit_scores, 'top_hit_text': top_hit_text}
        top_hits = pd.DataFrame(return_data)

        # reranker

        return top_hits
