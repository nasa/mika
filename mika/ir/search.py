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
    retrieval_model :
    reranker_model :
        
    """

    def __init__(self, column_to_search, data=None, retrieval_model='', reranker_model=''):
        self.retrieval_model = retrieval_model
        self.reranker_model = reranker_model
        self.doc_ids = data.data_df[data.id_col]
        
        self.corpus = data.data_df[column_to_search].tolist()
        self.corpus_ids = self.doc_ids.tolist()
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

    def __make_passages(self, retrieved_docs, retrieved_doc_ids, passage_length=3):
        passage_corpus = []
        passage_doc_idx = []
        i = 0
        for doc in retrieved_docs:
            sentences = sent_tokenize(doc)
            num_sentences = len(sentences)
            for start_idx in range(0, num_sentences, passage_length):
                end_idx = min(start_idx+passage_length, num_sentences)
                passage_corpus.append(" ".join(sentences[start_idx:end_idx]))
                passage_doc_idx.append(retrieved_doc_ids[i])
            i = i + 1
        return (passage_corpus, passage_doc_idx)

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
        self.sentence_corpus_embeddings = self.retrieval_model.encode(self.sentence_corpus, convert_to_tensor=True)
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
    
    def __semantic_search(self, query, return_k=1):
        k_range = range(0,return_k)
        query_embedding = self.retrieval_model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.sentence_corpus_embeddings, top_k=return_k)
        top_hit_sentence_ids = [hits[0][k]['corpus_id'] for k in k_range]
        top_hit_idx = [self.sentence_doc_idx[id_k] for id_k in top_hit_sentence_ids]
        top_hit_scores = [hits[0][k]['score'] for k in k_range]
        top_hit_text = [self.corpus[self.sentence_doc_idx[id_k]] for id_k in top_hit_sentence_ids]
        top_hit_doc = [self.corpus_ids[top_hit_idx[id_j]] for id_j in range(0,len(top_hit_idx))]
        return_data = {'top_hit_doc': top_hit_doc, 'top_hit_scores': top_hit_scores, 'top_hit_text': top_hit_text}
        top_hits = pd.DataFrame(return_data)
        return top_hits

    def __rerank(self, query, top_hits, use_passages):
        if use_passages == True:
            rerank_corpus, rerank_doc_idx = self.__make_passages(top_hits['top_hit_text'], top_hits['top_hit_doc'], passage_length=3)
        else:
            rerank_corpus = top_hits['top_hit_text'].to_list()
            rerank_doc_idx = top_hits['top_hit_doc'].to_list()
        reranker_inputs = [[query, text] for text in rerank_corpus]
        scores = self.reranker_model.predict(reranker_inputs)
        rerank_results = [{'input': input, 'score': score} for input, score in zip(reranker_inputs, scores)]
        rerank_results = sorted(rerank_results, key=lambda x: x['score'], reverse=True)
        ranked_hit_scores = []
        ranked_hit_text = []
        ranked_hit_doc = []
        for result in rerank_results:
            result_text = result['input'][1]
            ranked_hit_scores.append(result['score'])
            ranked_hit_text.append(result_text)
            ranked_hit_doc.append(rerank_doc_idx[rerank_corpus.index(result_text)])
        ranked_data = {'ranked_hit_doc': ranked_hit_doc, 'ranked_hit_scores': ranked_hit_scores, 'ranked_hit_text': ranked_hit_text}
        ranked_hits = pd.DataFrame(ranked_data)
        return ranked_hits
    
    def run_search(self, query, rank_k=1, return_k=1, use_passages=False):
        """
        Run the search using a query and loaded corpus embeddings.
        
        PARAMETERS
        ----------
        query : str
            Query to search
        rank_k : int
            Number of results to rank in semantic search
        return_k : int
            Number of results to return (must be less than or equal to rank_k)
        
        RETURNS
        -------
        ranked_hits : DataFrame
            Doc ID, scores, and text of top k hits, structured in a DataFrame
        """

        top_hits = self.__semantic_search(query, rank_k)
        ranked_hits = self.__rerank(query, top_hits[0:return_k], use_passages)

        return ranked_hits
