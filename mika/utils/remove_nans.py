# hswalsh
import pandas as pd

def remove_nans(docs):
    """
    Removes nans from a list of documents and replaces with ''.
    """
    
    for i in range(0,len(docs)):
        if pd.isnull(docs[i]):
            docs[i] = ''
    return docs
