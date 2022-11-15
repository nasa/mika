# hswalsh
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd

class custom_ir_model():
    """
    Class to train an SBERT model for IR. 
    
    PARAMETERS
    ----------
    base_model : SBERT model object
        Base SBERT model to fine-tune
        
    training_data : Data object
        Training data imported as Data object
        
    RETURNS
    -------
    None.
    
    """

    def __init__(self, base_model=None, training_data=None):
        self.base_model = base_model
        self.training_data = training_data
        self.cols = training_data.text_columns
        return

    def __setup_sentence_data(self):
        return
        
    def __split_data(self):
        return
    
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
    
    def prepare_training_data(self, tokenizer, model, save_filepath=None, max_length=64, do_sample=True, top_p=0.95, num_return_sequences=3):
        """
        Prepares data for fine tuning the model.
        
        PARAMETERS
        ----------
        tokenizer :
        
        model :
        
        save_filepath : str
            filepath to save prepared training data
        
        RETURNS
        -------
        None.
        
        """
        
        # define training corpus based on designated text columns in Data object; this can be changed using cols
        training_corpus = self.training_data.data_df[self.cols].agg(' '.join, axis=1)
                
        input_ids = tokenizer.encode(training_corpus, return_tensors='pt')
        outputs = model.generate(input_ids=input_ids, max_length=max_length, do_sample=do_sample, top_p=top_p, num_return_sequences=num_return_sequences)
        training_data = []
        for i in range(len(outputs)):
            training_data.append([tokenizer.decode(outputs[i], skip_special_tokens=True), training_corpus[i]])
        training_data = pd.DataFrame(training_data)
        self.training_data = training_data
        training_data.to_csv(save_filepath)
        
    def fine_tune_model(self, data_filepath=None, train_batch_size=16, model=None, num_epochs=3, model_name='custom_model'):
        """
        Fine tunes the specified SBERT model.
        
        PARAMETERS
        ----------
        
        RETURNS
        -------
                
        """
        
        if data_filepath == None:
            training_data = self.training_data
        else:
            training_data = pd.read_csv(data_filepath)
        training_data = training_data.values.tolist()
        self.train_examples = []
        for i in range(0,len(training_data)):
            self.train_examples.append(InputExample(texts=[training_data[i][0], training_data[i][1]]))
               
        # setup data loader and loss function
        train_batch_size = train_batch_size
        train_dataloader = NoDuplicatesDataLoader(self.train_examples, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)
        
        # fine tune the model
        num_epochs = num_epochs
        warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=warmup_steps, show_progress_bar=True)
        
        # save the model
        model.save(os.path.join('results',model_name)

    
