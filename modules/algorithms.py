import modules
import re
import tqdm
import numpy as np
from nltk.utils import ngrmas
from collections import defaultdict


class NaieveBayesForDC():
    
    
    def __init__(
        self,
        n_gram : int = 1,
        use_laplace_smoothing : bool = False,
        tokenizer = None,
    ):
        self.n_gram = n_gram
        self.use_laplace_smoothing = use_laplace_smoothing
        self.tokenizer = tokenizer
        self.n_gram_table = None
        self.n_gram_map = dict()
    
    
    def _update_n_gram_map_and_get_ngrams(
        self,
        document,
        n_gram_map : dict
    ) -> list :
        ##TODO:
        ## implement this method
        
        ## make n_gran list
        n_grams = []
        
        ## update n_grma
        
        ## get n_gram_as_token_idx
        n_grams_tokens = []
        
        return n_grams_tokens
        
    
    def create_n_gram_table(
        self,
        dataloader : modules.DataLoader
    ) -> dict :

        number_of_class = dataloader.dataset.get_class_num()
        n_gram_table = defaultdict(np.zeros(number_of_class))
        
        for input, label in tqdm(dataloader, total = dataloader.len):
            one_hot_encode = np.zeros(number_of_class)
            one_hot_encode[label] = 1
            
            n_grams_tokens = self._update_n_gram_map_and_get_ngrams(input, self.n_gram_map)
            for token in n_grams_tokens :
                n_gram_table[token] = n_gram_table[token] + one_hot_encode
        
        if self.use_laplace_smoothing:
            n_gram_table + 1
        
        self.n_gram_table = n_gram_table
        
        return n_gram_table
    
    
    def classify_document(
        self,
        document,
        num_of_labels
    ) -> int :
        
        if not self.n_gram_table:
            raise('Need to make n_gram_table before inferencing, use ".create_n_gram_table"')
        
        n_grams = list(ngrmas(document, self.n_gram))
        tokens = []
        for token in n_grams:
            try:
                ## appeard in map
                tokens.append(self.n_gram_map[token])
            except:
                ## not appeared in map
                None
        
        frequency = np.zeros(num_of_labels)
        for token_index in tokens:
            frequency = frequency + self.n_gram_table[token_index]
        
        label_rank = np.argsort(frequency)
        
        return label_rank[0]
    
    
    def inference(
        self,
        dataloader : modules.DataLoader
    ) :
        number_of_class = dataloader.dataset.get_class_num()
        confusion_matrix = np.zeros((number_of_class, number_of_class))
        
        for input, label in tqdm(dataloader, total = dataloader.len):
            
            prediction = self.classify_document(input,
                                                num_of_labels = number_of_class)
            
            confusion_matrix[label, prediction] + 1
        
        evaluation = modules.evaluate_confusion_matrix(confusion_matrix)
        
        return evaluation