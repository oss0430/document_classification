
import pandas as pd
import re
import tqdm
import numpy as np
#from nltk.util import ngrmas
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

class DocumnetClassificationAlgorithms():
    
    
    def __init__(
        self,
        categories,
        *args,
        **kwargs
    ):

        self.categories = categories
        
        
    def document_to_words(
        self,
        document 
    ) :
        None
        
        
    def score_document(
        self,
        documents : list,
    ) ->  float :
        None
        
    
    def label_document(
        self,
        document
    ) -> None:
        None
        
        
    def evaluate(
        self,
        evaluation_document
    ) :
        None
        

class KnnForDC(DocumnetClassificationAlgorithms):
    
    def __init__(
        self,
        categories,
        *args,
        **kwargs
    ):
        super().__init__(self, categories, *args, **kwargs)
        self.classifier = KNeighborsClassifier(n_neighbors = len(categories))
        self.tfidf = TfidfVectorizer()
        
    def train(
        self,
        dataset,
        method
    ) -> KNeighborsClassifier :
        
        train_data = dataset.get_dataset_for_scikitlearn(method)

        X_train = self.tfidf.fit_transform(train_data["X"])
        y_train = train_data["Y"]

        self.classifier.fit(X_train, y_train)
        
    
    def evaluate(
        self,
        dataset,
        method
    ) -> dict :
        
        test_data = dataset.get_dataset_for_scikitlearn(method)

        X_test = self.tfidf.transform(test_data["X"])
        y_test = test_data["Y"]
        
        y_pred = self.classifier.predict(X_test)
 
        return {"acc" : accuracy_score(y_test, y_pred),
                "confusion_matrix" : confusion_matrix(y_test, y_pred)
                }      


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
        dataloader
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
        dataloader
    ) :
        number_of_class = dataloader.dataset.get_class_num()
        confusion_matrix = np.zeros((number_of_class, number_of_class))
        
        for input, label in tqdm(dataloader, total = dataloader.len):
            
            prediction = self.classify_document(input,
                                                num_of_labels = number_of_class)
            
            confusion_matrix[label, prediction] + 1
        
        evaluation = None#evaluate_confusion_matrix(confusion_matrix)
        
        return evaluation
    

class Word2VecForDC(DocumnetClassificationAlgorithms):
    
    
    def __init__(
        self,
        train_df,
        categories,
        *args,
        **kwargs
    ):
        super().__init__(self, train_df, categories, *args, **kwargs)
        documents = self._get_documents(train_df)
        self.model = Word2Vec(documents, min_count=1, size=100, window=5, sg=1)
        self.classifier = LogisticRegression(random_state=42)
    
    
    def _get_document_embeddings(
        self,
        document : list
    ):
        X = []
        for word in document:
            if word in self.model.wv.vocab:
                vector.append(self.model.wv[word])
        if len(vector) > 0:
            vector = sum(vector) / len(vector)
            X.append(vector)
    
        
    def _get_documents(
        self,
        document_df
    ) -> dict :
        return {"X" : [],
                "Y" : []
                }
    
    
    def train(
        self,
        document_df : pd.DataFrame
    ):
        
        train_data = self._get_documents(document_df)
        documents = train_data["X"]
        y_train = train_data["Y"]
        
        X_train = []
        for document in documents :
            X_train.append(self._get_document_embeddings(document))

        self.classifier.fit(X_train, y_train)
        
        
    def evaluate(
        self,
        document_df : pd.DataFrame
    ):
        test_data = self._get_documents(document_df)
        documents = test_data["X"]
        y_test = test_data["Y"]
        
        X_test = []
        for document in documents :
            X_test.append(self._get_document_embeddings(document))
        
        y_pred = self.classifier.predict(X_test)
        return classification_report(y_test, y_pred)

        
   