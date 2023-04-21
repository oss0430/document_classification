
import pandas as pd
import re
import tqdm
import numpy as np
import nltk
import copy
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
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
    ) -> None :
        
        train_data = dataset.get_dataset_for_scikitlearn(method)

        X_train = self.tfidf.fit_transform(train_data["X"])
        y_train = train_data["Y"]

        self.classifier.fit(X_train, y_train)
        
    
    def predict(
        self,
        document :str
    ) :
        tfidf = self.tfidf.transform([document])
        prediction = self.classifier.predict(tfidf)
        
        return prediction
    
    
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

    
    def inference(
        self,
        documents : list
    ) -> list :
        
        results = []
        
        for document in documents:
            results.append(self.predict(document))
            
        return results
    

class NaieveBayesForDC(DocumnetClassificationAlgorithms):
    
    
    def __init__(
        self,
        categories,
        n_gram : int = 1,
        use_laplace_smoothing : bool = False,
        *args,
        **kwargs
    ):
        super().__init__(self, categories, *args, **kwargs)
        self.n_gram = n_gram
        self.use_laplace_smoothing = use_laplace_smoothing
        
        default_value = np.zeros(len(categories))
        self.n_gram_table = defaultdict(lambda: default_value)
        self.categories = categories
    
    
    def get_n_gram_list(
        self,
        document : str
    ) -> list:
        
        words = nltk.word_tokenize(document.lower())
        n_grams = nltk.ngrams(words, self.n_gram)
        
        return n_grams
    
    
    def get_label_one_hot_encoding(
        self,
        label : str
    ) -> np.ndarray :
        
        num_labels = len(self.categories)
        onehot_matrix = np.eye(num_labels)
        
        index = np.where(self.categories == label)[0]

        onehot_label = onehot_matrix[index]
        onehot_label = onehot_label.flatten()

        return onehot_label
    
    
    def _get_prior(
        self,
        frequency_table : dict
    ) -> np.ndarray :
        
        total_frequency = np.zeros(len(self.categories))
        for word, frequency_array in frequency_table.items():
            total_frequency += frequency_array
        
        prior_probabilities = total_frequency / np.sum(total_frequency)
        
        return prior_probabilities
         

    def _get_likelihood(
        self,
        frequency_table : dict
    ) -> dict :

        ##            word frequency on that class  + 1
        ##  ------------------------------------------------------------
        ##   Total Vocaburary Count + Vocaburarys that apperas in the class

        total_frequency = np.zeros(len(self.categories))
        for word, frequency_array in frequency_table.items():
            total_frequency += frequency_array
        
        unique_vocaburary_counts = np.zeros(len(self.categories))
        for word, frequency_array in frequency_table.items():
            unique_vocaburary_counts += (frequency_array >= 1).astype(int)

        likelihoods = {}
        for word, frequency_array in frequency_table.items():
            likelihoods[word] = (frequency_array + 1) / (unique_vocaburary_counts + total_frequency)

        return likelihoods
            
    
    def train(
        self,
        dataset,
        method
    ) -> None:
        train_data = dataset.get_dataset_for_scikitlearn(method)
        X_train = train_data["X"]
        y_train = train_data["Y"]
        
        n_gram_frequency_table = defaultdict(lambda: np.zeros(len(self.categories)))
        ## Build Frequency Table
        for X, y in zip(X_train, y_train):
            ## split string to n_gram
            n_gram_list = self.get_n_gram_list(X)
            one_hot_encoded_label = self.get_label_one_hot_encoding(y)
            
            ## Add to n_gram_table
            for n_gram in n_gram_list :
                n_gram_frequency_table[n_gram] += one_hot_encoded_label

        self.n_gram_frequency_table = n_gram_frequency_table
        self.likelihoods = self._get_likelihood(copy.deepcopy(n_gram_frequency_table))
        self.prior = self._get_prior(copy.deepcopy(n_gram_frequency_table))
        #print("likelihoods and prior")
        #print(self.likelihoods)
        #print(self.prior)
        

    def evaluate(
        self,
        dataset,
        method
    ) -> dict:

        test_data = dataset.get_dataset_for_scikitlearn(method)
        X_test = test_data["X"]
        y_test = test_data["Y"]
        y_pred = []
        
        for X in X_test:
            y_pred.append(self.classify_document(X))
        
        return {"acc" : accuracy_score(y_test, y_pred),
                "confusion_matrix" : confusion_matrix(y_test, y_pred)
                }   
    
    
    def classify_document(
        self,
        document
    ) -> int :
 
        n_grams = self.get_n_gram_list(document)
        logit = np.zeros(len(self.categories))
        for n_gram in n_grams :
            try :
                logit += self.likelihoods[n_gram]
            except :
                ## Newly Appeared Dropping it
                None
        logit = self.prior * logit
        #print(logit)
        highest_index = logit.argmax()
        
        return self.categories[highest_index]
    

    def inference(
        self,
        documents : list
    ) -> list :
        
        results = []
        
        for document in documents:
            results.append(self.classify_document(document))
            
        return results
    
    
    
class Word2VecForDC(DocumnetClassificationAlgorithms):
    
    
    def __init__(
        self,
        categories,
        *args,
        **kwargs
    ):
        super().__init__(self, categories, *args, **kwargs)
        self.embedding_size = 100
        self.model = Word2Vec(min_count=1, vector_size = self.embedding_size, window=5, sg=1)
        self.classifier = RandomForestClassifier()
    
    
    def _get_document_embedding(
        self,
        document : str
    ):
        tokenized_document = nltk.tokenize.word_tokenize(document.lower())
        
        vector = []
        vocabulary = list(self.model.wv.key_to_index.keys())
        for word in tokenized_document:
            if word in vocabulary:
                vector.append(self.model.wv[word])

        if len(vector) > 0:
            documment_embedding = sum(vector) / len(vector)
        else :
            documment_embedding = np.zeros(self.embedding_size)
        return documment_embedding
    
    
    def transform_list_of_documents_to_tokenized_version(
        self,
        documents : list
    ) -> list :
        
        tokenized_documents = []
        
        for document in documents:
            tokenized_document = nltk.tokenize.word_tokenize(document.lower())
            tokenized_documents.append(tokenized_document)    
        
        return tokenized_documents 
    
    
    def train(
        self,
        dataset,
        method
    ):
        
        train_data = dataset.get_dataset_for_scikitlearn(method)
        documents = train_data["X"]
        tokenized_documents = self.transform_list_of_documents_to_tokenized_version(documents)
        y_train = train_data["Y"]
        
        ## train word embeddings
        self.model = Word2Vec(tokenized_documents, min_count=1, vector_size = self.embedding_size, window=5, sg=1)
        
        X_train = []
        for document in documents :
            X_train.append(self._get_document_embedding(document))

        self.classifier.fit(X_train, y_train)
        
    
    def predict(
        self,
        document : str
    ) :
        document_embedding = self._get_document_embedding(document)
        prediction = self.classifier.predict([document_embedding])
        
        return prediction
        
        
    def evaluate(
        self,
        dataset,
        method
    ):
        test_data = dataset.get_dataset_for_scikitlearn(method)
        documents = test_data["X"]
        tokenized_documents = self.transform_list_of_documents_to_tokenized_version(documents)
        y_test = test_data["Y"]
        
        X_test = []
        for document in documents :
            X_test.append(self._get_document_embedding(document))

        y_pred = self.classifier.predict(X_test)
        
        return {"acc" : accuracy_score(y_test, y_pred),
                "confusion_matrix" : confusion_matrix(y_test, y_pred)
                }  

    
    def inference(
        self,
        documents : list
    ) -> list :
        
        results = []
        
        for document in documents:
            results.append(self.predict(document))
            
        return results
   