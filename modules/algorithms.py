
import pandas as pd
import re
import tqdm
import numpy as np
import nltk
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
    ) -> None :
        
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
        self.n_gram_map = defaultdict(int)
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
    
    
    def train(
        self,
        dataset,
        method
    ) -> None:
        train_data = dataset.get_dataset_for_scikitlearn(method)
        X_train = train_data["X"]
        y_train = train_data["Y"]
        
        for X, y in zip(X_train, y_train):
            ## split string to n_gram
            n_gram_list = self.get_n_gram_list(X)
            one_hot_encoded_label = self.get_label_one_hot_encoding(y)
            
            ## Add to n_gram_table
            for n_gram in n_gram_list :
                self.n_gram_table[n_gram] += one_hot_encoded_label
                
   
    def evaluate(
        self,
        dataset,
        method
    ) -> dict:
        
        ## for testing :
        """
        count = 0
        last_num = 20
        for key, value in self.n_gram_table.items():
            print(key, value)
            count += 1
            if count == last_num :
                break
        """

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
            logit += self.n_gram_table[n_gram]
            
        highest_index = logit.argmax()
        
        return self.categories[highest_index]
    

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

        
   