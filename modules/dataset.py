import os
import torch
import numpy as np
import pandas as pd
import transformers


class DCDataset():


    def __init__(
        self,
        path : str = None,
        title_column_name : str = "headline",
        description_column_name : str = "short_description",
        label_column_name : str = "category",
        df : pd.DataFrame = None,
        categories : np.ndarray = None
    ) :
        
        if path:
            file_format = os.path.splitext(path)[1]
            if file_format == '.csv':
                df = pd.read_csv(path, usecols=[title_column_name, description_column_name, label_column_name])
            elif file_format == '.excel':
                df = pd.read_excel(path, usecols=[title_column_name, description_column_name, label_column_name])
            elif file_format == '.json':
                df = pd.read_json(path, lines= True)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
        self.df = df
        
        if categories is None :
            categories = df[label_column_name].unique()
        self.categories = categories
        
        self.title_column_name = title_column_name
        self.description_column_name = description_column_name
        self.label_column_name = label_column_name
    
    
    def get_dataset_for_scikitlearn(
        self,
        method : str = 'concat'
    ) -> dict :
        
        X = []
        Y = []
        
        if method == 'concat' :
            title_list = self.df[self.title_column_name].tolist()
            description_list = self.df[self.description_column_name].tolist()
            label_list = self.df[self.label_column_name].tolist()
            
            for title, description, label in zip(title_list, description_list, label_list) :
                concated_str = title + " " + description
                X.append(concated_str)
                Y.append(label)
                
        elif method == 'only_title' :
            title_list = self.df[self.title_column_name].tolist()
            label_list = self.df[self.label_column_name].tolist()
            
            for title, label in zip(title_list, label_list) :
                X.append(title)
                Y.append(label)
            
        elif method == 'only_description' :
            title_list = self.df[self.title_column_name].tolist()
            description_list = self.df[self.label_column_name].tolist()
            
            for description, label in zip(description_list, label_list) :
                X.append(description)
                Y.append(label)
                
        return {"X" : X,
                "Y" : Y   
                }
    
    
    def split_train_test(
        self,
        train_ratio : float = 0.9
    ) -> tuple:
        train_ratio = train_ratio
        
        dftrain = self.df.sample(frac = train_ratio, random_state=200)
        dftest  = self.df.drop(dftrain.index)
        
        train_dataset = DCDataset(title_column_name = self.title_column_name,
                                  description_column_name = self.description_column_name,
                                  label_column_name = self.label_column_name,
                                  df = dftrain,
                                  categories = self.categories)
        
        test_dataset  = DCDataset(title_column_name = self.title_column_name,
                                  description_column_name = self.description_column_name,
                                  label_column_name = self.label_column_name,
                                  df = dftest,
                                  categories = self.categories)
        
        return (train_dataset, test_dataset)
    
    
    def get_by_index(
        self,
        index : int 
    ):
        return
            

