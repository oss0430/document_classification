from modules.dataset import DCDataset
from modules.algorithms import KnnForDC, NaieveBayesForDC
import nltk
#nltk.download('punkt')

def main():
    ## get dataset
    dataset = DCDataset(path = "dataset/News_Category_Dataset_v3_balanced.json")
    dataset_for_scikit = dataset.get_dataset_for_scikitlearn()
    
    #print(dataset.df.head(5))
    #print(dataset_for_scikit["X"][0], dataset_for_scikit["Y"][0])
    #print(dataset.categories)
    
    train_dataset, test_dataset = dataset.split_train_test()
    
    #algorithms = [KnnForDC(dataset.categories)]
    algorithms = [NaieveBayesForDC(dataset.categories, n_gram = 3)]
    
    for algorithm in algorithms:
        algorithm.train(train_dataset, method = "concat")
        evaluation_result = algorithm.evaluate(test_dataset, method = "concat")
        print(evaluation_result["acc"])
        #print(evaluation_result["confusion_matrix"])

if __name__ == "__main__":
    main()