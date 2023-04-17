from modules.dataset import DCDataset
from modules.algorithms import KnnForDC, NaieveBayesForDC, Word2VecForDC
from modules.evaluation import evaluate_confusion_matrix
import nltk
import json
#nltk.download('punkt')

def _result_to_file_json_file(
    path : str,
    result : dict
) :
    evaluation_result = evaluate_confusion_matrix(result["confusion_matrix"])
    evaluation_result["acc"] = result["acc"]

    with open(path, "w") as f:
        json.dump(evaluation_result, f)

    

def main():
    ## get dataset
    dataset = DCDataset(path = "dataset/News_Category_Dataset_v3_balanced.json")
    dataset_for_scikit = dataset.get_dataset_for_scikitlearn()
    
    train_dataset, test_dataset = dataset.split_train_test()
    
    #algorithms = [KnnForDC(dataset.categories)]
    #algorithms = [NaieveBayesForDC(dataset.categories, n_gram = 3)]
    algorithms = [Word2VecForDC(dataset.categories)]
    #algorithms = [KnnForDc(dataset.categories),
    #              NaieveBayesForDC(dataset.categories, n_gram = 1), 
    #              NaieveBayesForDC(dataset.categories, n_gram = 2),
    #              NaieveBayesForDC(dataset.categories, n_gram = 3),
    #              Word2VecForDC(dataset.categories)]
    
    for algorithm in algorithms:
        algorithm.train(train_dataset, method = "concat")
        evaluation_result = algorithm.evaluate(test_dataset, method = "concat")
        print(evaluation_result["acc"])
        #print(evaluation_result["confusion_matrix"])

if __name__ == "__main__":
    main()