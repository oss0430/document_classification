from modules.dataset import DCDataset
from modules.algorithms import KnnForDC, NaieveBayesForDC, Word2VecForDC
from modules.evaluation import evaluate_confusion_matrix
import nltk
import json
import pandas as pd
#nltk.download('punkt')

def result_to_file_json_file(
    path : str,
    result : dict
) -> dict :
    evaluation_result = evaluate_confusion_matrix(result["confusion_matrix"])
    evaluation_result["acc"] = result["acc"]

    with open(path, "w") as f:
        json.dump(evaluation_result, f)

    return evaluation_result


def qulatative_to_json_file(
    path : str,
    X : list,
    y : list,
    y_pred : list
) -> pd.DataFrame :

    qulatative_result = pd.DataFrame({'document' : X,
                                      'label' : y,
                                      'prediction' : y_pred})
    
    qulatative_result.to_json(path)
    
    return qulatative_result


def main():
    ## get dataset
    dataset = DCDataset(path = "dataset/News_Category_Dataset_v3_balanced.json")

    train_dataset, test_dataset = dataset.split_train_test()
    qulatative = test_dataset.get_dataset_for_scikitlearn()
    qulatative = {"X" : qulatative["X"][:10],
                  "Y" : qulatative["Y"][:10]}
    
    print(dataset.df["category"].value_counts(sort=False).reindex(dataset.categories))
    #print(qulatative)
    
    algorithms = [KnnForDC(dataset.categories),
                  NaieveBayesForDC(dataset.categories, n_gram = 1), 
                  NaieveBayesForDC(dataset.categories, n_gram = 2),
                  NaieveBayesForDC(dataset.categories, n_gram = 3),
                  NaieveBayesForDC(dataset.categories, n_gram = 4),
                  Word2VecForDC(dataset.categories)
                  ]
    
    for index, algorithm in enumerate(algorithms):
        print("Training algorithm #", index)
        algorithm.train(train_dataset, method = "concat")
        print("Evaluating algorithm #", index)
        evaluation_result = algorithm.evaluate(test_dataset, method = "concat")
        print(evaluation_result["confusion_matrix"])
        qulatative_prediction = algorithm.inference(qulatative['X'])
        
        evaluation_result = result_to_file_json_file("algorithm"+str(index)+"_result.json", evaluation_result)
        print(evaluation_result)
        qulatative_result = qulatative_to_json_file("algorithm"+str(index)+"_qulatative.json", qulatative['X'], qulatative['Y'], qulatative_prediction) 
        print(qulatative_result)

if __name__ == "__main__":
    main()