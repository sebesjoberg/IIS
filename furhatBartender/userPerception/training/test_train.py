import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import numpy as np

def train_and_eval(model, train_in, train_out, val_in, val_out):
    model.fit(train_in, train_out)
    predicted_val = model.predict(val_in)
    # Evaluate model
    return accuracy_score(val_out, predicted_val)

def main():

    #Read data
    data = pd.read_csv("C:/Users/adams/OneDrive/Dokument/Skola/InteractiveSystems/Project/IIS/furhatBartender/userPerception/training/dataset.csv")
    # submit_data = pd.read_csv("test_to_submit.csv")

    labels = data["emotion"]
    inputs = data.drop("emotion", axis=1)

    #We implement a 70/20/10 split of the data
    data_in, test_in, data_out, test_out = train_test_split(inputs, labels, test_size=0.15, stratify=labels, random_state=42)
    train_in, val_in, train_out, val_out = train_test_split(data_in, data_out, test_size=(0.2 / 0.85), stratify=data_out, random_state=42)

    print("Length of input: ", len(inputs))
    print("Splitted data distribution: ", len(train_in), len(val_in), len(test_in))
    print("Sum of distribution: " ,len(train_in)+ len(val_in)+ len(test_in))

    results = {}

    #Test different models and try different parameters for them: 

    # SVC()
    param_grid_1 = [
        {
            "kernel": ["rbf", "linear", "poly", "sigmoid"]
    }
    ]
    model1 = GridSearchCV(SVC(), param_grid=param_grid_1)
    eval1 = train_and_eval(model1, train_in, train_out, val_in, val_out)
    print(
            "\nAccuracy of SVC: ",
            eval1*100, "%",
            "\nBest parameter for SVC: ",
            model1.best_params_["kernel"]
        )    
    results["SVC"] = [eval1, model1.best_params_]

    # #K neighbors
    param_grid_2 = [
            {
                "n_neighbors": list(range(1,40))
        }
        ]
    
    model2 = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid_2)
    eval2 = train_and_eval(model2, train_in, train_out, val_in, val_out)
    print(
            "\nAccuracy of KNeighbors: ",
            eval2*100, "%",
            "\nBest parameter for KNeighbors: ",
            model2.best_params_["n_neighbors"]
        )    
    results["KNeighborsClassifier"] = [eval2, model2.best_params_]
        
    # #Multi Layer Perceptron
    param_grid_3 = [
        {
            "solver": ['lbfgs', 'sgd', 'adam']
        }
    ]

    model3 = GridSearchCV(MLPClassifier(max_iter=100), param_grid_3)
    eval3 = train_and_eval(model3, train_in, train_out, val_in, val_out)
    print(
                "\nAccuracy of MLP: ",
                eval3*100, "%",
                "\nBest parameter for MLP: ",
                model3.best_params_["solver"]
            )  
    results["MLPClassifier"] = [eval3, model3.best_params_]

    #Gaussian Process Classifier

    param_grid_4 = [
        {
            "multi_class": ["one_vs_rest", "one_vs_one"]
        }
    ]

    model4 = GridSearchCV(GaussianProcessClassifier(), param_grid_4)
    eval4 = train_and_eval(model4, train_in, train_out, val_in, val_out)
    print(
                "\nAccuracy of GPC: ",
                eval4*100, "%",
                "\nBest parameter for GPC: ",
                model4.best_params_["multi_class"]
            )  
    results["GaussianProcessClassifier"] = [eval4, model4.best_params_]

    max_key = max(results, key=lambda k: results[k][0])
    max_value = results[max_key]

    print(f"The highest accuracy is associated with {max_key} with a value of {max_value[0]} and the best parameter is {max_value[1]}")


    #Final prediction
    # def write_ndarray_to_file(data, filename):
    #     with open(filename, 'w') as file:
    #         for element in data.flatten():
    #             file.write(str(element) + '\n')

    # final_model = SVC(kernel="linear")
    # final_model.fit(train_in, train_out)
    # pred_fin = final_model.predict(submit_data)

    # write_ndarray_to_file(pred_fin, "outputs")


if __name__ == "__main__":
    main()