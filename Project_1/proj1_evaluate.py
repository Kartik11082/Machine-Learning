# Sample file, make changes to this file as needed.
# Example run at terminal:
# python proj1_evaluate.py --data mushroom_test.csv --model mymodel.skop
# Please provide how to use your code if you make changes to input parameters.

from sklearn.tree import (
    DecisionTreeClassifier,
)  # Replace with the algorithm of the model you have chosen.
import pandas as pd
import argparse
import skops.io as sio
import pickle
import joblib


def load_model(model_name):
    model = None
    if model_name.endswith(".skop"):
        model = sio.load(model_name)
    if model_name.endswith(".pkl") or model_name.endswith(".sav"):
        model = pickle.load(open(model_name, "rb"))
    if model_name.endswith(".joblib"):
        model = joblib.load(model_name)

    return model


if __name__ == "__main__":
    # Keep the code as it is for argument parser.
    parser = argparse.ArgumentParser(description="Train on decision tree")
    parser.add_argument("--data", required=True, help="input test data file")
    parser.add_argument("--model", required=True, help="input model file")
    args = parser.parse_args()
    test_filename = args.data
    model_filename = args.model

    df = pd.read_csv(test_filename, header=0)
    X = df.iloc[:, 1:]

    # Prepare your data as needed. Make decision on how to handle missing values
    #
    #
    # Prepare your model as needed.
    model = load_model(model_filename)

    total_right = 0
    total_wrong = 0
    Y_pred = model.predict(X)

    ############# Try not to change the accuracy formula ############
    df["Predicted"] = Y_pred

    for index, row in df.iterrows():
        y_target = row["class"]
        y_pred = row["Predicted"]
        if y_pred == y_target:  # change if needed
            total_right = total_right + 1
        else:
            total_wrong = total_wrong + 1
        # print("prediction:", y_pred, ", target:", y_target, ", right:", total_right, ", wrong:", total_wrong)
    print("correct:", total_right, ", wrong:", total_wrong)
    print("Final Accuracy is ", total_right / (total_right + total_wrong))
