# Sample file, make changes to this file as needed.
# Example run at terminal:
# python proj1_evaluate.py --data mushroom_test.csv --model mymodel.skop
# Please provide how to use your code if you make changes to input parameters.

from sklearn.tree import (
    DecisionTreeClassifier,
)  # Replace with the algorithm of the model you have chosen.
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import argparse
import skops.io as sio
import pickle
import joblib
import numpy as np

pd.set_option("future.no_silent_downcasting", True)


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
    X = X.replace("?", np.nan)
    X.fillna(np.nan, inplace=True)
    X.drop(columns=["veil-type"], inplace=True)
    # Fill missing values with mode (most frequent value)
    mode_cols = [
        "cap-surface",
        "gill-attachment",
        "gill-spacing",
        "stem-surface",
        "veil-color",
        "ring-type",
        "spore-print-color",
    ]

    for col in mode_cols:
        if X[col].mode().empty:
            X[col] = X[col].fillna(0)
        else:
            X[col] = X[col].fillna(X[col].mode()[0])

    # Fill 'stem-root' with "unknown"
    X["stem-root"] = X["stem-root"].fillna("unknown")

    label_encoders = joblib.load("label_encoders.pkl")

    categorical_cols = [
        # "class",
        "cap-shape",
        "cap-surface",
        "cap-color",
        "does-bruise-or-bleed",
        "gill-attachment",
        "gill-spacing",
        "gill-color",
        "stem-root",
        "stem-surface",
        "stem-color",
        "veil-color",
        "has-ring",
        "ring-type",
        "habitat",
        "season",
        "spore-print-color",
    ]

    def safe_transform(series, le, default=-1):
        # Create mapping from known labels to encoded values
        mapping = {label: idx for idx, label in enumerate(le.classes_)}
        # Apply the mapping, assigning a default value for unseen labels
        return series.apply(lambda x: mapping.get(x, default))

    # print(X.info())
    # Use the pre-saved encoders to transform the test data,
    # mapping any unseen label to -1.
    for column in categorical_cols:
        if column in X.columns:
            le = label_encoders[column]
            # Convert the column to string for consistency
            X[column] = X[column].astype(str)
            # Use safe_transform instead of direct transform
            X[column] = safe_transform(X[column], le)

    # for col, le in label_encoders.items():
    #     mapping = dict(zip(le.classes_, range(len(le.classes_))))
    #     print(f"Mapping for {col}: {mapping}")

    X = X.astype({col: "int64" for col in X.select_dtypes(include=["float"]).columns})

    # print(X.info())
    # print(X.describe(include="all"))
    # Prepare your model as needed.
    print(X)
    model = load_model(model_filename)

    total_right = 0
    total_wrong = 0
    # After predicting:
    Y_pred = model.predict(X)

    # Convert numeric predictions back to original labels using the saved encoder
    le_class = label_encoders["class"]
    Y_pred_labels = le_class.inverse_transform(Y_pred)

    df["Predicted"] = Y_pred_labels

    total_right = 0
    total_wrong = 0
    for index, row in df.iterrows():
        y_target = row["class"]  # original string label (e.g., "p" or "e")
        y_pred = row["Predicted"]
        print("prediction:", y_pred, ", target:", y_target)
        if y_pred == y_target:
            total_right += 1
        else:
            total_wrong += 1

    print("correct:", total_right, ", wrong:", total_wrong)
    print("Final Accuracy is ", total_right / (total_right + total_wrong))

    # Y_pred = model.predict(X)

    # ############# Try not to change the accuracy formula ############
    # df["Predicted"] = Y_pred

    # for index, row in df.iterrows():
    #     y_target = row["class"]
    #     y_pred = row["Predicted"]
    #     print("prediction:", y_pred, ", target:", y_target)
    #     if y_pred == y_target:  # change if needed
    #         total_right = total_right + 1
    #     else:
    #         total_wrong = total_wrong + 1
    #     # print("prediction:", y_pred, ", target:", y_target, ", right:", total_right, ", wrong:", total_wrong)
    # print("correct:", total_right, ", wrong:", total_wrong)
    # print("Final Accuracy is ", total_right / (total_right + total_wrong))
