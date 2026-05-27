import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import nn_core


def train_and_test(dataset_path):
    df = pd.read_csv(dataset_path)

    print("DATASET:", dataset_path)
    print(df.head())
    print(df.shape)
    print(df.iloc[:, -1].value_counts())

    X = df.iloc[:, :-1].astype(float).values.tolist()
    y = df.iloc[:, -1].astype(float).values.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = nn_core.NeuralNetwork(len(X[0]), 16)

    nn_core.train_model(
        model,
        X_train,
        y_train,
        epochs=1000,
        learning_rate=0.1
    )

    preds = [model.predict_class(row) for row in X_test]
    f1 = f1_score(y_test, preds)

    return f1


f1_d1 = train_and_test("dataset1.csv")
f1_d2 = train_and_test("dataset2.csv")


final_score = 0.5 * f1_d1 + 0.5 * f1_d2

print("F1d2", f1_d2)
print("F1d1", f1_d1)
print("Finalscore", final_score)