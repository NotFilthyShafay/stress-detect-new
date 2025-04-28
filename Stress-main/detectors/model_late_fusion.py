import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import ast  # To safely evaluate string dictionary representation
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import GridSearchCV


DATAFRAME_PATH=''

# Load the dataset
train_file_path = 'train_dataframe.csv'
val_file_path = 'val_dataframe.csv'

# print(os.listdir())




class RandomForestFusionNode:
    def __init__(self, X_train, y_train, reoptimize=False, forest_random_state=10):
        self.params=None
        if reoptimize:
            self.clf, self.params=grid_search_random_forest(X_train, y_train)
        else:
            self.clf = RandomForestClassifier(n_estimators=200, random_state=forest_random_state, max_depth=None, min_samples_leaf=2, min_samples_split=10)
        self.clf.fit(X_train, y_train)

    def make_prediction(self, path):
        pass

    def simulate_prediction(self, outputs):
        prediction = self.clf.predict(outputs)
        return prediction


class MLPFusionNode:
    def __init__(self, X_train, y_train, hidden_size=256, epochs=30, batch_size=64, learning_rate=0.01):
        input_size = X_train.shape[1]
        self.model=train_mlp_model(X_train, y_train, input_size, hidden_size, 2, epochs, batch_size, learning_rate)

    def simulate_prediction(self, outputs):
        outputs_tensor = torch.tensor(outputs.values, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(outputs_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predictions = predicted.numpy()

        return predictions

def grid_search_random_forest(X_train, y_train):
    # Define the parameter grid to search
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
    }

    # Initialize the classifier
    clf = RandomForestClassifier(random_state=10, oob_score=f1_score)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1')

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Best parameters and best score
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_}")

    return grid_search.best_estimator_, grid_search.best_params_


def read_data(df, use_prosodic=True):
    # Step 1: Extract Features
    # Convert string representations of dictionaries to actual dictionaries
    df['audio_pred'] = df['audio_pred'].apply(ast.literal_eval)
    df['face_pred'] = df['face_pred'].apply(ast.literal_eval)
    df['prosodic_pred'] = df['prosodic_pred'].apply(ast.literal_eval)

    # Initialize columns for audio and face features
    for key in df['audio_pred'].iloc[1].keys():
        df[f'audio_{key}'] = df['audio_pred'].apply(lambda x: x.get(key, 0))
    for key in df['face_pred'].iloc[0].keys():
        df[f'face_{key}'] = df['face_pred'].apply(lambda x: x.get(key, 0))

    if use_prosodic:
        for key in df['prosodic_pred'].iloc[1].keys():
            df[f'prosodic_{key}'] = df['prosodic_pred'].apply(lambda x: x.get(key, 0))

    # Drop original columns
    df.drop(['audio_pred', 'face_pred', 'prosodic_pred', 'audio_truth', 'face_truth', 'fidget_truth', 'prosodic_truth'], axis=1, inplace=True)
    try:
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
    except KeyError:
        pass

    # Step 2: Prepare the Data
    X = df.drop('ground truth', axis=1)  # Features
    y = df['ground truth']  # Target variable

    # Split data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X,y


# Random Forest Model Function
def random_forest_classifier(X_train, y_train, X_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=None, min_samples_split=10, min_samples_leaf=2)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(X_test)
    return predictions

# MLP in PyTorch
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train_mlp_model(X_train, y_train, input_size, hidden_size, num_classes=2, epochs=40,
                    batch_size=16, learning_rate=0.0001):
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

    # DataLoader
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    model = MLP(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for i, (features, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


# Common Evaluation Function
def evaluate_model(y_truth, y_pred):
    accuracy = accuracy_score(y_truth, y_pred)
    precision = precision_score(y_truth, y_pred, average='binary')
    recall = recall_score(y_truth, y_pred, average='binary')
    f1 = f1_score(y_truth, y_pred, average='binary')
    confusion = confusion_matrix(y_truth, y_pred, normalize='true')

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'confusion matrix:\n{confusion}')
    ret_dic = {'precision': precision, "recall": recall, "f1": f1, "accuracy": accuracy,
               "confusion": confusion}

    return ret_dic


def tune_MLP_hyperparams():
    train_df = pd.read_csv(os.path.join(DATAFRAME_PATH, train_file_path))
    val_df = pd.read_csv(os.path.join(DATAFRAME_PATH, val_file_path))
    # Split data into training and validation sets
    X_train, y_train = read_data(train_df)
    X_val, y_val = read_data(val_df)
    X_val_tensor=torch.tensor(X_val.values, dtype=torch.float32)

    input_size = X_train.shape[1]  # Number of features
    hidden_size = 128  # 32  # You can tweak this
    num_classes = 2  # Stressed or not stressed

    epochs_range = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    batch_size_range = [4, 8, 16, 32, 64, 128, 256]
    learning_rate_range = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    # epochs_range = [20]
    # batch_size_range = [16]
    # learning_rate_range = [1e-3]

    best_test_f1 = 0
    best_params = {}

    for epochs in epochs_range:
        for batch_size in batch_size_range:
            for learning_rate in learning_rate_range:
                print(
                    f"Training with epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}")

                # Train the model with the current set of parameters
                # mlp_predictions = train_mlp_model(X_train, y_train, X_val, y_val, input_size,
                #                                   hidden_size, num_classes, epochs, batch_size,
                #                                   learning_rate)
                new_model = train_mlp_model(X_train, y_train, input_size,
                                                  hidden_size, num_classes, epochs, batch_size,
                                                  learning_rate)

                with torch.no_grad():
                    outputs = new_model(X_val_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    mlp_predictions = predicted.numpy()

                # Evaluate on the training set (optional, for insight)
                ret_dict = evaluate_model(mlp_predictions, y_val)
                val_f1 = ret_dict["f1"]
                print(f"Val F1: {val_f1}")

                # Update best parameters if current test_f1 is better
                if val_f1 > best_test_f1:
                    best_test_f1 = val_f1
                    best_params = {
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                    }
                    print(f"New best Test F1: {best_test_f1} with parameters: {best_params}")
                    # import pdb; pdb.set_trace()

    # Output the best parameters and corresponding F1 score
    print(f"Best Val F1: {best_test_f1} with parameters: {best_params}")
    with open('best_MLP_hyperparams.txt', 'w') as outfile:
        outfile.write(str(best_params))
    print('params saved to best_MLP_hyperparams.txt')

    return best_params

def tune_random_forest_hyperparams():
    train_df = pd.read_csv(os.path.join(DATAFRAME_PATH, train_file_path))
    val_df = pd.read_csv(os.path.join(DATAFRAME_PATH, val_file_path))
    # Split data into training and validation sets
    X_train, y_train = read_data(train_df)
    X_val, y_val = read_data(val_df)

    # print("RANDOM FOREST")
    # Train and evaluate Random Forest
    # rf_predictions = random_forest_classifier(X_train, y_train, X_val)
    # evaluate_model(rf_predictions, y_val)

    # Use GridSearchCV to find the best hyperparameters for Random Forest
    print("RANDOM FOREST with Grid Search")
    best_rf_classifier, best_params = grid_search_random_forest(X_train, y_train)


    with open('best_random_forest_hyperparams.txt', 'w') as outfile:
        outfile.write(str(best_params))
    print('params saved to best_random_forest_hyperparams.txt')


    # Predict on the validation set with the best found parameters
    rf_predictions = best_rf_classifier.predict(X_val)
    evaluate_model(y_val, rf_predictions)


if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join(DATAFRAME_PATH, train_file_path))
    val_df = pd.read_csv(os.path.join(DATAFRAME_PATH, val_file_path))

    # Split data into training and validation sets
    X_train, y_train = read_data(train_df)
    X_val, y_val = read_data(val_df)

    # print("RANDOM FOREST")
    # # Train and evaluate Random Forest
    # # rf_predictions = random_forest_classifier(X_train, y_train, X_val)
    # # evaluate_model(rf_predictions, y_val)
    #
    # # Use GridSearchCV to find the best hyperparameters for Random Forest
    # print("RANDOM FOREST with Grid Search")
    # best_rf_classifier, forest_params = grid_search_random_forest(X_train, y_train)
    # print(forest_params)
    #
    # # Predict on the validation set with the best found parameters
    # rf_predictions = best_rf_classifier.predict(X_val)
    # evaluate_model(y_val, rf_predictions)

    # print("-----")
    # print("MLP")
    # Train and evaluate MLP



    print('TUNING RANDOM FOREST\n-----')
    tune_random_forest_hyperparams()
    print('TUNING MLP\n-----')
    print(tune_MLP_hyperparams())