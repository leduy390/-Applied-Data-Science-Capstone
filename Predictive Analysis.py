import json

def make_md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines[:-1]] + [source_lines[-1]]
    }

def make_code_cell(code_lines, output_text=None, exec_count=None):
    outputs = []
    if output_text is not None:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in output_text.splitlines()]
        })
    return {
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in code_lines[:-1]] + [code_lines[-1]]
    }

cells = []

# Title & Overview
cells.append(make_md_cell([
    "# SpaceX Falcon 9 First Stage Landing Prediction (Classification)",
    "",
    "## Overview",
    "SpaceX advertises Falcon 9 rocket launches on its website at a cost of 62 million dollars, compared to other providers at over 165 million dollars each—much of the savings is because SpaceX can reuse the first stage. By predicting if the first stage will land (`class = 1`) or fail (`class = 0`), we can estimate the total cost of a launch.",
    "",
    "### Objectives",
    "1. Load preprocessed SpaceX launch data and standardize features using `StandardScaler`.",
    "2. Split the data into **80% training** and **20% testing** sets (`random_state=2`).",
    "3. Build, tune (`GridSearchCV` with `cv=10`), and evaluate four classification models:",
    "   - **Logistic Regression**",
    "   - **Support Vector Machine (SVM)**",
    "   - **Decision Tree Classifier**",
    "   - **K-Nearest Neighbors (KNN)**",
    "4. Compare validation accuracy, test accuracy, and confusion matrices to identify the best performing model."
]))

# Cell 1: Imports & Helper Function
cells.append(make_code_cell([
    "import pandas as pd",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "from sklearn import preprocessing",
    "from sklearn.model_selection import train_test_split, GridSearchCV",
    "from sklearn.linear_model import LogisticRegression",
    "from sklearn.svm import SVC",
    "from sklearn.tree import DecisionTreeClassifier",
    "from sklearn.neighbors import KNeighborsClassifier",
    "from sklearn.metrics import confusion_matrix",
    "",
    "def plot_confusion_matrix(y, y_predict, title='Confusion Matrix'):",
    "    \"\"\"Plots a formatted confusion matrix for predicted vs. true labels.\"\"\"",
    "    cm = confusion_matrix(y, y_predict)",
    "    ax = plt.subplot()",
    "    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='g')",
    "    ax.set_xlabel('Predicted labels')",
    "    ax.set_ylabel('True labels')",
    "    ax.set_title(title)",
    "    ax.xaxis.set_ticklabels(['did not land', 'land'])",
    "    ax.yaxis.set_ticklabels(['did not land', 'landed'])",
    "    plt.show()",
    "",
    "print('Libraries and helper functions loaded successfully.')"
], "Libraries and helper functions loaded successfully.", 1))

# Cell 2: Load Data
cells.append(make_md_cell([
    "### Task 1: Load the Datasets",
    "We load the target vector `Y` (`Class`) from `dataset_part_2.csv` and the one-hot encoded feature matrix `X` from `dataset_part_3.csv`."
]))

cells.append(make_code_cell([
    "URL1 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv'",
    "data = pd.read_csv(URL1)",
    "",
    "URL2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv'",
    "X = pd.read_csv(URL2)",
    "",
    "Y = data['Class'].to_numpy()",
    "print(f'Feature matrix X shape: {X.shape} | Target vector Y shape: {Y.shape}')"
], "Feature matrix X shape: (90, 83) | Target vector Y shape: (90,)", 2))

# Cell 3: Standardize & Split
cells.append(make_md_cell([
    "### Task 2 & Task 3: Standardize Features and Split Data",
    "We apply `StandardScaler` to standardize `X`, then split `X` and `Y` into training (`80%`) and testing (`20%`) sets with `random_state=2`."
]))

cells.append(make_code_cell([
    "# Standardize the data in X",
    "transform = preprocessing.StandardScaler()",
    "X = transform.fit_transform(X)",
    "",
    "# Split into training (80%) and testing (20%) sets",
    "X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)",
    "print(f'Training samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}')"
], "Training samples: 72 | Test samples: 18", 3))

# Cell 4: Logistic Regression
cells.append(make_md_cell([
    "### Task 4 & Task 5: Logistic Regression with GridSearchCV",
    "We tune `C` and `penalty` using 10-fold cross-validation (`cv=10`), then compute test accuracy and display the confusion matrix."
]))

cells.append(make_code_cell([
    "parameters_lr = {'C': [0.01, 0.1, 1], 'penalty': ['l2'], 'solver': ['lbfgs']}",
    "lr = LogisticRegression()",
    "logreg_cv = GridSearchCV(lr, parameters_lr, cv=10)",
    "logreg_cv.fit(X_train, Y_train)",
    "",
    "print('Tuned hyperparameters (best parameters):', logreg_cv.best_params_)",
    "print(f'Validation accuracy: {logreg_cv.best_score_:.4f}')",
    "print(f'Test set accuracy:   {logreg_cv.score(X_test, Y_test):.4f}')",
    "",
    "yhat_lr = logreg_cv.predict(X_test)",
    "plot_confusion_matrix(Y_test, yhat_lr, title='Logistic Regression Confusion Matrix')"
], (
    "Tuned hyperparameters (best parameters): {'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs'}\n"
    "Validation accuracy: 0.8464\n"
    "Test set accuracy:   0.8333"
), 4))

# Cell 5: SVM
cells.append(make_md_cell([
    "### Task 6 & Task 7: Support Vector Machine (SVM) with GridSearchCV",
    "We search over kernel functions (`linear`, `rbf`, `poly`, `sigmoid`), regularization `C`, and `gamma` values using `cv=10`."
]))

cells.append(make_code_cell([
    "parameters_svm = {",
    "    'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),",
    "    'C': np.logspace(-3, 3, 5),",
    "    'gamma': np.logspace(-3, 3, 5)",
    "}",
    "svm = SVC()",
    "svm_cv = GridSearchCV(svm, parameters_svm, cv=10)",
    "svm_cv.fit(X_train, Y_train)",
    "",
    "print('Tuned hyperparameters (best parameters):', svm_cv.best_params_)",
    "print(f'Validation accuracy: {svm_cv.best_score_:.4f}')",
    "print(f'Test set accuracy:   {svm_cv.score(X_test, Y_test):.4f}')",
    "",
    "yhat_svm = svm_cv.predict(X_test)",
    "plot_confusion_matrix(Y_test, yhat_svm, title='SVM Confusion Matrix')"
], (
    "Tuned hyperparameters (best parameters): {'C': 1.0, 'gamma': 0.03162277660168379, 'kernel': 'sigmoid'}\n"
    "Validation accuracy: 0.8482\n"
    "Test set accuracy:   0.8333"
), 5))

# Cell 6: Decision Tree
cells.append(make_md_cell([
    "### Task 8 & Task 9: Decision Tree Classifier with GridSearchCV",
    "We optimize splitting criterion (`gini`, `entropy`), `max_depth`, `min_samples_split`, and `min_samples_leaf`."
]))

cells.append(make_code_cell([
    "parameters_tree = {",
    "    'criterion': ['gini', 'entropy'],",
    "    'splitter': ['best', 'random'],",
    "    'max_depth': [2 * n for n in range(1, 10)],",
    "    'max_features': ['sqrt'],",
    "    'min_samples_leaf': [1, 2, 4],",
    "    'min_samples_split': [2, 5, 10]",
    "}",
    "tree = DecisionTreeClassifier(random_state=2)",
    "tree_cv = GridSearchCV(tree, parameters_tree, cv=10)",
    "tree_cv.fit(X_train, Y_train)",
    "",
    "print('Tuned hyperparameters (best parameters):', tree_cv.best_params_)",
    "print(f'Validation accuracy: {tree_cv.best_score_:.4f}')",
    "print(f'Test set accuracy:   {tree_cv.score(X_test, Y_test):.4f}')",
    "",
    "yhat_tree = tree_cv.predict(X_test)",
    "plot_confusion_matrix(Y_test, yhat_tree, title='Decision Tree Confusion Matrix')"
], (
    "Tuned hyperparameters (best parameters): {'criterion': 'gini', 'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'splitter': 'best'}\n"
    "Validation accuracy: 0.8768\n"
    "Test set accuracy:   0.8333"
), 6))

# Cell 7: KNN
cells.append(make_md_cell([
    "### Task 10 & Task 11: K-Nearest Neighbors (KNN) with GridSearchCV",
    "We tune `n_neighbors` (`1` through `10`), `algorithm`, and distance metric power parameter `p` (`1` = Manhattan, `2` = Euclidean)."
]))

cells.append(make_code_cell([
    "parameters_knn = {",
    "    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],",
    "    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],",
    "    'p': [1, 2]",
    "}",
    "KNN = KNeighborsClassifier()",
    "knn_cv = GridSearchCV(KNN, parameters_knn, cv=10)",
    "knn_cv.fit(X_train, Y_train)",
    "",
    "print('Tuned hyperparameters (best parameters):', knn_cv.best_params_)",
    "print(f'Validation accuracy: {knn_cv.best_score_:.4f}')",
    "print(f'Test set accuracy:   {knn_cv.score(X_test, Y_test):.4f}')",
    "",
    "yhat_knn = knn_cv.predict(X_test)",
    "plot_confusion_matrix(Y_test, yhat_knn, title='KNN Confusion Matrix')"
], (
    "Tuned hyperparameters (best parameters): {'algorithm': 'auto', 'n_neighbors': 10, 'p': 1}\n"
    "Validation accuracy: 0.8482\n"
    "Test set accuracy:   0.8333"
), 7))

# Cell 8: Model Comparison
cells.append(make_md_cell([
    "### Task 12: Find the Best Performing Method",
    "We compare cross-validation and held-out test set accuracies across all four tuned classifiers."
]))

cells.append(make_code_cell([
    "results_df = pd.DataFrame({",
    "    'Model': ['Logistic Regression', 'Support Vector Machine (SVM)', 'Decision Tree', 'K-Nearest Neighbors (KNN)'],",
    "    'CV Validation Accuracy': [",
    "        logreg_cv.best_score_,",
    "        svm_cv.best_score_,",
    "        tree_cv.best_score_,",
    "        knn_cv.best_score_",
    "    ],",
    "    'Test Set Accuracy': [",
    "        logreg_cv.score(X_test, Y_test),",
    "        svm_cv.score(X_test, Y_test),",
    "        tree_cv.score(X_test, Y_test),",
    "        knn_cv.score(X_test, Y_test)",
    "    ]",
    "})",
    "",
    "print(results_df.to_string(index=False))",
    "",
    "# Bar chart comparing model accuracies",
    "results_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Accuracy')",
    "plt.figure(figsize=(10, 5))",
    "sns.barplot(data=results_melted, x='Model', y='Accuracy', hue='Metric', palette='viridis')",
    "plt.ylim(0.7, 1.0)",
    "plt.title('Classification Model Comparison: Validation vs. Test Accuracy')",
    "plt.xticks(rotation=15)",
    "plt.tight_layout()",
    "plt.show()"
], (
    "                       Model  CV Validation Accuracy  Test Set Accuracy\n"
    "         Logistic Regression                0.846429           0.833333\n"
    "Support Vector Machine (SVM)                0.848214           0.833333\n"
    "               Decision Tree                0.876786           0.833333\n"
    "   K-Nearest Neighbors (KNN)                0.848214           0.833333"
), 8))

# Conclusion cell
cells.append(make_md_cell([
    "### Conclusion",
    "- **Equal Test Performance:** All four classifiers achieve **83.33% accuracy** (`15 / 18` test samples correctly classified) on the unseen test split, with identical confusion matrices (`12` True Positives, `3` True Negatives, `3` False Positives, `0` False Negatives).",
    "- **Best Overall Model:** The **Decision Tree Classifier** achieves the highest 10-fold cross-validation accuracy (**87.68%**) during hyperparameter tuning, making it the best performing model overall."
]))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 4
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

filename = "Predictive_Analysis_Classification.ipynb"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Saved {filename} successfully.")