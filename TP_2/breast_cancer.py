import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LinearRegression

# Load the boston dataset.
data = load_breast_cancer()
X, y = data['data'], data['target']

# Idx for each set
idx_train = range(300)
idx_val = range(300, 400)
idx_test = range(400, len(X))
print(f"Train: {len(idx_train)} - Val: {len(idx_val)} - Test: {len(idx_test)}")


# Confusion matrix
def create_confusionMatrix(pred, gt):
    confusionMatrix = np.zeros((2,2)).astype(np.float32)

    # Loop over each pred/gt and update the confusion matrix
    for p, g, in zip(pred, gt):
        # TODO
        pass

    return confusionMatrix

# Compute metrics
def get_metrics(confusion_matrix):
    """ compute accuracy - precision - recall - F-measure """
    # TODO
    return None

