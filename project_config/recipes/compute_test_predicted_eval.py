# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Read recipe inputs
test_predicted_merged = dataiku.Dataset("test_predicted_merged")
test_predicted_merged_df = test_predicted_merged.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

def evaluate_binary_classification(y_true, y_pred):
  """
  Evaluates the performance of a binary classification model.

  Args:
    y_true: True labels.
    y_pred: Predicted labels.

  Returns:
    A dictionary containing the following metrics:
      - accuracy
      - precision
      - recall
      - f1_score
      - roc_auc_score
      - confusion_matrix
  """

  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)
  roc_auc = roc_auc_score(y_true, y_pred)
  conf_matrix = confusion_matrix(y_true, y_pred)
  tn, fp, fn, tp = conf_matrix.ravel() 
  return {
      'accuracy': accuracy,
      'precision': precision,
      'recall': recall,
      'f1_score': f1,
      'roc_auc': roc_auc,
#     'confusion_matrix': conf_matrix
      'True Negatives': [tn],
      'False Positives': [fp],
      'False Negatives': [fn],
      'True Positives': [tp]
  }

# 
y_true = test_predicted_merged_df['fraud_bool']
y_pred = test_predicted_merged_df['prediction']

metrics = evaluate_binary_classification(y_true, y_pred)

#print("Accuracy:", metrics['accuracy'])
#print("Precision:", metrics['precision'])
#print("Recall:", metrics['recall'])
#print("F1-score:", metrics['f1_score'])
#print("ROC AUC:", metrics['roc_auc'])
#print("Confusion Matrix:\n", metrics['confusion_matrix'])

test_predicted_eval_df = df = pd.DataFrame.from_dict(metrics)  # For this sample code, simply copy input to output


# Write recipe outputs
test_predicted_eval = dataiku.Dataset("test_predicted_eval")
test_predicted_eval.write_with_schema(test_predicted_eval_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#print("Confusion Matrix:\n", metrics['confusion_matrix'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#metrics['confusion_matrix'].ravel()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#(211572+2956)/(82086+386+211572+2956)