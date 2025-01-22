# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
test_prepared_to_predict_scored = dataiku.Dataset("test_prepared_to_predict_scored")
test_prepared_to_predict_scored_df = test_prepared_to_predict_scored.get_dataframe()
test = dataiku.Dataset("test")
test_df = test.get_dataframe()


# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.
col_test_prepared_to_predict_scored_df= test_prepared_to_predict_scored_df[['proba_0','proba_1', 'prediction']]
test_predicted_merged_df = pd.concat([test_df, col_test_prepared_to_predict_scored_df], axis=1) 
# Compute a Pandas dataframe to write into test_predicted_merged

# Write recipe outputs
test_predicted_merged = dataiku.Dataset("test_predicted_merged")
test_predicted_merged.write_with_schema(test_predicted_merged_df)
