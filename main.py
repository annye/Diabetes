import numpy as np
import pandas as pd
import os
import sys
from process import get_data
import process as p
import classification as c 
import anova as a 
import keras_net as kn 


cols = ["admission_description", "discharge_description", "admission_type_description", "race", "gender", "metformin","repaglinide",
        "nateglinide", "chlorpropamide", "glimepiride","acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone", 'A1Cresult', 'max_glu_serum',
        'change', 'diabetesMed']

def set_trace():
    """A Poor mans break point"""
    # without this in iPython debugger can generate strange characters.
    from IPython.core.debugger import Pdb
    Pdb().set_trace(sys._getframe().f_back)


def main():
   data = p.mapping()
   data, med = p.one_hot_dataframe(data, cols, replace=True)
   X_train, y_train, X_test, y_test = p.subset_and_split(data)
   kn.nn_sequential(X_train, y_train, X_test, y_test)
  # c.logistic_regression(X_train, y_train, X_test, y_test)
   #a.anova_svc(X_train, y_train, X_test, y_test)
   set_trace()


if __name__ == "__main__":
    main() 

