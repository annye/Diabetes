# Importing Libraries
from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#%pylab inline
from scipy.stats import spearmanr
from pylab import rcParams
import seaborn as sn
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn import feature_extraction
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import os
import sys


def set_trace():
    """A Poor mans break point"""
    # without this in iPython debugger can generate strange characters.
    from IPython.core.debugger import Pdb
    Pdb().set_trace(sys._getframe().f_back)


# Read in data
data = pd.read_csv('diabetic_data.csv')
discharge = pd.ExcelFile('dataset_diabetes/discharge.xlsx')
discharge = discharge.parse('Data')
admission = pd.ExcelFile('dataset_diabetes/admission.xlsx')
admission = admission.parse('Data')


data.replace('?', np.nan, inplace=True)
data.drop(['weight', 'medical_specialty', 'payer_code'], axis=1, inplace=True)
data.drop(columns=['encounter_id', 'patient_nbr'], inplace=True)

# Map discharge and admission id descriptions
discharge_map = dict(zip(discharge['discharge_disposition_id'], discharge['description']))
data['discharge_description'] = data['discharge_disposition_id'].map(discharge_map)
admission_map = dict(zip(admission['admission_source_id'], admission['description']))
data['admission_description'] = data['admission_source_id'].map(admission_map)
# bring in admission type id map 


def one_hot_dataframe(data, cols, replace=False):
    """Create one-hot encodings."""
    vec = feature_extraction.DictVectorizer()

    def mkdict(row): return dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(
        data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData)


# add addmission_type_description

data, med = one_hot_dataframe(data, ["admission_description",
                                     "discharge_description",
                                     "race",
                                     "gender",
                                     "metformin",
                                     "repaglinide",
                                     "nateglinide",
                                     "chlorpropamide",
                                     "glimepiride",
                                     "acetohexamide",
                                     "glipizide",
                                     "glyburide",
                                     "tolbutamide",
                                     "pioglitazone",
                                     "rosiglitazone",
                                     "acarbose",
                                     "miglitol",
                                     "troglitazone",
                                     "tolazamide",
                                     "examide",
                                     "citoglipton",
                                     "insulin",
                                     "glyburide-metformin",
                                     "glipizide-metformin",
                                     "glimepiride-pioglitazone",
                                     "metformin-rosiglitazone",
                                     "metformin-pioglitazone",
                                     "readmitted"], replace=True)


# Mapping Age
age_ranges = list(set(data['age']))
age_ordinal = [1, 8, 7, 6, 0, 9, 4, 2, 5, 3]
age_map = dict(zip(age_ranges, age_ordinal))
data['Age_Level'] = data['age'].map(age_map)
# Mapping A1Cresult
A1Cresult_ranges = list(set(data['A1Cresult']))
A1Cresult_ordinal = [0, 5, 7, 8, ]
A1Cresult_map = dict(zip(A1Cresult_ranges, A1Cresult_ordinal))
data['A1Cresult'] = data['A1Cresult'].map(A1Cresult_map)

set_trace()


##########################

X = data[A1Cresult]
y = data[readmitted]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

set_trace()