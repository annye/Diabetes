'''

Predictive Patient Re-admission Model

Date: 22/03/18

Author: Annye Braca

Email: annyebraca@gmail.com

'''


import numpy as np
import pandas as pd
import os
import sys 
from scipy.stats import spearmanr
from pylab import rcParams
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn import feature_extraction
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import model_selection


def set_trace():
    """A Poor mans break point"""
    # without this in iPython debugger can generate strange characters.
    from IPython.core.debugger import Pdb
    Pdb().set_trace(sys._getframe().f_back)


def get_data():
    data = pd.read_csv('../static/diabetic_data.csv')
    discharge = pd.ExcelFile('../static/discharge.xlsx')
    discharge = discharge.parse('Data')
    admission = pd.ExcelFile('../static/admission.xlsx')
    admission = admission.parse('Data')
    admission_type = pd.ExcelFile('../static/admission_type_ID.xlsx')
    admission_type = admission_type.parse('Data')
    return data, discharge, admission, admission_type


def mapping():
    """Creates various features."""
    data, discharge, admission, admission_type = get_data()
    # Map discharge and admission id descriptions
    discharge_map = dict( zip(discharge['discharge_disposition_id'], discharge['description']))
    data['discharge_description'] = data['discharge_disposition_id'].map(discharge_map)
    admission_map = dict(zip(admission['admission_source_id'], admission['description']))
    data['admission_description'] = data['admission_source_id'].map(admission_map)
    # bring in admission type id
    admission_type_map = dict(zip(admission_type['admission_type_id'], admission['description']))
    data['admission_type_description'] = data['admission_type_id'].map(admission_type_map)
    # Drop original ids as these are not dropped in one-hot_encoding function
    data.drop(['discharge_disposition_id', 'admission_source_id','admission_type_id'], axis=1, inplace=True)    
    # Mapping Age
    age_ranges = list(set(data['age']))
    age_ordinal = [1, 8, 7, 6, 0, 9, 4, 2, 5, 3]
    age_map = dict(zip(age_ranges, age_ordinal))
    data['age'] = data['age'].map(age_map)
    data.replace('?', np.nan, inplace=True)
    data.drop(['weight', 'medical_specialty', 'payer_code', 'diag_2','diag_3',  'patient_nbr', 'encounter_id'], axis=1, inplace=True)
    data = data[~(data['diag_1'].str.startswith('V') | (data['diag_1'].str.startswith('E')))]
    data.dropna(inplace=True)
    # Mapping readmitted
    readmitted_ranges = list(set(data['readmitted']))
    readmitted_ordinal = [1, 1, 0]
    readmitted_map = dict(zip(readmitted_ranges, readmitted_ordinal))
    data['readmitted'] = data['readmitted'].map(readmitted_map)
    return data 


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


def subset_and_split(data):
    """Subset diabetes Patients and train test."""
    diabetes=data[data['diabetesMed=Yes'] == 1]
    X=diabetes
    inds=pd.isnull(X).any(1).nonzero()[0]
    y=diabetes['readmitted']
    X.drop(['readmitted', 'readmitted'], axis=1, inplace=True)
    # Separate training and test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X,
                         y,
                         test_size=0.30,
                         random_state=50)
    return X_train, y_train, X_test, y_test                     

    
