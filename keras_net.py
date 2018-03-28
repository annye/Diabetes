# Importing Libraries
from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
import sys
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time
import keras
from keras.layers import Flatten, Conv1D, Dense, GlobalAvgPool1D, GlobalMaxPool1D, Dropout, MaxPool1D, BatchNormalization
from keras.regularizers import L1L2

def set_trace():
    """A Poor mans break point"""
    # without this in iPython debugger can generate strange characters.
    from IPython.core.debugger import Pdb
    Pdb().set_trace(sys._getframe().f_back)


def create_baseline(input_nodes,
                    hidden1_nodes,
                    hidden2_nodes,
                    hidden3_nodes,
                    hidden4_nodes,
                    output_nodes,
                    dropout_rate):
        # NN architecturE
        model = Sequential()
        #model.add(GlobalAvgPool1D(input_shape=(146,2,)))
        model.add(Dense(output_dim=hidden1_nodes,
                        input_dim=input_nodes,
                        activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim=hidden2_nodes,
                        input_dim=hidden1_nodes,
                        activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim=hidden3_nodes,
                        input_dim=hidden2_nodes,
                        activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim=hidden4_nodes,
                        input_dim=hidden3_nodes,
                        activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim=output_nodes,
                        input_dim=hidden4_nodes,
                        activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


def structure_input_categorical_crossentropy(X_train, y_train, X_test, y_test):
    """A Poor mans break point"""
    #categorical_crossentrophy expects targets to be binary matrices - if target are integer classes
    # we need to convert to the expected format using to categorical
    # fix random seed for reproducibility
    y_train = np.array(y_train)
    y_train = [[x] for x in y_train]
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_test = [[x] for x in y_test]
    y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test


def nn_sequential(X_train, y_train, X_test, y_test):
    """ Keras Sequential model with Manual validation."""
    start_time = time.time()
    # Only for categorical_crossentrophy - Comment out for binary_crossentrophy
    #y_train = keras.utils.to_categorical(y_train, num_classes=3)
    #y_test = keras.utils.to_categorical(y_test, num_classes=3)

    seed = 13
    np.random.seed(seed)
    cvscores = []
    learning_rate = 0.02
    sgd = SGD(lr=learning_rate, momentum=0.05, nesterov=False)

    input_nodes = 146
    hidden1_nodes = 50
    hidden2_nodes = 50
    hidden3_nodes = 50
    hidden4_nodes = 50
    output_nodes = 1
    dropout_rate = 0.1


    model = create_baseline(input_nodes,
                            hidden1_nodes,
                            hidden2_nodes,
                            hidden3_nodes,
                            hidden4_nodes,
                            output_nodes,
                            dropout_rate)


    model.fit(X_train, y_train, epochs=50, batch_size=120)
    print('--- %s seconds ---' % (time.time() - start_time))
    '''
    # evaluate model with standardized dataset
    estimator = KerasClassifier(build_fn=model,
                                nb_epoch=100,
                                batch_size=200,
                                verbose=2)
    kfold = StratifiedKFold(n_splits=10,
                            shuffle=True,
                            random_state=seed)
    results = cross_val_score(estimator,
                            X,
                            y,
                            cv=kfold)
    '''
    #print("Results of 10-fold Cross-Validation: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
