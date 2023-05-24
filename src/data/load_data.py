# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from features.process_features import standardize

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           os.pardir, os.pardir))

RAW_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'raw')
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')

# Raw data directories
ORG_DIR = os.path.join(RAW_DATA_DIR, 'GLOBAL', 'WLD_2021_FINDEX_v02_M_csv')
FULL_DIR = os.path.join(RAW_DATA_DIR, 'AFRICA', 'FULL')
DEM_FEAT_DIR = os.path.join(RAW_DATA_DIR, 'AFRICA', 'DEMOGRAPHIC')

ORG = os.path.join(ORG_DIR, 'micro_world.csv')
FULL_MERGED = os.path.join(FULL_DIR, 'africa_data_2021_receivewages_fullfeatures_dkrfmerged.csv')
DEM_MERGED = os.path.join(DEM_FEAT_DIR, 'africa_data_2021_receivewages_dem_features_dkrfmerged.csv')

FULL_MERGED_DUMVAR = os.path.join(FULL_DIR, 'africa_data_2021_receivewages_fullfeatures_dkrfmerged_dum.csv')
DEM_MERGED_DUMVAR = os.path.join(DEM_FEAT_DIR, 'africa_data_2021_receivewages_dem_features_dkrfmerged_dum.csv')

# additional data for testing models
_2011_DIR = os.path.join(RAW_DATA_DIR, '2011')
_2011_FULL_MERGED = os.path.join(_2011_DIR, 'africa_data_2011_receivewages_fullfeatures_dkrfmerged.csv')
_2011_HUM_SEL_MERGED = os.path.join(_2011_DIR, 'africa_data_2011_receivewages_humanselectfeatures_dkrfmerged.csv')

_2014_DIR = os.path.join(RAW_DATA_DIR, '2014')
_2014_FULL_MERGED = os.path.join(_2014_DIR, 'africa_data_2014_receivewages_fullfeatures_dkrfmerged.csv')
_2014_HUM_SEL_MERGED = os.path.join(_2014_DIR, 'africa_data_2014_receivewages_humanselectfeatures_dkrfmerged.csv')

_2017_DIR = os.path.join(RAW_DATA_DIR, '2017')
_2017_FULL_MERGED = os.path.join(_2017_DIR, 'africa_data_2017_receivewages_fullfeatures_dkrfmerged.csv')
_2017_HUM_SEL_MERGED = os.path.join(_2017_DIR, 'africa_data_2017_receivewages_humanselectfeatures_dkrfmerged.csv')




DATA_LIST = ['full_merged', 'dem_merged', 'full_merged_dumvar', 'dem_merged_dumvar', 'best models', 'africa-2011', 'africa-2017', 'africa-2014']
CATEGORY_LIST = ['train', 'test']


def get_data_filepaths(data):
    if data not in DATA_LIST:
        raise ValueError("{} not a valid dataset we cover, which are {}".format(data, DATA_LIST))

    data_dir = os.path.join(DATA_DIR, data)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return (os.path.join(data_dir, 'train.pkl'),
            os.path.join(data_dir, 'test.pkl'))


def split_features_labels_weights(path,
                                  weights_col=['pop_scaled_wgt'],
                                  label_col=['receive_digital_wages']):
    '''Split data into features, labels, and weights dataframes'''
    data = pd.read_pickle(path)
    return (data.drop(weights_col + label_col, axis=1),
            data[label_col],
            data[weights_col])


def load_data(path, selected_columns=None, ravel=True, standardize_columns='numeric'):
    X, y, w = split_features_labels_weights(path)
    if selected_columns is not None:
        X = X[[col for col in X.columns.values if col in selected_columns]]
    if standardize_columns == 'numeric':
        standardize(X)
    elif standardize_columns == 'all':
        standardize(X, numeric_only=False)
    if ravel is True:
        y = np.ravel(y)
        w = np.ravel(w)
    return (X, y, w)
