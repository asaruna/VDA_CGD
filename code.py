
import numpy as np
import os
from matplotlib import pyplot
from numpy import interp
import sklearn, tensorflow
import xlsxwriter 
import xlrd
from sklearn import svm 
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import gzip
import pandas as pd
import pdb
import random
from random import randint
import scipy.io
from tensorflow.keras import layers
from keras.layers import Input, Dense
from keras.engine.training import Model
from keras.models import Sequential, model_from_config,Model
from keras.layers.core import  Dropout, Activation, Flatten
from keras.layers import PReLU
from keras.utils import np_utils, generic_utils
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import LSTM
from keras.layers import Embedding
from keras import regularizers
from keras.constraints import maxnorm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from genetic_selection import GeneticSelectionCV
from sklearn.preprocessing import LabelEncoder
from keras.layers.serialization import activation
from keras.models import Sequential, model_from_config,Model
from sklearn.neighbors import KNeighborsClassifier