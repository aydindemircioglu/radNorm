#
import cv2
from functools import partial
import math
import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectFromModel

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline

from ITMO_FS.filters.univariate import anova

from helpers import *



def mrmre_score (X, y, nFeatures):
    Xp = pd.DataFrame(X, columns = range(X.shape[1]))
    yp = pd.DataFrame(y, columns=['Target'])

    # we need to pre-specify the max solution length...
    solutions = mrmr.mrmr_ensemble(features = Xp, targets = yp, solution_length=nFeatures, solution_count=5)
    scores = [0]*Xp.shape[1]
    for k in solutions.iloc[0]:
        for j, z in enumerate(k):
            scores[z] = scores[z] + Xp.shape[1] - j
    scores = np.asarray(scores, dtype = np.float32)
    scores = scores/np.sum(scores)
    return scores


def bhattacharyya_score_fct (X, y):
    yn = y/np.sum(y)
    yn = np.asarray(yn, dtype = np.float32)
    scores = [0]*X.shape[1]
    for j in range(X.shape[1]):
        xn = (X[:,j] - np.min(X[:,j]))/(np.max(X[:,j] - np.min(X[:,j])))
        xn = xn/np.sum(xn)
        xn = np.asarray(xn, dtype = np.float32)
        scores[j] = cv2.compareHist(xn, yn, cv2.HISTCMP_BHATTACHARYYA)

    scores = np.asarray(scores, dtype = np.float32)
    return -scores


def train_model(paramSet, X_train, y_train, X_train_scaled = None):
    p_Pre, p_FSel, p_Clf = paramSet
    scaler = createScaler (p_Pre)
    fselector = createFSel (p_FSel)
    classifier = createClf (p_Clf)

    # apply both
    with np.errstate(divide='ignore',invalid='ignore'):
        # apply at all?
        if X_train_scaled is None:
            scaler.fit (X_train.copy())
            X_train_scaled = scaler.transform (X_train.copy())
        fselector.fit (X_train_scaled.copy(), y_train.copy())
        X_fs_train = fselector.transform (X_train_scaled)
        y_fs_train = y_train

        classifier.fit (X_fs_train, y_fs_train)

        # extract selected feats as well
        selected_feature_idx = fselector.get_support()
        #selected_feature_idx = X_train_scaled.columns[selected_feature_idx].copy()

    return [scaler, fselector, classifier, selected_feature_idx]



def test_model(trained_model, X_valid, y_valid, X_valid_scaled = None):
    scaler, fselector, classifier, _ = trained_model

    # apply model
    if X_valid_scaled is None:
        X_valid_scaled = scaler.transform (X_valid.copy())
    X_fs_valid = fselector.transform (X_valid_scaled)
    y_fs_valid = y_valid

    #print (X_fs_valid)
    y_pred = classifier.predict_proba (X_fs_valid)[:,1]
    t = np.array(y_valid)
    p = np.array(y_pred)

    return p, t



def createScaler (fExp, ignoreApply = False):
    method = fExp[0][0]

    # override scaling here, if we apply things globally
    if ignoreApply == False:
        applyGlobalScaling = fExp[0][1]["ApplyGlobalScaling"]
        if applyGlobalScaling == True:
            # create none scaler
            sscal = RobustScaler(quantile_range = (0, 100), with_centering = False, with_scaling = False, unit_variance = False)
            return sscal

    if method == "MinMax":
        scale = fExp[0][1]["Scale"]
        clip = fExp[0][1]["Clip"]
        sscal = MinMaxScaler(scale)

    if method == "zScore":
        quantiles = fExp[0][1]["Quantiles"]
        if quantiles == (0,100):
            sscal = StandardScaler()
        else:
            sscal = RobustScaler(quantile_range = quantiles, unit_variance = True)

    if method == "None":
        # dummy scaler does nothing
        sscal = RobustScaler(quantile_range = (0, 100), with_centering = False, with_scaling = False, unit_variance = False)

    if method == "tanh":
        def tanhScaling(data):
            m = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std[std<1e-6] = 1e-6
            data = 0.5 * (np.tanh(0.01 * ((data - m) / std)) + 1)
            return data

        sscal = FunctionTransformer(func = tanhScaling, check_inverse=False, feature_names_out=None)

    if method == "Power":
        sscalpre = StandardScaler(with_std = True)
        sscalpwr = PowerTransformer()
        sscal = Pipeline([('scaler_std', sscalpre), ('scaler_power', sscalpwr)])

    if method == "Quantile":
        sscal = QuantileTransformer(n_quantiles = 50, output_distribution='normal')

    if method == "QuantileFull":
        sscal = QuantileTransformer(n_quantiles = 10000, output_distribution='normal')

    return sscal



def createFSel (fExp):
    method = fExp[0][0]
    nFeatures = fExp[0][1]["nFeatures"]

    if method == "LASSO":
        C = fExp[0][1]["C"]
        clf = LogisticRegression(penalty='l1', max_iter = 100, solver='liblinear', C = C, random_state = 42)
        pipe = SelectFromModel(clf, prefit=False, max_features=nFeatures, threshold=-np.inf)

    if method == "Anova":
        pipe = SelectKBest(anova, k = nFeatures)

    if method == "Bhattacharyya":
        pipe = SelectKBest(bhattacharyya_score_fct, k = nFeatures)

    if method == "ET":
        clf = ExtraTreesClassifier(random_state = 42)
        pipe = SelectFromModel(clf, prefit=False, max_features=nFeatures, threshold=-np.inf)

    return pipe



def createClf (cExp):
    method = cExp[0][0]

    if method == "LDA":
        model = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 'auto')

    if method == "LogisticRegression":
        C = cExp[0][1]["C"]
        model = LogisticRegression(max_iter=500, solver='liblinear', C = C, random_state = 42)

    if method == "NaiveBayes":
        model = GaussianNB()

    if method == "RandomForest":
        n_estimators = cExp[0][1]["n_estimators"]
        model = RandomForestClassifier(n_estimators = n_estimators)

    if method == "RBFSVM":
        C = cExp[0][1]["C"]
        g = cExp[0][1]["gamma"]
        model = SVC(kernel = "rbf", C = C, gamma = g, probability = True)

    if method == "NeuralNetwork":
        N1 = cExp[0][1]["layer_1"]
        N2 = cExp[0][1]["layer_2"]
        N3 = cExp[0][1]["layer_3"]
        model = MLPClassifier (hidden_layer_sizes=(N1,N2,N3,), random_state=42, max_iter = 1000)

    return model

#
