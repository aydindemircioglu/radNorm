#

import math
import numpy as np
import random
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, LeavePOut
from joblib import dump, load

from helpers import *
from models import train_model, test_model, createScaler



def getAUCs (preds, gts):
    fpr, tpr, thresholds = roc_curve (gts, preds)
    area_under_curve = auc (fpr, tpr)

    if (math.isnan(area_under_curve) == True):
        print ("ERROR: Unable to compute AUC of ROC curve. NaN detected!")
        raise Exception ("Unable to compute AUC")

    sens, spec = findOptimalCutoff (fpr, tpr, thresholds) # need this?
    return area_under_curve, sens, spec



def cross_validation(data, targets, hyperparameters = None, k = 5, repeatctr = 0):
    np.random.seed(repeatctr)
    random.seed(repeatctr)

    pred_matrix = {p:{f:None for f in range(k)} for p in range(len(hyperparameters))}

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state = repeatctr)
    for f, (train_index, valid_index) in enumerate(skf.split(data, targets)):
        X_train = data.iloc[train_index]
        Y_train = targets.iloc[train_index]
        X_valid = data.iloc[valid_index]
        Y_valid = targets.iloc[valid_index]

        # for each of the models or hyperparamters you want to try train and evaluate
        scalingCacheTrain = {}
        scalingCacheValid = {}
        for idx, paramSet in enumerate(hyperparameters):
            p_Pre, _, _ = paramSet
            if str(p_Pre) not in scalingCacheTrain:
                scaler = createScaler (p_Pre)
                scaler.fit (X_train.copy())
                scalingCacheTrain[str(p_Pre)] = scaler.transform (X_train.copy())
                scalingCacheValid[str(p_Pre)] = scaler.transform (X_valid.copy())

            X_train_scaled = scalingCacheTrain[str(p_Pre)].copy()
            X_valid_scaled = scalingCacheValid[str(p_Pre)].copy()

            trained_model = train_model(paramSet, X_train, Y_train, X_train_scaled = X_train_scaled)
            selected_features_idx = trained_model[3]
            selected_features = X_train.columns[selected_features_idx].copy()

            preds, gt = test_model(trained_model, X_valid, Y_valid, X_valid_scaled = X_valid_scaled)

            pred_matrix[idx][f] = (preds, gt, selected_features)

    # calculates the mean of the models for each fold given the parameter your testing
    fold_mean_auc = [None for _ in range(len(hyperparameters))]
    # dump ([pred_matrix, hyperparameters], f"./tmp/{repeatctr}.dump")
    # pred_matrix, hyperparameters = load ("./tmp/0.dump")
    for idx, _ in enumerate(hyperparameters):
        preds, gts, _ = list(zip(*pred_matrix[idx].values()))
        preds = np.concatenate(preds).ravel()
        gts = np.concatenate(gts).ravel()
        # liu, naive bayes+quantile has NANs, we replace them here
        preds[preds != preds] = 0.5
        area_under_curve, sens, spec = getAUCs (preds, gts) # need for sens spec?
        fold_mean_auc[idx] = (area_under_curve, sens, spec)
    # picks out the index of the best parameter/model, based on AUC
    best_parameter_idx = np.argmax(list(zip(*fold_mean_auc))[0])
    best_parameter = hyperparameters[best_parameter_idx]

    print ("X", end = '', flush = True)
    return {"AUC": fold_mean_auc[best_parameter_idx][0],
                "Sens": fold_mean_auc[best_parameter_idx][1],
                "Spec": fold_mean_auc[best_parameter_idx][2],
                "Preds": pred_matrix[best_parameter_idx],
                "Params": best_parameter, "Repeat": repeatctr}

#
