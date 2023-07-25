#
from math import sqrt
import os
import shutil


def getName (s, detox = False):
    if "[('MinMax', {'Scale': (-1, 1), 'Clip': False, 'ApplyGlobalScaling'" in s:
        name = "Min-Max-NoClip"
    if "[('MinMax', {'Scale': (-1, 1), 'Clip': True, 'ApplyGlobalScaling'" in s:
        name = "Min-Max"
    if "[('None', {'ApplyGlobalScaling'" in s:
        name = "None"
    if "[('Power', {'ApplyGlobalScaling'" in s:
        name = "Power transform"
    if "[('Quantile', {'ApplyGlobalScaling'" in s:
        name = "Quantile transform (100)"
    if "[('QuantileFull', {'ApplyGlobalScaling'" in s:
        name = "Quantile transform"
    if "[('tanh', {'ApplyGlobalScaling'" in s:
        name = "Tanh transform"
    if "[('zScore', {'Quantiles': (0, 100), 'ApplyGlobalScaling'" in s:
        name = "z-Score"
    if "[('zScore', {'Quantiles': (25, 75), 'ApplyGlobalScaling'" in s:
        name = "Robust z-Score (25, 75)"
    if "[('zScore', {'Quantiles': (5, 95), 'ApplyGlobalScaling'" in s:
        name = "Robust z-Score (5, 95)"
    if detox == True:
        name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "").replace(",", "_")
    return name



def recreatePath (path, create = True):
    print ("Recreating path ", path)
    try:
        shutil.rmtree (path)
    except:
        pass

    if create == True:
        try:
            os.makedirs (path)
        except:
            pass
    print ("Done.")



def findOptimalCutoff (fpr, tpr, threshold, verbose = False):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    fpr, tpr, threshold

    Returns
    -------
    list type, with optimal cutoff value

    """

    # own way
    minDistance = 2
    bestPoint = (2,-1)
    for i in range(len(fpr)):
        p = (fpr[i], tpr[i])
        d = sqrt ( (p[0] - 0)**2 + (p[1] - 1)**2 )
        if verbose == True:
            print (p, d)
        if d < minDistance:
            minDistance = d
            bestPoint = p

    if verbose == True:
        print ("BEST")
        print (minDistance)
        print (bestPoint)
    sensitivity = bestPoint[1]
    specificity = 1 - bestPoint[0]
    return sensitivity, specificity


#
