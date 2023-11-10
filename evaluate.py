import itertools
from joblib import parallel_backend, Parallel, delayed, load, dump
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import re

from sklearn.calibration import calibration_curve
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms


import os
import pandas as pd
from scipy.stats import ttest_rel
import seaborn as sns
import time
from pprint import pprint

from scipy.stats import pearsonr, friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

from parameters import *
from helpers import *



def iou(outputs: np.array, labels: np.array):
    SMOOTH = 0
    #print (labels.shape)
    #print (outputs.shape)
    outputs = np.asarray(outputs).astype(bool)
    labels = np.asarray(labels).astype(bool)

    intersection = (outputs & labels).sum((0))
    union = (outputs | labels).sum((0))

    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou



def checkSplits(df):
    # check that the splits were the same by checking gts
    # we have for each repeat 8 (or whatever) scaling and these must be the same
    for r in range(nRepeats):
        z = df.query("Repeat == @r").copy()
        for f in range(kFold):
            K = [tuple(k) for k in z[f"Fold_{f}_GT"].values]
            assert(len(set(K)) == 1)
    pass


# https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d#:~:text=Definition,into%20M%20equally%20spaced%20bins.
def expected_calibration_error(samples, true_labels, M=5):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

   # keep confidences / predicted "probabilities" as they are
    confidences = samples
    # get binary class predictions from confidences
    predicted_label = (samples>0.5).astype(float)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prop_in_bin = in_bin.astype(float).mean()

        if prop_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece



def addScores (resultsA):
    for j in range(len(resultsA)):
        all_preds = []; all_gt = []
        for k in range(5):
            all_preds.extend(resultsA[j]["Preds"][k][0])
            all_gt.extend(resultsA[j]["Preds"][k][1])
        fpr, tpr, thresholds = roc_curve (all_gt, all_preds)
        area_under_curve = auc (fpr, tpr)
        assert (np.abs(resultsA[j]["AUC"] - area_under_curve) < 1e-4)

        # detrmine spec/sens
        sens, spec = findOptimalCutoff (fpr, tpr, thresholds) # need this?
        resultsA[j]["Sens"] = sens
        resultsA[j]["Spec"] = spec

        brier_score = brier_score_loss(all_gt, all_preds)
        prob_true, prob_pred = calibration_curve(all_gt, all_preds, n_bins=5, strategy='quantile')
        ECE = float(expected_calibration_error(prob_pred, prob_true, 5))
        resultsA[j]["Brier"] = brier_score
        resultsA[j]["ECE"] = ECE
        resultsA[j]["SampleSize"] = len(all_gt)

    return resultsA


def extractDF (resultsA):
    df = []
    for r in range(len(resultsA)):
        res = {"AUC":resultsA[r]["AUC"], "Repeat": resultsA[r]["Repeat"]}
        res["Sens"] = resultsA[r]["Sens"]
        res["Spec"] = resultsA[r]["Spec"]
        res["Brier"] = resultsA[r]["Brier"]
        res["ECE"] = resultsA[r]["ECE"]
        res["Scaling"] = str(resultsA[r]["Params"][0]) # same for all
        res["FSel"] = str(resultsA[r]["Params"][1])
        res["FSel_method"] = str(resultsA[r]["Params"][1][0][0])
        for f in range(kFold):
            res[f"Fold_{f}_GT"] = resultsA[r]["Preds"][f][1]
            res[f"Fold_{f}_Preds"] = resultsA[r]["Preds"][f][0]
            res[f"FSel_{f}_names"] = str(resultsA[r]["Preds"][f][2])
        res["SampleSize"] = resultsA[r]["SampleSize"]
        df.append(res)
    df = pd.DataFrame(df)
    return df



def featureSelectionAgreement (dfA, d):
    dfA['FSel_code'] = pd.factorize(dfA['FSel_method'])[0]
    dfA['Paramcode'], codeIndex = pd.factorize(dfA['Scaling'])
    nM = len(set(dfA["Paramcode"]))
    fMat = np.zeros((nM, nM))
    for i in set(dfA["Paramcode"]):
        for j in set(dfA["Paramcode"]):
            agreement = []
            wi = dfA.query("Paramcode == @i")
            wj = dfA.query("Paramcode == @j")
            assert (len(wi) == len(wj))
            vi = wi.sort_values(["Repeat"])["FSel_code"].values
            vj = wj.sort_values(["Repeat"])["FSel_code"].values
            agreement = np.sum(vi == vj)/len(vi)
            fMat[i,j] = np.mean(agreement)

    fMat = (fMat*100).round(0).astype(int)
    pMat = pd.DataFrame(fMat)
    pMat.columns = [getName(k) for k in codeIndex]
    pMat.index = pMat.columns
    drawArray(pMat, cmap = [("o", 0, 50, 100)], fsize = (10,7), fName = f"Table_Fsel_{d}")
    plt.close('all')
    plt.rc('text', usetex=False)
    return pMat



def featureAgreement (dfA, d):
    dfA['Paramcode'], codeIndex = pd.factorize(dfA['Scaling'])
    nM = len(set(dfA["Paramcode"]))
    fMat = np.zeros((nM, nM))
    for i in set(dfA["Paramcode"]):
        for j in set(dfA["Paramcode"]):
            agreement = []
            for r in range(nRepeats):
                z = dfA.query("Repeat == @r")
                wi = z.query("Paramcode == @i")
                wj = z.query("Paramcode == @j")
                for f in range(kFold):
                    # expect al FSel_codes to be the same
                    vi = wi[[f'FSel_{f}_names']]
                    vj = wj[[f'FSel_{f}_names']]
                    vi = list(eval("pd."+vi[f"FSel_{f}_names"].iloc[0]))
                    vj = list(eval("pd."+vj[f"FSel_{f}_names"].iloc[0]))
                    fb = set(vi) | set(vj)
                    vi = np.array([1 if v in vi else 0 for v in fb])
                    vj = np.array([1 if v in vj else 0 for v in fb])
                    agreement.extend( [iou(vi, vj)] )
            fMat[i,j] = np.mean(agreement)

    fMat = (fMat*100).round(0).astype(int)
    pMat = pd.DataFrame(fMat)
    pMat.columns = [getName(k) for k in codeIndex]
    pMat.index = pMat.columns
    drawArray(pMat, cmap = [("o", 0, 50, 100)], fsize = (10,7), fName = f"Table_Features_{d}")
    plt.close('all')
    plt.rc('text', usetex=False)
    return pMat



def getPal (cmap):
    if cmap == "g":
        #np.array([0.31084112, 0.51697441, 0.22130127, 1.        ])*255
        pal = sns.light_palette("#4f8338", reverse=False, as_cmap=True)
    elif cmap == "o":
        pal = sns.light_palette("#ff4433", reverse=False, as_cmap=True)
    elif cmap == "+":
        pal  = sns.diverging_palette(20, 120, as_cmap=True)
    elif cmap == "-":
        pal  = sns.diverging_palette(120, 20, as_cmap=True)
    else:
        pal = sns.light_palette("#ffffff", reverse=False, as_cmap=True)
    return pal



def drawArray (table3, cmap = None, clipRound = True, fsize = (9,7), aspect = None, DPI = 220, fName = None):
    table3 = table3.copy()
    if clipRound == True:
        for k in table3.index:
            for l in table3.columns:
                if str(table3.loc[k,l])[-2:] == ".0":
                    table3.loc[k,l] = str(int(table3.loc[k,l]))
    # display graphically
    scMat = table3.copy()
    strMat = table3.copy()
    strMat = strMat.astype( dtype = "str")
    # replace nans in strMat
    strMat = strMat.replace("nan", "")

    if 1 == 1:
        plt.rc('text', usetex=True)
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"]})
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{mathtools}
            \usepackage{helvet}
            \renewcommand{\familydefault}{\sfdefault}        '''


        fig, ax = plt.subplots(figsize = fsize, dpi = DPI)
        sns.set(style='white')
        #ax = sns.heatmap(scMat, annot = cMat, cmap = "Blues", fmt = '', annot_kws={"fontsize":21}, linewidth = 2.0, linecolor = "black")
        dx = np.asarray(scMat, dtype = np.float64)


        if len(cmap) > 1:
            for j, (cm, vmin, vcenter, vmax) in enumerate(cmap):
                pal = getPal(cm)
                m = np.ones_like(dx)
                m[:,j] = 0
                Adx = np.ma.masked_array(dx, m)
                tnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                ax.imshow(Adx, cmap=pal, norm = tnorm, interpolation='nearest', aspect = aspect)
                #cba = plt.colorbar(pa,shrink=0.25)
        else:
            if cmap[0][0] == "*":
                for j in range(scMat.shape[1]):
                    pal = getPal("o")
                    m = np.ones_like(dx)
                    m[:,j] = 0
                    Adx = np.ma.masked_array(dx, m)
                    vmin = np.min(scMat.values[:,j])
                    vmax = np.max(scMat.values[:,j])
                    vcenter = (vmin + vmax)/2
                    tnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                    ax.imshow(Adx, cmap=pal, norm = tnorm, interpolation='nearest', aspect = aspect)
                    #cba = plt.colorbar(pa,shrink=0.25)
            else:
                cm, vmin, vcenter, vmax = cmap[0]
                pal = getPal(cm)
                tnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                ax.imshow(dx, cmap=pal, norm = tnorm, interpolation='nearest', aspect = aspect)


        # Major ticks
        mh, mw = scMat.shape
        ax.set_xticks(np.arange(0, mw, 1))
        ax.set_yticks(np.arange(0, mh, 1))

        # Minor ticks
        ax.set_xticks(np.arange(-.5, mw, 1), minor=True)
        ax.set_yticks(np.arange(-.5, mh, 1), minor=True)

     # Gridlines based on minor ticks
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

        for i, c in enumerate(scMat.index):
            for j, f in enumerate(scMat.keys()):
                ax.text(j, i, strMat.at[c, f],    ha="center", va="center", color="k", fontsize = 22)
        plt.tight_layout()
        ax.xaxis.set_ticks_position('top') # the rest is the same
        ax.set_xticklabels(scMat.keys(), rotation = 45, ha = "left", fontsize = 22)
        ax.set_yticklabels(scMat.index, rotation = 0, ha = "right", fontsize = 22)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_tick_params ( labelsize= 22)

    if fName is not None:
        fig.savefig(f"./results/{fName}.png", facecolor = 'w', bbox_inches='tight')




def drawArray2 (scMat, strMat, colmaps = None, cell_widths = None, clipRound = True, fsize = (9,7), vofs = 0.8, hofs = None, aspect = None, DPI = 220, fName = None):
    plt.rc('text', usetex=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"]})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'''
        \usepackage{mathtools}
        \usepackage{helvet}
        \renewcommand{\familydefault}{\sfdefault}        '''

    # Create the Matplotlib figure
    fig, ax = plt.subplots(figsize = fsize, dpi = DPI)

    cmaps = {}
    for j, column in enumerate(scMat.columns):
        cm, vmin, vcenter, vmax = colmaps[j]
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        cmap = getPal(cm)
        cmaps[column] = cmap, norm

    table = plt.table(cellText=scMat.values, colLabels=scMat.columns, rowLabels=scMat.index, cellLoc='center', loc='center', colColours=['#f5f5f5']*len(scMat.columns))

    # Plot the DataFrame as a table with colored cells
    for key, cell in table._cells.items():
        if key[0] == 0:
            cell.set_text_props(weight='bold')
        cell.set_fontsize(12)
        try:
            column_name = scMat.columns[key[1]]
            cmap, norm = cmaps[column_name]
            cell.set_facecolor(cmap(norm(float(cell.get_text().get_text()))))
            cell.set_width(cell_widths[key[1]])
        except:
            pass

    for key, cell in table._cells.items():
        if key[1] == -1:
            cell.set_text_props(ha='right')

        cell.set_fontsize(22)
        try:
            cell.set_width(cell_widths[key[1]])
        except:
            pass
        if key[0] == 0:
            cell.set_height(0.5)  # Set the cell height
            cell.set_text_props(rotation=45, ha='left', va = 'top')  # Rotate and align column names

        else:
            cell.set_height(0.125)  # Set the cell height

    # Remove the x and y ticks
    ax.axis('off')

    # Hide grid lines for index and column names
    table.auto_set_font_size(False)
    table.set_fontsize(22)
    for key, cell in table._cells.items():
        if key[0] == 0 or key[1] == -1:
            cell.set_edgecolor('white')

    for i in range(strMat.shape[0]):
        for j in range(strMat.shape[1]):
            table.get_celld()[(i+1, j)].get_text().set_text(strMat.iloc[i,j])

    # delete columns, we add them in a moment
    for j in range(strMat.shape[1]):
        table.get_celld()[(0, j)].get_text().set_text("")


    ofs = hofs+cell_widths[0]/2
    cp_widths = [ofs]
    for k in range(1,len(cell_widths)):
        ofs += cell_widths[k-1]/2
        ofs += cell_widths[k]/2
        cp_widths.append(ofs)

    for j, (key, cell) in enumerate(table._cells.items()):
        if j == 0:
            cofs = cell.get_x()
        if key[0] == 0:
            column_name = scMat.columns[key[1]]
            cell.set_facecolor('white')
            ax.annotate(column_name,
                        xytext = (cp_widths[key[1]], vofs),
                        textcoords='axes fraction',
                        xy=(-0, vofs),
                        fontsize=22, rotation = 45)

    # Save the figure as a PNG file
    plt.tight_layout()

    if fName is not None:
        fig.savefig(f"./results/{fName}.png", facecolor = 'w', bbox_inches='tight')



def createFeatureFigures (table2, table3):
    tableFS = pd.concat(table2).groupby(level=0).mean()
    tableFS = tableFS.loc[tR.index]
    drawArray(tableFS.round(0), cmap = [("o", 0, 50, 100)], fsize = (10,7), fName = "Fig4a")

    tableFeats = pd.concat(table3).groupby(level=0).mean()
    tableFeats = tableFeats.loc[tR.index]
    drawArray(tableFeats.round(0), cmap = [("o", 0, 50, 100)], fsize = (10,7), fName = "Fig4b")
    return None



if __name__ == '__main__':
    computeFeatures = False
    tableFSA = []
    tableFA = []

    meanTables = {mmeth:[] for mmeth in ["AUC", "Brier", "ECE", "Sens", "Spec"]}
    diffTables = {mmeth:[] for mmeth in ["AUC", "Brier", "ECE", "Sens", "Spec"]}
    biasTables = {mmeth:[] for mmeth in ["AUC", "Brier", "ECE", "Sens", "Spec"]}
    sampleSizes = {}

    for d in dList:
        try:
            resultsA = load(f"results/norm_A_{d}.dump")
            resultsB = load(f"results/norm_B_{d}.dump")
        except:
            print ("Does not exist:", d)
            continue
        for mmeth in ["AUC", "Brier", "ECE", "Sens", "Spec"]:
            resultsA = addScores (resultsA)
            dfA = extractDF (resultsA)
            sampleSizes[d] = dfA.iloc[0]["SampleSize"]
            checkSplits(dfA)

            # are fsels affected by scaling?
            if computeFeatures == True:
                tableFSA.append(featureSelectionAgreement (dfA, d))
                tableFA.append(featureAgreement (dfA, d))

            # get Ranking
            mTbl = dfA.pivot(index='Repeat', columns='Scaling', values=mmeth)
            mTbl.columns = [getName(s) for s in mTbl.columns]
            meanTables[mmeth].append(mTbl.copy())

            # get AUC diffs and SDs
            n = mTbl["None"]
            dTbl = mTbl.copy()
            for c in dTbl:
                dTbl[c] = dTbl[c] - n
            diffTables[mmeth].append(dTbl.copy())

            # for other exp too
            resultsB = addScores (resultsB)
            dfB = extractDF (resultsB)
            checkSplits(dfB)
            nTbl = dfB.pivot(index='Repeat', columns='Scaling', values=mmeth)
            nTbl.columns = [getName(s) for s in nTbl.columns]

            bTbl = nTbl - mTbl[nTbl.columns]
            biasTables[mmeth].append(bTbl.copy())


    rTable = {}
    K = pd.concat(meanTables["AUC"], axis = 0).mean(axis = 0)
    print ("Mean gain", np.round(K["z-Score"] - K["None"], 3))
    S = pd.concat(meanTables["AUC"], axis = 0).std(axis = 0)
    print ("AUC of z-score:", np.round(K["z-Score"], 3), "+/-", np.round(S["z-Score"], 3))
    print ("AUC of None:", np.round(K["None"], 3), "+/-", np.round(S["None"], 3))

    scMat = pd.DataFrame(K)
    scMat.columns = ["Mean AUC"]
    strMat = pd.DataFrame(np.round(K, 3).astype(str) + " +/- " + np.round(S, 3).astype(str))
    strMat.columns = ["Mean AUC"]
    strMat = strMat.applymap(lambda x: x.replace('+/-', '±') if isinstance(x, str) else x)

    cell_widths = [0.20]
    drawArray2(scMat, strMat, fsize = (13,10), cell_widths = cell_widths, hofs = 0.38, \
            colmaps = [("+", 0.70, 0.71, 0.72)],
                    fName = "FigS2")



    tableAUC = pd.concat([m.mean(axis = 0) for m in meanTables["AUC"]], axis = 1)
    tR = tableAUC.rank(axis = 0, ascending = False).mean(axis = 1)
    tR = pd.DataFrame(tR).round(1)
    tR = tR.sort_values([0])
    rTable["Mean rank (AUC)"] = tR[0]

    tmps = []
    for j, d in enumerate(dList):
        tmp = meanTables["AUC"][j].mean(axis = 0)
        tmps.append(tmp - tmp["None"])
    rTable["Max gain in AUC"] = pd.concat(tmps, axis = 1).max(axis = 1).round(3)
    sTable = rTable.copy()


    meanGains = {}
    sdGains = {}
    for mmeth in ["AUC", "Sens", "Spec", "Brier", "ECE"]:
        meanGains[mmeth] = pd.concat(meanTables[mmeth], axis = 0).mean(axis=0).round(3)
        sdGains[mmeth] = pd.concat(diffTables[mmeth], axis = 0).std(axis=0).round(3)
        #rTable["Mean gain in "+mmeth] = meanGains[mmeth]-meanGains[mmeth]["None"]
        smeth = mmeth.replace("Sens", "Sensitivity").replace("Spec", "Specificity").replace("Brier", "Brier score")
        rTable["Mean gain in "+smeth] = meanGains[mmeth]-meanGains[mmeth]["None"]
        sTable["Mean gain in "+smeth] = np.round(meanGains[mmeth]-meanGains[mmeth]["None"], 3).astype(str) + " +/- " + np.round(sdGains[mmeth],3).astype(str)

    scMat = pd.concat(rTable, axis = 1)
    strMat = pd.concat(sTable, axis = 1)
    strMat = strMat.applymap(lambda x: x.replace('+/-', '±') if isinstance(x, str) else x)

    cell_widths = [0.12, 0.15, 0.25, 0.25, 0.25, 0.25, 0.25]
    drawArray2(scMat, strMat, fsize = (23,10), cell_widths = cell_widths, hofs = -0.27, \
            colmaps = [("-", 3.5, (3.5+6.5)/2, 6.5), \
                    ("+", -0.015, 0.0, 0.06), \
                    ("+", -0.03, 0.0, 0.03), \
                    ("+", -0.08, 0.0, 0.08), \
                    ("+", -0.08, 0.0, 0.08), \
                    ("-", -0.05, 0.0, 0.05), \
                    ("-", -0.09, 0.0, 0.09)], \
                    fName = "Fig3a")


    tA = tableAUC.round(3).rank(axis = 0, ascending = False, method = 'average')
    tA.columns = dList

    methods = strMat.index
    tO = np.zeros((len(methods), len(methods)))
    tO = pd.DataFrame (tO)
    tO.index = methods
    tO.columns = methods

    for m in tA.index:
        for n in tA.index:
            wins = np.sum(tA.loc[m] < tA.loc[n])
            draws = np.sum(tA.loc[m] == tA.loc[n])
            score = wins + draws*0.5
            #tA.loc[m] < tA.loc[n]
            tO.loc[m,n] = score
        tO.loc[m,m] = None
    drawArray(tO, fsize = (10,7), cmap = [("+", 0, len(dList)/2, len(dList))], fName = "Fig3b")

    tA = tableAUC.round(3).rank(axis = 0,  ascending = False, method = 'average')
    tA = tA.loc[tR.index]
    tA.columns = dList

    drawArray(tA, cmap = [("-", 1.0, 4.5, 9.0)], fsize = (17,7), fName = "Fig3c")

    # S1

    tableAUC = pd.concat([m.mean(axis = 0) for m in meanTables["AUC"]], axis = 1).round(3)
    tableAUCSD = pd.concat([m.std(axis = 0) for m in meanTables["AUC"]], axis = 1).round(3)
    tableAUC.columns = dList
    tableAUCSD.columns = dList
    tableAUC = tableAUC.loc[tR.index]
    tableAUCSD = tableAUCSD.loc[tR.index]


    # split into two tables
    scMat = tableAUC
    strMat = tableAUC.astype(str) + " +/- " + tableAUCSD.astype(str)
    strMat = strMat
    strMat = strMat.applymap(lambda x: x.replace('+/-', '±') if isinstance(x, str) else x)


    for f in range(3):
        f_scMat = scMat.iloc[:,f*5:f*5+5].copy()
        f_strMat = strMat.iloc[:,f*5:f*5+5].copy()
        cell_widths = [0.20]*f_scMat.shape[1]
        colmaps = []
        for k in f_scMat.columns:
            vmin =  f_scMat[k].min(axis = 0)
            vmax =  f_scMat[k].max(axis = 0)
            colmaps.append( ("g", vmin-0.01, (vmax+vmin)/2, vmax+0.01))
        drawArray2(f_scMat, f_strMat, fsize = (16,9), cell_widths = cell_widths, hofs = -0.01,\
                colmaps = colmaps, fName = f"FigS1_{f}")


    print (f"Friedman test: {friedmanchisquare(*[scMat[d] for d in scMat.columns])[1]:.3f}")
    print (posthoc_nemenyi_friedman(scMat.T))



    ### >>> for  bias

    rTable = {}
    tableBias = pd.concat([m.mean(axis = 0) for m in biasTables["AUC"]], axis = 1)
    rTable["Max bias in AUC"] = tableBias.max(axis = 1).round(3)
    sTable = rTable.copy()

    meanGains = {}
    sdGains = {}
    for mmeth in ["AUC", "Sens", "Spec", "Brier", "ECE"]:
        meanGains[mmeth] = pd.concat(biasTables[mmeth], axis = 0).mean(axis=0).round(3)
        sdGains[mmeth] = pd.concat(biasTables[mmeth], axis = 0).std(axis=0).round(3)
        #rTable["Mean gain in "+mmeth] = meanGains[mmeth]-meanGains[mmeth]["None"]
        smeth = mmeth.replace("Sens", "Sensitivity").replace("Spec", "Specificity").replace("Brier", "Brier score")
        rTable["Mean bias in "+smeth] = meanGains[mmeth]
        sTable["Mean bias in "+smeth] = np.round(meanGains[mmeth], 3).astype(str) + " +/- " + np.round(sdGains[mmeth],3).astype(str)

    scMat = pd.concat(rTable, axis = 1)
    strMat = pd.concat(sTable, axis = 1)
    strMat = strMat.applymap(lambda x: x.replace('+/-', '±') if isinstance(x, str) else x)
    cell_widths = [0.15, 0.25, 0.25, 0.25, 0.25, 0.25]
    drawArray2(scMat, strMat, fsize = (19,10), cell_widths = cell_widths, hofs = -0.21, vofs = 0.72, \
            colmaps = [("-", -0.025, 0, 0.025), \
                    ("+", -0.03, 0.0, 0.03), \
                    ("+", -0.08, 0.0, 0.08), \
                    ("+", -0.08, 0.0, 0.08), \
                    ("-", -0.05, 0.0, 0.05), \
                    ("-", -0.09, 0.0, 0.09)], \
                    fName = "Fig5")

    tmps = []
    for j, d in enumerate(dList):
        tmp = biasTables["AUC"][j].mean(axis = 0)
        tmps.append(tmp )
    tmps = pd.concat(tmps, axis = 1).round(3)
    tmps.columns = dList
    scMat = tmps.loc[tR.index.drop(["None"])]
    strMat = scMat.copy()
    cell_widths = [0.10]*len(dList)
    drawArray2(scMat, strMat, fsize = (25,10), cell_widths = cell_widths, hofs = -0.27, vofs = 0.72, \
            colmaps = [("-", -0.025, 0, 0.025)]*len(dList), fName = "Fig6")


    # features
    if computeFeatures == True:
        tableFSA = [k.loc[tR.index][tR.index] for k in tableFSA]
        tableFA = [k.loc[tR.index][tR.index] for k in tableFA]
        createFeatureFigures (tableFSA, tableFA)



#
