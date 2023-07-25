import itertools
from joblib import parallel_backend, Parallel, delayed, load, dump
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import os
import pandas as pd
from scipy.stats import ttest_rel
import seaborn as sns
import time
from pprint import pprint

from scipy.stats import pearsonr, friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman

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



def extractDF (resultsA):
    df = []
    for r in range(len(resultsA)):
        res = {"AUC":resultsA[r]["AUC"], "Repeat": resultsA[r]["Repeat"]}
        res["Scaling"] = str(resultsA[r]["Params"][0]) # same for all
        res["FSel"] = str(resultsA[r]["Params"][1])
        res["FSel_method"] = str(resultsA[r]["Params"][1][0][0])
        for f in range(kFold):
            res[f"Fold_{f}_GT"] = resultsA[r]["Preds"][f][1]
            res[f"Fold_{f}_Preds"] = resultsA[r]["Preds"][f][0]
            res[f"FSel_{f}_names"] = str(resultsA[r]["Preds"][f][2])
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



def getAUCTable (dfA, d):
    table1 = []
    Amean = pd.DataFrame(dfA).groupby(["Scaling"])["AUC"].mean().round(3)
    Amean = Amean.rename({s:getName(s) for s in Amean.keys()})
    Astd = pd.DataFrame(dfA).groupby(["Scaling"])["AUC"].std().round(3)
    Astd = Astd.rename({s:getName(s) for s in Astd.keys()})
    ctable = pd.DataFrame(Amean)
    #ctable[d] = [str(s[0]) + " +/- " + str(s[1]) for s in list(zip(*[Amean.values, Astd.values]))]
    ctable[d] = [s[0] for s in list(zip(*[Amean.values, Astd.values]))]
    ctable = ctable.drop(["AUC"], axis = 1)
    table1.append (ctable)
    return table1



def drawArray (table3, cmap = None, clipRound = True, fsize = (9,7), aspect = None, DPI = 220, fName = None):
    #table3 = tO.copy()
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



def createBiasFigure (tR):
    df = pd.read_excel("./paper/table1.xlsx")
    tableBias = []
    for d in dList:
        try:
            resultsA = load(f"results/norm_A_{d}.dump")
            resultsB = load(f"results/norm_B_{d}.dump")
        except:
            print ("Does not exist:", d)
            continue

        # extract infos
        dfA = extractDF (resultsA)
        dfB = extractDF (resultsB)
        checkSplits(dfB)

        tableA = getAUCTable(dfA, d)
        tableB = getAUCTable(dfB, d)

        diffs = tableB[0].copy()
        for s in list(tableB[0].index):
            diffs.at[s,d] = tableB[0].loc[s][d] - tableA[0].loc[s][d]

        tableBias.append(pd.DataFrame(diffs))

    tableBias = pd.concat(tableBias, axis = 1)
    newIndex = tR.index.drop(["None"])
    tableBias = tableBias.loc[list(newIndex)]

    drawArray(tableBias.round(3), cmap = [("+", -0.022, 0, 0.022)], fsize = (14,7), aspect = 0.7, fName = "Fig5")
    plt.close('all')
    plt.rc('text', usetex=False)
    return None



def createFeatureFigures (table2, table3):
    tableFS = pd.concat(table2).groupby(level=0).mean()
    tableFS = tableFS.loc[tR.index]
    drawArray(tableFS.round(0), cmap = [("o", 0, 50, 100)], fsize = (10,7), fName = "Fig4a")

    tableFeats = pd.concat(table3).groupby(level=0).mean()
    tableFeats = tableFeats.loc[tR.index]
    drawArray(tableFeats.round(0), cmap = [("o", 0, 50, 100)], fsize = (10,7), fName = "Fig4b")
    return None



def createRankingTable (table1):
    tableAUC = pd.concat(table1, axis = 1)
    tR = tableAUC.rank(axis = 0, ascending = False).mean(axis = 1)
    tR = pd.DataFrame(tR).round(1)
    tR.columns = ["Mean rank"]
    tR = tR.sort_values(["Mean rank"])

    # how often the method performed best
    tA = tableAUC.rank(axis = 0,  ascending = False)
    tA = tA.loc[tR.index]
    rTable = tR.copy()
    tM = tableAUC.mean(axis= 1)
    tM = tM - tM["None"]
    tM = tM.round(3)
    rTable["Mean gain in AUC"] = tM
    tX = tableAUC - tableAUC.loc["None"]
    tX = tX.max(axis = 1)
    tX = tX.round(3)
    rTable["Maximum gain in AUC"] = tX
    drawArray(rTable, aspect = 0.7, fsize = (10,7), cmap = [("-", 3.5, (3.5+6.5)/2, 6.5), ("+", -0.015, 0.0, 0.015), ("+", -0.06, 0.0, 0.06)], fName = "Fig3a")

    methods = tA.index
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

    tA = tableAUC.rank(axis = 0,  ascending = False)
    tA = tA.loc[tR.index]
    drawArray(tA, cmap = [("-", 1.0, 4.5, 9.0)], fsize = (17,7), fName = "Fig3c")

    # S1
    table1 = [k.loc[tR.index] for k in table1]
    tableA = pd.concat(table1).groupby(level=0).mean()
    tableA = tableA.loc[tR.index]
    drawArray(tableA, cmap = [("*", 0, 0.50, 1.00)], fsize = (14,7), aspect = 0.7, fName = "FigS1")


    print (f"Friedman test: {friedmanchisquare(*[tableA[d] for d in tableA.columns])[1]:.3f}")
    print (posthoc_nemenyi_friedman(tableA.T))
    posthoc_nemenyi_friedman(tableA.T)

    return tR



def getTables ():
    table1 = []; table2 = []; table3 = []
    for d in dList:
        try:
            resultsA = load(f"results/norm_A_{d}.dump")
        except:
            print ("Does not exist:", d)
            continue

        # extract infos
        dfA = extractDF (resultsA)
        checkSplits(dfA)

        tableA = getAUCTable(dfA, d)
        table1.append(pd.DataFrame(tableA[0]))

        # are fsels affected by scaling?
        table2.append(featureSelectionAgreement (dfA, d))
        table3.append(featureAgreement (dfA, d))
    return table1, table2, table3



if __name__ == '__main__':
    # gather data
    table1, table2, table3 = getTables()

    # ranking table
    tR = createRankingTable (table1)

    # features
    table2 = [k.loc[tR.index][tR.index] for k in table2]
    table3 = [k.loc[tR.index][tR.index] for k in table3]
    createFeatureFigures (table2, table3)

    # application bias
    createBiasFigure (tR)


#
