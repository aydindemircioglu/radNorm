import itertools
from joblib import parallel_backend, Parallel, delayed, load, dump
import os
from scipy.stats import ttest_rel
import time

from loadData import *
from parameters import *

def testScalers(X):
    from models import createScaler
    from exp_normalization import generateAllExperiments


    preExperiments = generateAllExperiments (preParameters)
    for p in preExperiments:
        sscal = createScaler(p)
        sscal.fit(X)
        Xstd = sscal.transform(X)
        print (X.values)
        print (Xstd)


def testConstantVariables(X):
    dropKeys = [z for z in X.keys() if len(set(X[z].values))==1]
    print (len(dropKeys))


def printMissingFeatures(df):
    missing_info = df.isnull().sum().reset_index()
    missing_info.columns = ['Column', 'MissingCount']
    columns_with_missing_values = missing_info[missing_info['MissingCount'] > 0]
    columns_with_missing_values = columns_with_missing_values.sort_values(by='MissingCount', ascending=False)
    columns_with_missing_values = columns_with_missing_values[columns_with_missing_values['MissingCount'] > 10]
    print("Missings in features:")
    print(columns_with_missing_values)



if __name__ == '__main__':
    # iterate over datasets
    datasets = {}

    for d in dList:
        eval (d+"().info()")
        datasets[d] = eval (d+"().getData('./data/')")
        data = datasets[d]

        # test transform too
        y = data["Target"]
        X = data.drop(["Target"], axis = 1)

        # check if we have any dataset with NAs
        printMissingFeatures(X)
        simp = SimpleImputer(strategy="mean")
        Ximp = pd.DataFrame(simp.fit_transform(X),columns = X.columns)
        diffs = np.sum(np.sum(Ximp != X))
        if diffs == 0:
            assert( ((Ximp == X).all()).all() )
        else:
            print ("Imputing changed", diffs, "values")
            print("Missingness in %", round(diffs/X.count().sum()*100,3))

        #testScalers(X)
        testConstantVariables(X)


#
