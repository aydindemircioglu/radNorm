from collections import OrderedDict
import numpy as np

ncpus = 25
nRepeats = 100
kFold = 5

dList =  [ "Arita2018",  "Carvalho2018", \
                "Hosny2018A", "Hosny2018B", "Hosny2018C", \
                "Ramella2018",  "Saha2018", "Lu2019", "Sasaki2019", "Toivonen2019", "Keek2020", "Li2020", \
                "Park2020",
                "Song2020", "Veeraraghavan2020" ]

#dList = dList[::-1]
preParameters = OrderedDict({
    "Normalization": {
        "Methods": {
            "MinMax": {"Scale": [(-1,1)], "Clip": [True]},
            "zScore": {"Quantiles": [(0,100), (5, 95), (25, 75)]},
            "Power": {},
            "QuantileFull": {},
            "tanh": {},
            "None": {}
        }
    }
})


fselParameters = OrderedDict({
    # these are 'one-of'
    "FeatureSelection": {
        "N": [1,2,4,8,16,32,64],
        "Methods": {
            "LASSO": {"C": [1.0]},
            "ET": {},
            "Anova": {},
            "Bhattacharyya": {},
        }
    }
})


clfParameters = OrderedDict({
    "Classification": {
        "Methods": {
            "LogisticRegression": {"C": np.logspace(-6, 6, 7, base = 2.0) },
            "NaiveBayes": {},
            "RBFSVM": {"C":np.logspace(-6, 6, 7, base = 2.0), "gamma":["auto"]},
            "RandomForest": {"n_estimators": [125, 250]},
        }
    }
})


#
