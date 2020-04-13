from sklearn import ensemble
from sklearn import linear_model

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=150,
                                                    n_jobs=-1,
                                                    verbose=2,
                                                    random_state=1997),
    "extratrees": ensemble.RandomForestClassifier(n_estimators=500,
                                                  n_jobs=-1,
                                                  verbose=2,
                                                  random_state=123),
    "logreg": linear_model.LogisticRegression(penalty="l2",
                                              C=0.0001,
                                              max_iter=150,
                                              verbose=2,
                                              n_jobs=-1)
}