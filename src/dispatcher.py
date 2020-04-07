from sklearn import ensemble
from sklearn import linear_model

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=100,
                                                    n_jobs=-1,
                                                    verbose=2),
    "extratrees": ensemble.RandomForestClassifier(n_estimators=200,
                                                  n_jobs=-1,
                                                  verbose=2)
    "logreg": linear_model.LogisticRegression(penalty="l2",
                                              C=0.0001,
                                              max_iter=150,
                                              verbose=2,
                                              n_jobs=-1)
}