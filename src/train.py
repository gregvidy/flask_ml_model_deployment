import os
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
import joblib
from .cross_validation import CrossValidation

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    # calling the data
    df = pd.read_csv(TRAINING_DATA)
    df["kfold"] = -1

    cv = CrossValidation(df,
                         shuffle=True,
                         target_cols=["Survived"],
                         problem_type="binary_classification")
    df_split = cv.split()

    train_df = df_split[df_split.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df_split[df_split.kfold==FOLD].reset_index(drop=True)

    ytrain = train_df.Survived.values
    yvalid = valid_df.Survived.values

    train_df = train_df.drop(["Survived", "kfold"], axis=1)
    valid_df = valid_df.drop(["Survived", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    # data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(yvalid, preds))
    print(metrics.log_loss(yvalid, preds))

    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")
    print(f"{MODEL}_{FOLD} model is saved!")