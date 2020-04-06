import pandas as pd
import numpy as np
from categorical_features import CategoricalFeatures


def preprocessing(df):
    # has cabin or not
    df["Has_Cabin"] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    # split and cleansing title name
    df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    title_names = (df["Title"].value_counts() < 10)
    df["Title"] = df["Title"].apply(lambda x: "Misc" if title_names.loc[x] == True else x)

    # impute age by title    
    age_by_title = df.groupby(["Title"])["Age"].mean()
    df["Age"] = df.apply(lambda row: age_by_title[row["Title"]] if np.isnan(row["Age"]) else row["Age"], axis=1)

    # impute fare by passenger class
    fare_by_class = df.groupby(["Pclass"])["Fare"].median()
    df["Fare"] = df.apply(lambda row: fare_by_class[row["Pclass"]] if np.isnan(row["Fare"]) else row["Fare"], axis=1)

    # add several new variables
    df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
    df["Fare_Bin"] = pd.qcut(df["Fare"], 4)
    df["Age_Bin"] = pd.cut(df["Age"].astype(int), 5)

    # categorical features encoding
    df.drop(columns=["Cabin","Name","PassengerId","Ticket"], inplace=True)
    cols = [c for c in df.columns if c not in ["Age",
                                               "Fare",
                                               "Cabin",
                                               "Name",
                                               "PassengerId",
                                               "Survived",
                                               "Ticket"]]
    cat_feats = CategoricalFeatures(df, categorical_features=cols, encoding_type="one_hot",handle_na=True)
    df_transformed = cat_feats.fit_transform()

    train_df = df_transformed[df_transformed["Survived"] != -1]
    test_df = df_transformed[df_transformed["Survived"] == -1]

    test_df.drop(["Survived"], axis=1, inplace=True)
    return train_df, test_df


if __name__ == "__main__":
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")    
    test["Survived"] = -1

    full_data = pd.concat([train, test], axis=0)    
    train_df, test_df = preprocessing(full_data)

    train_df.to_csv("../input/train_preprocessed.csv", index=False)
    test_df.to_csv("../input/test_preprocessed.csv", index=False)
    print("Data saved!")