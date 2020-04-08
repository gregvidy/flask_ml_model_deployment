import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
    cols = [c for c in df.columns if c not in ["Age",
                                               "Fare",
                                               "SibSp",
                                               "Parch",
                                               "Family_Size",
                                               "Cabin",
                                               "Name",
                                               "PassengerId",
                                               "Survived",
                                               "Ticket"]]
    cat_feats = CategoricalFeatures(df, categorical_features=cols, encoding_type="one_hot", handle_na=True)
    df_transformed = cat_feats.fit_transform()

    # apply standard_scaler for all variables
    scaler = StandardScaler()
    numeric_cols = ["Age","SibSp","Parch","Fare","Family_Size"]
    for col in numeric_cols:
        scaler.fit(df_transformed[col].values.reshape(-1, 1))
        df_transformed[col] = scaler.transform(np.array(df_transformed[col]).reshape(-1, 1))

    return df_transformed


if __name__ == "__main__":
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")    
    test["Survived"] = -1

    full_data = pd.concat([train, test], axis=0)    
    df_transformed = preprocessing(full_data)
    
    # split to train_df and test_df
    train_df = df_transformed[df_transformed["Survived"] != -1]
    test_df = df_transformed[df_transformed["Survived"] == -1]

    # dropping columns for both train and test
    train_df.drop(columns=["Cabin","Name","PassengerId","Ticket"], axis=1, inplace=True)
    test_df.drop(columns=["Cabin","Name","Survived","Ticket"], axis=1, inplace=True)

    train_df.to_csv("../input/train_preprocessed.csv", index=False)
    test_df.to_csv("../input/test_preprocessed.csv", index=False)
    print("Data saved!")