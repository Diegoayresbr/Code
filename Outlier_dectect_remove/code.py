import numpy as np
import pandas as pd
from sklearn.covariance import (
    EllipticEnvelope,
)  # For multivariate df with Normal Distribution


def outlier_detector(df, percentage=0.01):

    model_detection = EllipticEnvelope(
        # contamination=percentage # outlier percentage. If you want to add manually
        random_state=0
    )

    predict_x = model_detection.fit_predict(
        df
    )  # prediction [1 or -1] # -1 are outliers

    outlier = np.where(predict_x == -1, 1, 0)  # convert to 1 or 0 # work as boolean

    # Add outlier column in DF
    df.loc[df.index, "outlier"] = outlier
    df["outlier"] = df["outlier"].astype("bool")

    # print summary
    df_check = df.copy()
    df_check["count"] = 1
    df_check = df_check.groupby(["outlier"])["count"].count()
    df_check = df_check.to_frame()
    sum_total = df_check.iloc[:, 0].sum()  # add results in percentage of total
    df_check["%_of_Total"] = df_check.iloc[:, 0].apply(
        lambda x: x / sum_total
    )  # .astype(str) + '%'
    print(df_check)

    df_no_outlier = df[df["outlier"] == False]
    df_no_outlier = df_no_outlier.iloc[:, :-1]

    print(f"\nAmount of Outliers Removed: {len(df.index) - len(df_no_outlier.index)}")

    return df_no_outlier


path = "real_state_data1.csv"
df = pd.read_csv(path, delimiter=",")

df_no_outliers = outlier_detector(df, percentage=0.1)
