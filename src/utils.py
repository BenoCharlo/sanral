import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified,  transforms all
        columns in X.
        """
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def data_args(data):
    """This function prints the shape and the columns of a dataset
    
    Arguments:
        data {pandas dataframe}
    """
    print("The shape of the data is: ", data.shape)
    print()
    print("The columns of the dataset are:")
    print()
    for col in data.columns:
        print(col)


def daterange(start_date, end_date):
    delta = datetime.timedelta(hours=1)
    while start_date <= end_date:
        yield start_date
        start_date += delta


def simple_date_features(data, date_var):
    """
    This function creates date-related features
    
    Arguments:
        data {dataframe} -- [should contain a date like varaible in str or datetime format]
    
    Returns:
        [dataframe] -- [with newly created date-related features]
    """
    assert set([date_var]).issubset(data.columns)

    data[date_var] = pd.to_datetime(data[date_var])

    data["year"] = [date.year for date in data[date_var]]
    data["month"] = [date.month for date in data[date_var]]
    data["day"] = [date.day for date in data[date_var]]
    data["hour"] = [date.hour for date in data[date_var]]

    data["name_day"] = [date.strftime("%A") for date in data[date_var]]

    return data


def separate_train_test(data):
    """
    This function separate a joint dataframe in train/test dataframe
    
    Arguments:
        data {[dataframe]} -- [should contain a variable named is_train (train_test)]
    
    Returns:
        [dataframes] -- [2 dataframes]
    """
    assert "train_test" in list(data.columns)

    train_data = data.loc[data["train_test"] == 1]
    test_data = data.loc[data["train_test"] == 0]

    return train_data.drop("train_test", axis=1), test_data.drop("train_test", axis=1)

