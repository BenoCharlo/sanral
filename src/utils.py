import datetime
import pandas as pd
import numpy as np


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

    data["year"] = data[date_var].year
    data["month"] = data[date_var].month
    data["day"] = data[date_var].day
    data["hour"] = data[date_var].hour

    data["name_day"] = [date.strftime("%A") for date in data[date_var]]

    return data
