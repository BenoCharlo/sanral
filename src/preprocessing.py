import pandas as pd
import numpy as np
import datetime

from src import utils


def create_daily_hours(start_date, end_date):
    """
    This function create then time range with a 
    step of an hour between start_date and end_date.

    Returns a list of datetime object
    """
    assert type(start_date) == str
    assert type(end_date) == str

    start_date = start_date + " 00:00:00"
    end_date = end_date + " 23:00:00"

    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

    dates = []
    for date in utils.daterange(start_date, end_date):
        dates.append(date)

    return dates


def create_new_train(dates, roads_id):

    dates_length = len(dates)
    roads_length = len(roads_id)
    dates = [str(date) for date in dates]

    dates = list(np.repeat(dates, roads_length))
    roads_id = roads_id * dates_length

    assert len(dates) == len(roads_id)
    data_output = pd.DataFrame({"datetime": dates, "segment_id": roads_id})

    return data_output
