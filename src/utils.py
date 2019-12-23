import datetime


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
