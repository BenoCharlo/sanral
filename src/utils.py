def data_printing(data):
    """This function prints the shape and the columns of a dataset
    
    Arguments:
        data {pandas dataframe}
    """
    print("The shape of the data is: ", data.shape)
    print()
    print("The columns of the dataset are:")
    for col in data.columns:
        print(col)
