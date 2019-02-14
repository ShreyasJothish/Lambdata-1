import numpy
import pandas
from sklearn.model_selection import train_test_split
from collections import defaultdict


def get_null_summary(df):
    """
    Returns the summary of NaN for each column of the dataframe and
    for complete dataframe as tuple.
    """
    return df.isna().sum(), df.isna().sum().sum()


def three_way_data_split(
    X, y, train_size=0.8, val_size=0.1, test_size=0.1,
        random_state=None, shuffle=False):
        """
        Splits the independent and dependent features data into
        train, validation and test datasets.
        Inputs:
        train_size - Size of data to be used as train dataset.
        val_size - Size of data to be used as validation dataset.
        test_size - - Size of data to be used as test dataset.
        Note: train_size + val_size + test_size should always be 1
        random_state - Needs to be set to have consistency in reproducing.
        shuffle -   True - Shuffle of dataset before spliting.
                    False - No Shuffle of dataset before spliting.
        Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        assert train_size + val_size + test_size == 1
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            shuffle=shuffle)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size/(train_size+val_size),
            random_state=random_state, shuffle=shuffle)
        return X_train, X_val, X_test, y_train, y_val, y_test


class Customer:
    """
    Customer class manages information about the customer.
    """
    def __init__(self, name, age, bill, pay_method):
        self.name = name
        self.age = age
        self.bill = bill
        self.pay_method = pay_method
