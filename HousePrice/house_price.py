import joblib
import streamlit as st
import pandas as pd
import numpy as np


def my_format(x):
    s = "{:,.0f}".format(x)
    L = len(s)
    if L < 14:
        s = '&nbsp'*(14-L) + s
    return s


forest_reg = joblib.load("./HousePrice/forest_reg_model.pkl")

column_names = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                'total_bedrooms', 'population', 'households', 'median_income',
                'rooms_per_household', 'population_per_household',
                'bedrooms_per_room', 'ocean_proximity_1',
                'ocean_proximity_2', 'ocean_proximity_3',
                'ocean_proximity_4', 'ocean_proximity_5']

x_test = pd.read_csv('./HousePrice/x_test.csv',
                     header=None, names=column_names)
y_test = pd.read_csv('./HousePrice/y_test.csv', header=None)
y_test = y_test.to_numpy()
N = len(x_test)


def get_5_rows():
    index = np.random.randint(0, N-1, 5)
    some_data = x_test.iloc[index]
    print(some_data)
    result_test = 'y_test:' + '&nbsp&nbsp&nbsp&nbsp'
    for i in index:
        print(y_test[i, 0])
        s = my_format(y_test[i, 0])
        result_test = result_test + s
    result_test = '<p style="font-family:Consolas; color:Blue; font-size: 15px;">' + \
        result_test + '</p>'

    some_data = some_data.to_numpy()
    y_pred = forest_reg.predict(some_data)
    result_predict = 'y_predict:' + '&nbsp'
    for i in range(0, 5):
        s = my_format(y_pred[i])
        result_predict = result_predict + s
    result_predict = '<p style="font-family:Consolas; color:Blue; font-size: 15px;">' + \
        result_predict + '</p>'

    return result_test, result_predict
