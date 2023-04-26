import streamlit as st
import pandas as pd
from HousePrice import house_price

st.markdown("""
    <h1 style='text-align: center;'>House price prediction in California</h1>
""", unsafe_allow_html=True)

column_names = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                'total_bedrooms', 'population', 'households', 'median_income',
                'rooms_per_household', 'population_per_household',
                'bedrooms_per_room', 'ocean_proximity_1',
                'ocean_proximity_2', 'ocean_proximity_3',
                'ocean_proximity_4', 'ocean_proximity_5']

x_test = pd.read_csv('./HousePrice/x_test.csv',
                     header=None, names=column_names)
st.dataframe(x_test)

btn_get_predictions = st.button("Get random 5 rows")

if btn_get_predictions:
    result_test, result_predict = house_price.get_5_rows()
    st.markdown(result_test, unsafe_allow_html=True)
    st.markdown(result_predict, unsafe_allow_html=True)
