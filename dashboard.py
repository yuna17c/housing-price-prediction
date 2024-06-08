import streamlit as st
import pickle
import joblib
import numpy as np
from stacking_model_class import StackingAveragedModels
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
stacked_model = joblib.load('stacked_model.pkl')

with open('stacked_model.pkl', 'rb') as file:
    stacked_model = pickle.load(file)

with open('xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

st.title('Predict your house price!')

st.header('Input')
f1 = st.number_input('Area')
f2 = st.number_input('Basement Area')
f3 = st.number_input('Garage Area')
f4 = st.number_input('Year Built')
f5 = st.number_input('Overall Quality')
f6 = st.number_input('Overall Condition')

if st.button('Predict'):
    input_features = [[f1, f2, f3, f4, f5, f6]]
    x = np.array([22.583655055810706, 11.198636321275528, 7, 2.290115055569869, 1992, 1992, -13.884279591841091, 10.378762467643101, -368.4565890213832, 10.609035186067386, 4.73306922547225, 6.805844797967125, 5.684399178682556, -369056525457388.9, 9.977432868874661, 9.999442302538419e-05, -26476949.38372713, 2, 9.999414614347741e-05, 3, 9.999692328757053e-05, 2.6740307857691086, 9.999516239672483e-05, 2, 564, 5.191454825504756, 3.9355608616562465, -189.13678177493756, -1.7063697605496562e+16, -3596.9451496908405, -1.5064526509977826e+61, -7325809.697544301, 43.98523673614452, 3.5, 1, 12, 1, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, True, False, True, False, True, False, False, False, False, False, False, True, True, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, True, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False])
    x_scaled = scaler.fit_transform(x.reshape(1,-1))
    xgb_pred = np.expm1(xgb_model.predict(x_scaled))
    stacked_pred = np.expm1(stacked_model.predict(x_scaled))
    ensemble_pred = stacked_pred*0.70 + xgb_pred*0.3
    st.subheader('Predicted Output')
    val = '$'+str(round(ensemble_pred[0],2))
    st.write(val)
