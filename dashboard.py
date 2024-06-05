import streamlit as st
import pickle

# Load the pre-trained model
# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)

# Custom CSS
# Title of the app
st.title('Predict your house price!')

# Input features
st.header('Input')

# Assuming the model expects three features
feature1 = st.number_input('Area')
feature2 = st.number_input('Basement Area')
feature3 = st.number_input('Garage Area')

# Make prediction
if st.button('Predict'):
    input_features = [[feature1, feature2, feature3]]
    prediction = model.predict(input_features)
    st.subheader('Predicted Output')
    st.write(prediction[0])