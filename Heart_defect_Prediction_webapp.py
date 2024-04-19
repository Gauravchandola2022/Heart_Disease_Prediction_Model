# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:03:04 2024

@author: Lenovo
"""


import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('C:/Users/Lenovo/OneDrive/Desktop/Deploying machine learning model/trained_model.sav','rb'))
#creating a function for prediction

def heart_defect_prediction(input_data):
    
    # Convert input data to float and handle empty inputs
    input_data = [float(x) if x else 0 for x in input_data]
    

    #change the input data to a nump array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshaping the numpy array as we are predicting for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0] == 0):
        return "The person has no heart defect"

    else:
        return "The person has heart disease"
    
    
    
    
def main():
    
    
    #giving a title
    st.title('Heart Disease Prediction Web App')
    
    #getting the input data from the user
   
    
    age = st.text_input('Age')
    sex = st.text_input('Gender(0=female and 1=male)')
    cp = st.text_input('chest pain')
    trestbps = st.text_input('Rests Blood Pressure')
    chol = st.text_input('Cholesterol')
    fbs = st.text_input(' Fasting Blood Sugar')
    restecg = st.text_input('Resting electrocardiographic')
    thalach = st.text_input('Maximum Heart rate')
    exang = st.text_input('Exercise induced angina')
    oldpeak = st.text_input('ST depression')
    slope = st.text_input('Slope peak exercise ST')
    ca = st.text_input('No. Of major vessels')
    thal = st.text_input('Thalassemia')
    
    
    
    #code for prediction
    diagnosis = ""
    #creating a button for prediction
    if st.button("Heart Defect Test Result"):
        diagnosis = heart_defect_prediction([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
    
    
    
    