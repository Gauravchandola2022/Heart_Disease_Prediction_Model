# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle


#loading the saved model
loaded_model = pickle.load(open("C:/Users/Lenovo/OneDrive/Desktop/Deploying machine learning model/trained_model.sav",'rb'))



input_data = (34,0,1,118,210,0,1,192,0,0.7,2,0,2)

#change the input data to a nump array
input_data_as_numpy_array = np.asarray(input_data)

#reshaping the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
    print("The person has no heart defect")

else:
    print("The person has heart disease")