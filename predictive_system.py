# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:09:18 2023

@author: laksh
"""

import numpy as np
import pickle


#loading the saved model
loaded_model=pickle.load(open('diabetes_model.sav','rb'))
loaded_standard=pickle.load(open('diabetes_standard.sav','rb'))



input_data = (1,89,66,23,94,28.1,0.167,21,)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = loaded_standard.transform(input_data_reshaped)
print(std_data)

prediction = loaded_model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')