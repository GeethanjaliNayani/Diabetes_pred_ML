# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/geeth_glh/ML basics/Diabetes_classification/trained_model_voting.sav','rb'))

input_data = (6,85,78,0,0,31.2,0.382,42)

input_data_as_numpy_data = np.asarray(input_data)
input_data_reshape = input_data_as_numpy_data.reshape(1,-1)


prediction = loaded_model.predict(input_data_reshape)
print(prediction)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')