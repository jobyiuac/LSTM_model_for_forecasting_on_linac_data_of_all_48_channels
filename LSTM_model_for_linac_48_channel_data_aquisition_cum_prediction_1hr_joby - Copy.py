# univariate lstm example
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import http.client
import string
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten


import time # for pause & delay
import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta



column_index = 10 # 10 entered here means T9 in control room


#__________function to read last value from the desired column of the space-separated TRV dataset which contains 40 columns_________
def read_last_values(filename, column_index):

        with open(filename, 'r') as file:
            last_value = None
            for line in file:
                elements = line.strip().split()  # Split the line by spaces
                if len(elements) > column_index:
                    last_value = elements[column_index]
            return last_value










#__________function to read read last two time format by reading the desired column of a txt file and print the difference of the two in seconds_________


def get_last_and_second_last_values_of_columns(txt_file, column_index1, column_index2):
    # Open the text file
    with open(txt_file, 'r') as file:
        # Read lines from the file
        lines = file.readlines()
        
        # Get the last line
        last_line = lines[-1]
        # Get the second last line
        second_last_line = lines[-2]
        
        # Split the last line by spaces
        last_values = last_line.split()
        # Split the second last line by spaces
        second_last_values = second_last_line.split()
        
        # Get the values of the desired columns from the last line
        last_column_value1 = last_values[column_index1]
        last_column_value2 = last_values[column_index2]
        
        # Get the values of the desired columns from the second last line
        second_last_column_value1 = second_last_values[column_index1]
        second_last_column_value2 = second_last_values[column_index2]
        
        return (last_column_value1, last_column_value2), (second_last_column_value1, second_last_column_value2)

# Example usage
txt_file = 'TRVDatafile_240323.txt'  # Path to your text file
desired_column_index1 = 0  # Index of the first desired column (0-based)
desired_column_index2 = 1  # Index of the second desired column (0-based)
(last_values, second_last_values) = get_last_and_second_last_values_of_columns(txt_file, desired_column_index1, desired_column_index2)


# Save second_last_values[0] and second_last_values[1] in one variable
second_last_combined ="{} {}".format(second_last_values[0] ,  second_last_values[1])
last_combined = "{} {}".format(last_values[0] , last_values[1])
print('\nsecond_last_combined ',second_last_combined)
print('last_combined',last_combined)





def subtract_datetimes(last_datetime_str, second_last_datetime_str):
    # Parse datetime strings into datetime objects
    last_datetime = datetime.strptime(last_datetime_str, '%d-%m-%Y %H:%M:%S.%f')
    second_last_datetime = datetime.strptime(second_last_datetime_str, '%d-%m-%Y %H:%M:%S.%f')
    
    # Subtract second_last_datetime from last_datetime
    time_difference = last_datetime - second_last_datetime
    
    return time_difference

# Example usage
last_datetime_str = last_combined
second_last_datetime_str = second_last_combined

time_difference = subtract_datetimes(last_datetime_str, second_last_datetime_str)
print("Time difference:", time_difference)



def convert_timedelta_to_seconds(time_delta):
    # Extract total seconds
    total_seconds = time_delta.total_seconds()
    
    return total_seconds

# Example usage
time_formatted = timedelta(hours=0, minutes=3, seconds=11.999983)
time_second_only = convert_timedelta_to_seconds(time_formatted)
print("Time in seconds:", time_second_only)


#___________function to selects the last 60 elements from the desired column of a space-separated text file___________________
def select_last_values(filename, column_index, num_values=60):
  desired_column = []
  
  with open(filename, 'r') as file:
      lines = file.readlines()  # Read all lines into a list

      # Keep track of the last line processed
      last_processed = 0

      for line in lines[-59:]:  # Loop through lines in reverse order
        data = line.strip().split()

        # Check if the desired column index is within bounds
        if 0 <= column_index < len(data):
          desired_column.append(float(data[column_index]))
          last_processed += 1

        # Stop if we have enough values
        if last_processed >= num_values:
          break

  return desired_column[-59:]  # Reverse the list to get the correct order







#________________select the parameters: file, column index, number of values for the function & cal the function__________________________
#filename = "TRVDatafile_new.txt"  # actual filename
filename = "K:\TRVDatafile_240323.txt"  # actual filename

num_values = 59        #  select number of last values

selected_column = (select_last_values(filename, column_index, num_values))
print("Selected last: ", num_values, "values from column:")
print("selected_column: ", selected_column)







#________Define the  array to store: last 21 value of the real temp., difference of real and forecasted temp, current time___________________________
my_20_array = [float('nan')] * 21
my_20_diirr = [float('nan')] * 21
my_time_array = [str('nan')] * 21








#________________convert the 2D dataset of column into 3D format that fit for LSTM model training____________________________

# choose a number of time steps
n_steps =  10 # windows value
n_features = 1

# preparing independent and dependent features
def prepare_data(selected_column, n_features): 
	X, y =[],[]
	for i in range(len(selected_column)):    # number of loop 'i' ranges between 0 to length of time series data
		# find the end of this pattern
		end_ix = i + n_features
		# check if we are beyond the sequence
		if end_ix > len(selected_column)-1:

			break
		
		# gather input and output parts of the pattern
		seq_x, seq_y = selected_column[i:end_ix], selected_column[end_ix]
		X.append(seq_x)
		y.append(seq_y)
		
	return np.array(X), np.array(y)






while True:

        # split into samples & reshape from [samples, timesteps] into [samples, timesteps, features]
        X, y = prepare_data(selected_column, n_steps)
        print('\n X: ', X),print('\n y: ', y)
        X = X.reshape((X.shape[0], X.shape[1], n_features)) # Reshape data into 3D for lstm processing
        print('\n reshape: ', X)
        print('X.shape', X.shape)



        #______________________Build LSTM Model________________________________________________________
        model = Sequential()
        model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        model.add(LSTM(68, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # fit model
        model.fit(X, y, epochs=100, verbose=1)


        #Using model for prediction_____________________________________________________

        temp_input = selected_column[-11:]
        lst_output=[] # createing an empty list to store the "predicted output value"
        
        y_in = []
        i = 0
        time_list = []
        current_time = datetime.now()
        new_time = current_time
        while(i<21): # loop for collecting predicted value in the list: "lst_output=[]" 

                if(len(temp_input)>3):
                        x_input=np.array(temp_input[1:])
                        x_input = x_input.reshape((1, n_steps, n_features))
                        yhat = model.predict(x_input, verbose=0)
                        temp_input.append(yhat[0][0])
                        temp_input=temp_input[1:]
                        lst_output.append(yhat[0][0])
                        
                        # Adding difference to current time
                        
                        new_time =  new_time + time_difference
                        
                        # Append the new timestamp to the list
                        time_list.append(new_time)

                        # Format the list elements to match the format of current_time
                        formatted_time_list = [time.strftime('%Y-%m-%d %H:%M:%S') for time in time_list]


                        #print("New time after adding difference:", new_time.strftime("%Y-%m-%d %H:%M:%S"))
                        
                        i=i+1
                else:
                        x_input = x_input.reshape((1, n_steps, n_features))
                        yhat = model.predict(x_input, verbose=0)
                        temp_input.append(yhat[0][0])
                        lst_output.append(yhat[0][0])
                        
                        new_time =  new_time + time_difference
                        # Append the new timestamp to the list
                        time_list.append(new_time)

                        # Format the list elements to match the format of current_time
                        formatted_time_list = [time.strftime('%Y-%m-%d %H:%M:%S') for time in time_list]


                        i=i+1

        print('lst_output: ',lst_output)
        print('time: ', time)
        #print('\nsuccessive_time_with_difference_of_interval', successive_time_with_difference_of_interval)
        

        df5 = pd.DataFrame()
        df6 = pd.DataFrame()
        df7 = pd.DataFrame(0, index=range(21), columns=['Zero_Column1'])
        df8 = pd.DataFrame(0, index=range(21), columns=['Zero_Column2'])
        df5['lst_output'] =  lst_output[:]
        df6['formatted_time_list'] =  formatted_time_list[:]

        df_forecasted = pd.DataFrame({
                'column1': df6.values.flatten(),
                'column2': df5.values.flatten(),
                'column3': df7.values.flatten(),
                'column4': df8.values.flatten(),
                })

        df_forecasted.to_csv('LSTM_CURRENT_forecast_for_T '+ str(column_index-1)+'.csv',mode='w', index=False, header=False)


      


#_____________________Plot__________________________________________________________________________________
        plt.plot(selected_column , color='b', label='Original dataset', marker = '*')  
        plt.plot(range(60,81),lst_output, color='r', label='Forecasted data', marker = '*')
        plt.title('Deep AI based LINAC data Forecaster - T' + str(column_index - 1), color='r', fontsize='25')
        plt.xlabel('Each 10 represent 30 minutes')
        plt.ylabel('Temperature in Kelvin')
        plt.legend()

        # hold the plot for 3 sec, if pause command is not used then plot will not be displayed
        plt.pause(3) 




        
#_______Loop to read at every 3 min: the latest real value in the CSV file, current time, measure the difference of real & forecasted values ____________
        for counter in range(21):
                print("reading online data" + str(counter))
                selected_column_1 = read_last_values(filename, column_index) #  calling the function to read the latest real value

              
                my_20_array[counter] = float(selected_column_1)   # put = "float(selected_column[0])" for selecting only the latest value in the dataset
                my_20_diirr[counter] = lst_output[counter] - my_20_array[counter] #  lst_output[counter + 60]


                
                # Get current time
                current_time = datetime.now()
                current_time=current_time.strftime("%Y-%m-%d %H:%M:%S")
                my_time_array[counter] = current_time

        
        
                print('At time', my_time_array[counter],'The projected T is', lst_output[counter], 'and real T is', my_20_array[counter],
                      ' difference= ', my_20_diirr[counter])  # lst_output[counter + 60]
                print('\n')

                
                time.sleep(time_second_only) # time delay of 180sec to perform the next iteration of loop






#_____________________Save current current time, realtime values, predicted values, difference in a single CSV file___________________________________
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        df3 = pd.DataFrame()
        df4 = pd.DataFrame()
        df1['my_time_array'] =  my_time_array[:]
        df2['my_20_array'] =  my_20_array[:]
        df3['ypredicted'] = lst_output[:]
        df4[' my_20_diirr'] =  my_20_diirr[:]
           
        df_combined = pd.DataFrame({
                'column1': df1.values.flatten(),
                'column2': df2.values.flatten(),
                'column3': df3.values.flatten(),
                'column4': df4.values.flatten()
                })
            
        df_combined.to_csv('LSTM_TOTAL_appended_real_predicted_error_of_T = '+ str(column_index-1)+'.csv',mode='a', index=False, header=False)

        



           
            
        time.sleep(1) # pause the while loop for 3 seconds
        lst_output.clear() # clear the list of predicted values
        time_list.clear()
        plt.clf() # to clear the plot otherwise the labels gets written multiple times for each loop
        plt.close()
