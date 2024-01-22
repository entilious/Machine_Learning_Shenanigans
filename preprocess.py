import pandas as pd
from sklearn import preprocessing
import csv

import glob


# The code below was used for combining the 9 separate CSV files into one. [2009.csv - 2018.csv]
# The code require the 'glob' module to be imported.

# final_df = pd.DataFrame()
#
# datasets_path = glob.glob('Dataset(s)/*.csv')
#
# for file in datasets_path:
#     df = pd.read_csv(file)
#     final_df = pd.concat([final_df, df], ignore_index=True)
#     print(f'Done concatenating file: {file}')
#
# final_df.to_csv('finalmente.csv')
#
# print("Done saving. Good luck loading the data.")


final_df = pd.DataFrame()

datasets_path = glob.glob('Dataset(s)/*.csv')

for file in datasets_path:
    data = pd.read_csv(file)
    print(f'Loaded file: {file}')
    data = data.drop(['FL_DATE', 'OP_CARRIER', 'OP_CARRIER_FL_NUM', 'ORIGIN', 'DEST', 'CANCELLATION_CODE', 'DIVERTED',
               'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'Unnamed: 27'],
              axis=1)  # Dropping the useless columns. NOTE: Do not mention 'ORIGIN' and 'DEST' columns if being used to train.

    # 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY' selective drop.

    data = data.dropna(subset=['CANCELLED'])  # Deleting the rows where the flight was cancelled.
    data = data.dropna(subset=['CARRIER_DELAY'])  # Deleting the rows where CARRIER_DELAY column is empty.
    data = data.dropna(subset=['WEATHER_DELAY'])  # Deleting the rows where WEATHER_DELAY column is empty.
    data = data.dropna(subset=['NAS_DELAY'])  # Deleting the rows where NAS_DELAY column is empty.
    data = data.dropna(subset=['SECURITY_DELAY'])  # Deleting the rows where SECURITY_DELAY column is empty.
    data = data.dropna(subset=['LATE_AIRCRAFT_DELAY'])  # Deleting the rows where LATE_AIRCRAFT_DELAY column is empty.

    # label_enc = preprocessing.LabelEncoder()

    # The two line below are for when the columns 'ORIGIN' and 'DEST' are being used.

    # data['ORIGIN'] = label_enc.fit_transform(data['ORIGIN'])
    # data['DEST'] = label_enc.fit_transform(data['DEST'])    print(data)

    final_df = pd.concat([final_df, data], ignore_index=True)
    print(f'Done concatenating file: {file}')

# Creating the 'Delayed' column and the instance values.

delay_class = []

for row_no in range(final_df.shape[0]):
    row = final_df.iloc[row_no, :]
    if row['DEP_DELAY'] or row['ARR_DELAY'] > 15:  # Refer to documentation as to why 15 was taken as the limit.
        delay_class.append(1)
    else:
        delay_class.append(0)


final_df['Delayed'] = delay_class

final_df.to_csv('final_dataset.csv')

print('Done saving the file. Good luck.')
