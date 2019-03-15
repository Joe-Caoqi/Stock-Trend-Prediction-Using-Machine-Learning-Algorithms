import pandas as pd
import sqlite3
import numpy
'''
Notes
    ☑️missing value checked 
        amdRaw_df.isnull().any()
        googleRaw_df.isnull().any()
    ☑️Transfer Date column into datetime type
    ❗️Create Features by indicators 
    ⭕️❌❓
'''
amd_raw_df=pd.read_csv('raw_data_amdgoogle/AMD.csv')
google_raw_df=pd.read_csv('raw_data_amdgoogle/GOOGL.csv')
amd_raw_df['Date']=pd.to_datetime(amd_raw_df['Date'],format='%Y-%m-%d')
google_raw_df['Date']=pd.to_datetime(google_raw_df['Date'],format='%Y-%m-%d')
engine = sqlite3.connect('stock_price_DB')
amd_raw_df.to_sql('amd_raw',con=engine,if_exists='replace',index=False)
google_raw_df.to_sql('google_raw',con=engine,if_exists='replace',index=False)
