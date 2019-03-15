import pandas as pd
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
amdRaw_df=pd.read_csv('raw_data_amdgoogle/AMD.csv')
googleRaw_df=pd.read_csv('raw_data_amdgoogle/GOOGL.csv')
amdRaw_df['Date']=pd.to_datetime(amdRaw_df['Date'],format='%Y-%m-%d')
googleRaw_df['Date']=pd.to_datetime(amdRaw_df['Date'],format='%Y-%m-%d')
