import pandas as pd
import os

frame = pd.DataFrame()
list_files = []
for nom in os.listdir('.'):
    if nom.endswith('.csv'):
        print(nom)
        file = pd.read_csv(nom)
        list_files.append(file)
frame=pd.concat(list_files)
frame.to_csv('dataSet/dataSet.csv',index=False)
