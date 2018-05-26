import pandas as pd
import numpy as np
EXP_FILE=pd.read_csv('test.csv')
N_ACTIONS = 16
N_STATES = EXP_FILE.columns.size-3
N_EXP=EXP_FILE.iloc[:,0].size
type(EXP_FILE)
#print(EXP_FILE.dtypes)
rang=EXP_FILE.loc[0]
#print(rang)
print(EXP_FILE.ix[1, 2:10])
rang_arr=np.array(EXP_FILE.ix[1, 2:10])
print("rang_arr=",rang_arr)
length = EXP_FILE.columns.size
print('length=',length)
s=EXP_FILE.ix[2, 2:N_STATES+2]
print("N_state=",N_STATES)
print('state=',s)
r = EXP_FILE.ix[3, N_STATES+2]
print('r=',r)
