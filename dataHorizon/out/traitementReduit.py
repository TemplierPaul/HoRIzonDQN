import numpy as np
import pandas as pd
import os
import math

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

log = open('outReduit/log.txt', 'w')
exempledata=pd.read_csv('up_0.csv')
nbCol = exempledata.columns.size
# print (os.listdir('.'))

for nom in os.listdir('.'):
    if nom.endswith('.csv'):
        print(nom)
        file = pd.read_csv(nom)
        mydata = pd.DataFrame(file)


        for ligne in mydata.iterrows():
            info = ligne[1]

            for i in range(0, 87):
                mydata.at[ligne[0] - 1, i + nbCol + 1] = info[i]
            #accumulate tree numbers and leaks number
            # for word in ('tree','leak'):
            #     for i in range (1,9):
            #         mydata.at[ligne[0], word+'0']=mydata.at[ligne[0], word+'0']+mydata.at[ligne[0], word+str(i)]
            # # accumulate clicks keys
            # for word in ('space','left','right','front','back'):
            #     for i in range (1,10):
            #         mydata.at[ligne[0], word+'0']=mydata.at[ligne[0], word+'0']+mydata.at[ligne[0], word+str(i)]
            mydata.at[ligne[0], 'robot_x']=int(mydata.at[ligne[0], 'robot_x'])
            mydata.at[ligne[0], 'robot_y'] = int(mydata.at[ligne[0], 'robot_y'])
            mydata.at[ligne[0], 'robot_angle'] = int(int(mydata.at[ligne[0], 'robot_angle'])*6.0/math.pi)

        #delete columns
        for word in ('tree', 'leak'):
            for i in range(0, 9):
                mydata=mydata.drop([word+str(i)],axis=1)

        for word in ('space', 'left', 'right', 'front', 'back','otherkey','otherkeyDown','otherkeyUp'):
            for i in range(0, 40):
                try:
                    mydata =mydata.drop([word + str(i)], axis=1)
                except ValueError:
                    continue
        mydata = mydata.drop(['remaining_time','temperature','battery_level','robot_tank_water_level','ground_tank_water_level','wrench','minus','plus','push','removeAlarm','clickLeak','otherkeyUp','otherkeyDown','otherClick','keyboard'],axis=1)
        mydata = mydata.drop(mydata.tail(2).index)
        nb = (os.listdir('.')).index(nom)
        name = 'outReduit/up_' + str(nb) + '.csv'
        mydata.to_csv(name, index=False)
        print(name + ' ok')
        log.write(nom + ' -> ' + name + '\n')

log.close()
