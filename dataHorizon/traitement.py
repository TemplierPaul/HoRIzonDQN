import numpy as np
import pandas as pd
import os

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
columns = ['remaining_time', 'robot_autonomous', 'alarm', 'robot_x', 'robot_y', 'robot_angle', 'trees_state',
           'battery_level', 'temperature', 'robot_tank_water_level', 'ground_tank_water_level', 'leaks_number',
           'used_keys', 'clicks', 'others', 'keyboard_shortcuts']

log = open('out/log.txt', 'w')
# print (os.listdir('.'))

for nom in os.listdir('.'):
    if nom.endswith('.csv'):
        print(nom)
        file = pd.read_csv(nom)
        mydata = pd.DataFrame(file)

        # corrections of names
        mydata.rename(columns={'remainingtime': 'remaining_time'}, inplace=True)
        mydata.rename(columns={' autonomousrobot': 'robot_autonomous'}, inplace=True)
        mydata.rename(columns={' alarm': 'alarm'}, inplace=True)
        mydata.rename(columns={' robotx': 'robot_x'}, inplace=True)
        mydata.rename(columns={' roboty': 'robot_y'}, inplace=True)
        mydata.rename(columns={' robotangle': 'robot_angle'}, inplace=True)
        mydata.rename(columns={' treesstate': 'trees_state'}, inplace=True)
        mydata.rename(columns={' batterylevel': 'battery_level'}, inplace=True)
        mydata.rename(columns={' temperature': 'temperature'}, inplace=True)
        mydata.rename(columns={' waterlevel': 'robot_tank_water_level'}, inplace=True)
        mydata.rename(columns={' groundtankwaterlevel': 'ground_tank_water_level'}, inplace=True)
        mydata.rename(columns={' leaksnumber': 'leaks_number'}, inplace=True)
        mydata.rename(columns={' usedkeys': 'used_keys'}, inplace=True)
        mydata.rename(columns={' clicks ': 'clicks'}, inplace=True)
        mydata.rename(columns={' others': 'others'}, inplace=True)
        mydata.rename(columns={' keyboardshortcuts ': 'keyboard_shortcuts'}, inplace=True)

        #        print (mydata)

        # change first occurence of temperature
        mydata.at[0, 'temperature'] = 50

        # if the file has been generated before that we record (useless keys and the keyboard shortcuts)
        # we add columns with only '-2' values (for 'not available')
        if len(mydata.columns) == 14:
            mydata['others'] = pd.Series([-2] * len(mydata.index), index=mydata.index)
            mydata['keyboard_shortcuts'] = pd.Series(['-2'] * len(mydata.index), index=mydata.index)

        #        print (mydata)
        # reindex columns
        mydata.columns = columns
        #        mydata = mydata.reindex(columns=columns)

        # cast each column as string
        mydata[columns] = mydata[columns].astype(str)
        #        print (mydata)

        # remove last occurence of battery level when the penultimate one is 0
        if mydata['battery_level'][len(mydata.index) - 2] == 0:
            mydata.drop(mydata.index[[len(mydata.index) - 1]], inplace=True)

        # change first occurence of temperature
        mydata.at[0, 'temperature'] = 50

        # correction of bug in column ' treesstate' and ' leaksnumber'
        mydata['trees_state'] = mydata['trees_state'].str.replace('0.0', 'e-')
        mydata['leaks_number'] = mydata['leaks_number'].str.replace('0.0', 'e-')
        mydata['used_keys'] = mydata['used_keys'].str.replace('0.0', 'e-')

        # Add new columns
        for i in range(0, 9):
            mydata['tree' + str(i)] = pd.Series([0] * len(mydata.index), index=mydata.index)

        for i in range(0, 9):
            mydata['leak' + str(i)] = pd.Series([0] * len(mydata.index), index=mydata.index)

        for i in range(0, 10):
            mydata['space' + str(i)] = pd.Series([0] * len(mydata.index), index=mydata.index)
            mydata['left' + str(i)] = pd.Series([0] * len(mydata.index), index=mydata.index)
            mydata['right' + str(i)] = pd.Series([0] * len(mydata.index), index=mydata.index)
            mydata['front' + str(i)] = pd.Series([0] * len(mydata.index), index=mydata.index)
            mydata['back' + str(i)] = pd.Series([0] * len(mydata.index), index=mydata.index)

        mydata['wrench'] = pd.Series([0] * len(mydata.index), index=mydata.index)
        mydata['minus'] = pd.Series([0] * len(mydata.index), index=mydata.index)
        mydata['plus'] = pd.Series([0] * len(mydata.index), index=mydata.index)
        mydata['push'] = pd.Series([0] * len(mydata.index), index=mydata.index)
        mydata['removeAlarm'] = pd.Series([0] * len(mydata.index), index=mydata.index)
        mydata['clickLeak'] = pd.Series([0] * len(mydata.index), index=mydata.index)

        mydata['otherkeyUp'] = pd.Series([0] * len(mydata.index), index=mydata.index)
        mydata['otherkeyDown'] = pd.Series([0] * len(mydata.index), index=mydata.index)
        mydata['otherClick'] = pd.Series([0] * len(mydata.index), index=mydata.index)

        mydata['keyboard'] = pd.Series([0] * len(mydata.index), index=mydata.index)

        # Outpu
        mydata['man_auto'] = pd.Series([0] * len(mydata.index), index=mydata.index)
        mydata['auto_man'] = pd.Series([0] * len(mydata.index), index=mydata.index)

        for i in range(0, 8):
            mydata['alarm' + str(i)] = pd.Series([0] * len(mydata.index), index=mydata.index)

        mydata['score'] = pd.Series([0] * len(mydata.index), index=mydata.index)

        onfire = 0
        auto = 0

        for ligne in mydata.iterrows():
            info = ligne[1]

            #            print (info)

            if int(info[1]) > auto:
                auto = int(info[1])
                mydata.at[ligne[0], 'man_auto'] = 1
            elif int(info[1]) < auto:
                auto = int(info[1])
                mydata.at[ligne[0], 'auto_man'] = 1

            if info[2] != '-1':
                mydata.at[ligne[0], 'alarm' + str(info[2])] = 1

            trees = info[6].split('-')
            #            print (trees)
            for i in range(0, 9):
                if trees[i] == 'true':
                    mydata.at[ligne[0], 'tree' + str(i)] = 1

            if trees.count('true') < onfire:
                mydata.at[ligne[0], 'score'] = 1
            onfire = trees.count('true')

            leaks = info[6].split('-')
            for i in range(0, 9):
                if leaks[i] == 'true':
                    mydata.at[ligne[0], 'leak' + str(i)] = 1

            # usedkeys
            if info[12] != ' -1' and info[12] != '-1' :
                keys = info[12]
                if keys[0] == ' ':
                    keys = keys[1:]
                keys = keys.split('-')
                for i in range(0, len(keys)):
                    mydata.at[ligne[0], keys[i] + str(i)] = 1

            clicks = info[13]
            if clicks[0] == ' ':
                clicks = clicks[1:]
            if info[13] != '-1' :
                clicks = clicks.split('-')
                for i in ['wrench', 'minus', 'plus', 'push', 'removeAlarm']:
                    mydata.at[ligne[0], i] = clicks.count(i)
                for i in range(0, 9):
                    mydata.at[ligne[0], 'clickLeak'] += clicks.count('clickLeak' + str(i))

            other = info[14]
            if other[0] == ' ':
                other = other[1:]
            if info[14] != '-1' :
                other = other.split('-')
                for i in ['otherkeyUp', 'otherkeyDown', 'otherClick']:
                    mydata.at[ligne[0], i] = other.count(i)

            keyboard = info[15]
            if keyboard[0] == ' ':
                keyboard = keyboard[1:]
            if info[15] != '-1' :
                keyboard = keyboard.split('-')
                for i in ['keyboard']:
                    mydata.at[ligne[0], i] = keyboard.count(i)

        mydata = mydata.drop(
            ['trees_state', 'clicks', 'leaks_number', 'alarm', 'used_keys', 'others', 'keyboard_shortcuts'], axis=1)

        nb = (os.listdir('.')).index(nom)
        name = 'out/up_' + str(nb) + '.csv'
        mydata.to_csv(name, index=False)
        print(name + ' ok')
        log.write(nom + ' -> ' + name + '\n')

log.close()
