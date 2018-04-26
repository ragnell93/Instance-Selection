"""
Created on 11/4/2018
@author: Antonio Alvarez
"""

import sys
import pandas as pd
import numpy as np 
import copy
from utils import *

if __name__ == '__main__':

    table = pd.read_csv("./data/" + sys.argv[1],na_values=['NA','NaN','nan','na'],header=None)

    #Fill missing values
    for i in range(0,len(table.columns)):
        #if value is numeric fill with mean 
        if ((table.iloc[:,i].dtype == np.float64) or (table.iloc[:,i].dtype == np.int64)):
            table.iloc[:,i].fillna(value=table.iloc[:,i].mean(),inplace=True)
        #if value is categorical fill with mode
        else:
            table.iloc[:,i].fillna(value=table.iloc[:,i].mode()[0],inplace=True)


    index,index2,table = reorganize(table)

    #trasnform objects into categories
    for i in range(index2,len(table.columns)):
        table.iloc[:,i] = table.iloc[:,i].astype('category')

    table2 = copy.deepcopy(table)

    #Doing normalization for euclidean distance
    for i in range(0,index2):
        table2.iloc[:,i] = (table2.iloc[:,i]-(table2.iloc[:,i].mean()))/table2.iloc[:,i].std()

    results = table.iloc[:,-1]
    cat_columns = table.select_dtypes(['category']).columns
    table2.iloc[:,-1] = table2.iloc[:,-1].astype('category')
    results2 = table2.iloc[:,-1]

    table2 = pd.get_dummies(table2,dummy_na = False, columns = cat_columns[0:-1])
    table[cat_columns] = table[cat_columns].apply(lambda x: x.cat.codes)
    dict( enumerate(table2[results2.name].cat.categories) )
    table2[results2.name] = table2[results2.name].cat.codes

    results = table[results.name]
    results2 = table2[results2.name]
    del table[results.name]
    del table2[results2.name]

    header2 = open("./euclidean/"+sys.argv[1]+".header",'w')
    header2.write("{:d} {:d} \n".format(table2.iloc[:,0].count(),len(table2.columns)))

    table2.to_csv(path_or_buf="./euclidean/"+sys.argv[1]+".data",sep=" ",header=False, index=False)
    results2.to_csv(path="./euclidean/"+sys.argv[1]+".results",sep=" ",header=False, index=False)

    header2.close()
