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
    results = table.iloc[:,-1]
    cat_columns = table.select_dtypes(['category']).columns
    table2.iloc[:,-1] = table2.iloc[:,-1].astype('category')
    results2 = table2.iloc[:,-1]

    table2 = pd.get_dummies(table,dummy_na = False, columns = cat_columns[0:-1])
    table[cat_columns] = table[cat_columns].apply(lambda x: x.cat.codes)
    dict( enumerate(table2[results2.name].cat.categories) )
    table2[results2.name] = table2[results2.name].cat.codes

    results = table[results.name]
    results2 = table2[results2.name]
    del table[results.name]
    del table2[results2.name]

    prob, width = ivdmPrep(index,table,results)

    #Write headers of files, data file and probability file
    header1 = open("./ivdm/"+sys.argv[1]+".header",'w')
    probFile = open("./ivdm/"+sys.argv[1]+".prob",'w')

    header1.write("{:d} {:d} \n".format(table.iloc[:,0].count(),len(table.columns)))
    header1.write("{:d} {:d} {:d} \n".format(prob.shape[0],prob.shape[1],prob.shape[2]))
    header1.write("{:d}\n".format(index))
    
    for j in range(0,table.iloc[0,:].count()):
        header1.write("{:f} ".format(table.iloc[:,j].max()))
    header1.write("\n")
    for j in range(0,table.iloc[0,:].count()):
        header1.write("{:f} ".format(table.iloc[:,j].min()))
    header1.write("\n")
    for i in range(0,len(width)):
        header1.write("{:f} ".format(width[i]))

    table.to_csv(path_or_buf="./ivdm/"+sys.argv[1]+".data",sep=" ",header=False,index=False)
    results.to_csv(path="./ivdm/"+sys.argv[1]+".results",sep=" ",header=False, index=False)

    for i in range (0,prob.shape[0]):
        for j in range (0,prob.shape[1]):
            for k in range (0,prob.shape[2]):
                probFile.write("{:1.4f} ".format(prob[i,j,k]))
            probFile.write("\n")
        probFile.write("\n")

    header2 = open("./euclidean/"+sys.argv[1]+".header",'w')
    header2.write("{:d} {:d} \n".format(table2.iloc[:,0].count(),len(table2.columns)))
    table2.to_csv(path_or_buf="./euclidean/"+sys.argv[1]+".data",sep=" ",header=False, index=False)
    results2.to_csv(path="./euclidean/"+sys.argv[1]+".results",sep=" ",header=False, index=False)


    header1.close()
    header2.close()
    probFile.close()