"""
Created on 11/4/2018
@author: Antonio Alvarez
"""
import math as m
import copy
import pandas as pd
import numpy as np 

def reorganize(df):

    #Reorganize the data frame putting the numerical values first and the categorical later. 
    flo = []
    integer = []
    cat = []
    for i in range(0,len(df.columns)-1):
        if (df.iloc[:,i].dtype == np.float64):
            flo += [df.iloc[:,i].name]
        elif (df.iloc[:,i].dtype == np.int64):
            integer += [df.iloc[:,i].name]
        else: 
            cat += [df.iloc[:,i].name]

    mark = len(flo) #where the categorical begins
    header = flo + integer
    mark2 = len(header)
    header += cat + [df.iloc[:,-1].name]
    newdf = df[header]

    return mark,mark2,newdf

def ivdmPrep(index1,df,results):

    #Preparations for IVDM distance
    numAttr = len(df.columns)
    numClass = results.unique().size
    s = max(numClass,5) # number of intervals
    width = []
    for i in range(0,index1):
        width += [(df.iloc[:,i].max() - df.iloc[:,i].min())/s]

    #discretization
    discTable = copy.deepcopy(df)
    for i in range(0,df.iloc[:,0].count()):
        for j in range(0,index1):
            if (discTable.iloc[i,j] == df.iloc[:,j].max()):
                discTable.iloc[i,j] = s
            else:
                discTable.iloc[i,j] = m.floor((df.iloc[i,j]-df.iloc[:,j].min())/width[j])+1
                
    #Getting the Conditional probabilities
    nxac = np.zeros(shape=(s,numAttr,numClass),dtype=np.int64)
    pxac = np.zeros(shape=(s,numAttr,numClass),dtype=np.float64)
    nxa = np.zeros(shape=(s,numAttr),dtype=np.int64)

    for i in range(0,df.iloc[:,0].count()):
        for j in range(0,df.iloc[0,:].count()):
            nxac[discTable.iloc[i,j]-1,j,results[i]] += 1
            nxa[discTable.iloc[i,j]-1,j] += 1

    for i in range(0,s):
        for j in range(0,numAttr):
            for k in range(0,numClass):
                if (nxa[i,j] == 0):
                    pxac[i,j,k] = 0
                else:
                    pxac[i,j,k] = nxac[i,j,k]/nxa[i,j]

    return pxac,width