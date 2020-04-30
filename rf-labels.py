#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Definition of the labels used in the Random Forest model
 
labels are defines as following:
    -1: Dowmward movement
     0: Stationary
    +1: Upward movement 

The labels are defined over 2 time horizons:
    - 10 seconds ahead
    - 20 seconds ahead

@author: thertrader@gmail.com
@Date: Mar 2020
"""

import pandas as pd
import os
import numpy as np
from datetime import datetime


# =============================================================================
# 1 - BASIC PARAMETERS
# =============================================================================
os.chdir(r'/home/arno/work/research/lobster/data')

nlevels = 10

col = ['Ask Price ','Ask Size ','Bid Price ','Bid Size ']

theNames = []
for i in range(1, nlevels + 1):
    for j in col:
        theNames.append(str(j) + str(i))

tickers = ['TSLA'] 

#----- Lists of messages files and order book files 
theFiles = [] 
theOrderBookFiles = []
theMessagesFiles = []
 
for tk in tickers:
    # tk = 'INTC'
    os.chdir(r"/home/arno/work/research/lobster/data/" + tk)
    theFiles.extend(sorted(os.listdir()))
    theOrderBookFiles.extend([sl for sl in theFiles if "_orderbook" in sl])
    theMessagesFiles.extend([sk for sk in theFiles if "_message" in sk])
    theFiles = []


# =============================================================================
# 2 - Y LABELS FUNCTION 
# =============================================================================
# outputFileName = '_Y10Sec.csv'
# fwdTimeLength = 10
# f = 'TSLA_2015-01-02_34200000_57600000_orderbook_10.csv'
os.chdir(r'/home/arno/work/research/lobster/data')
def labels(outputFileName, fwdTimeLength):    
    
    for f in theOrderBookFiles:
        print(f + ' **** ' + datetime.now().strftime("%H:%M:%S"))
        
        theName = f[0:15] + outputFileName
                
        if (f[0:4] == 'TSLA'):
            os.chdir(r'/home/arno/work/research/lobster/data/TSLA') 
        
            theMessageFile = f[0:15] + '_34200000_57600000_message_10.csv'
            
            #---- Method using Numpy array, Loop and reduced array size to speed up things big time. This is how it's done:
            mf = pd.read_csv(theMessageFile, names = ['timeStamp','EventType','Order ID','Size','Price','Direction'], float_precision='round_trip')
            dfTimeStamp = mf['timeStamp']
            timeStampInSeconds = dfTimeStamp.round(0) # used to define max lookback period 
            maxLookAhead = sum(timeStampInSeconds.value_counts().iloc[:(fwdTimeLength + 2)]) # used to define max lookback period (should be iloc[:11] but iloc[:12] is safer)    
            timeStamp = mf['timeStamp'].to_frame().values 
            
            start = [] 
             
#            start_time = time.time() # For profiling only    
            for i in range(len(timeStamp)):
                if i < (len(timeStamp) - maxLookAhead):
                    a = i + maxLookAhead
                    bb = np.column_stack((np.array(dfTimeStamp.iloc[i:a].index),timeStamp[i:a,0])) 
                    theIndexValue = bb[abs(bb[:,1] - (timeStamp[i,0] + fwdTimeLength)).argmin(),0]
                    start.append(int(theIndexValue))
                    
                elif i >= (len(timeStamp) - maxLookAhead):    
                    bb = np.column_stack((np.array(dfTimeStamp.iloc[i:].index),timeStamp[i:,0])) 
                    theIndexValue = bb[abs(bb[:,1] - (timeStamp[i,0] + fwdTimeLength)).argmin(),0]
                    start.append(int(theIndexValue))
            
            stop = list(range(len(timeStamp)))
#            print("--- %s seconds ---" % round(time.time() - start_time,2))  # For profiling only           
            
            theDataFile = f[0:15] + '_34200000_57600000_orderbook_10.csv'
            df = pd.read_csv(theDataFile, names = theNames)
            dfArray = df.values
            
            midRtn = np.array([(((dfArray[stop[j],0] + dfArray[stop[j],2])/2) / ((dfArray[start[j],0] + dfArray[start[j],2])/2)) - 1 for j in range(len(timeStamp))])
            midRtn = np.where(midRtn > 0,1,np.where(midRtn < 0,-1,0)) 
            dfToExport = pd.concat([pd.Series(timeStamp[:,0]),pd.Series(midRtn)],axis=1)
            dfToExport.columns = ['timeStamp', 'label']
                                     
            dfToExport.to_csv(r'/home/arno/work/research/lobster/data/features/' + theName, header = True, index = False)
            
            del dfToExport,midRtn,mf

# =============================================================================
# 3 - RUN LABELS FUNCTION 
# =============================================================================
labels(outputFileName = '_Y10Sec.csv', fwdTimeLength = 10)
labels(outputFileName = '_Y20Sec.csv', fwdTimeLength = 20)        
    



    
    

    
