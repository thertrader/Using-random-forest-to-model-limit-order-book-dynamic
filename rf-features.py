#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Definition of features used in the random forest modeling 

1. Basic parameters
2. Features definition
   - 2.1 START & STOP TIMESTAMPS: This is ran once for all for each file. Look for the timestamp 5sec backward. This is used for deciles definition.
   - 2.2 IMBALANCE: Absolute levels & deciles ranking compared to last 5min
   - 2.3 MID PRICE, B/A PRICE SPREAD, B/A VOLUME SPREAD: Absolute levels & deciles ranking compared to last 5min
   - 2.4 PRICE DIFFERENCES ACCROSS LEVELS: Bid & Ask differences between order book levels (1 to 10). Absolute levels & deciles ranking compared to last 5 min
   - 2.5 PRICE AVERAGE & VOLUME AVERAGE: Mean calculated across order book levels. Deciles ranking compared to last 5 min
   - 2.6 ACCUMULATED DIFFERENCES PRICE & VOLUME: Sum of all bid minus sum of all asks across levels. Deciles ranking compared to last 5 min
   - 2.7 BID, ASK, BID SIZE & ASK SIZE DERIVATIVEs: Last compared to price 1 sec. ago. Deciles ranking compared to last 5 sec. 
   - 2.8 AVERAGE TRADE INTENSITY: Count events of type 1,2,3 (see definition above) over the last second and deciles ranking compared to last 5 min
   - 2.9 RELATIVE TRADE INTENSITY 10s: Count events of type 1,2,3 over the last 10 sec (Intermediary step)
   - 2.10 RELATIVE TRADE INTENSITY 900s: Count events of type 1,2,3 over the last 900 sec. (Intermediary step)
   - 2.11 RELATIVE TRADE INTENSITY: Relative count events 10 sec./900 sec. Deciles ranking compared to last 5 min

EVENT TYPE (see LOBSTER website - [https://lobsterdata.com/](https://lobsterdata.com/)):
1. New limit order 
2. Partial deletion of limit order
3  Total cancellation of limit order

        
@author: thertrader@gmail.com
@Date: Mar 2020
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime


# =============================================================================
# 1 - Basic parameters
# =============================================================================
os.chdir(r'/home/arno/work/research/lobster/data')

tickers = ['TSLA'] 

nlevels = 10

col = ['Ask Price ','Ask Size ','Bid Price ','Bid Size ']

theNames = []
for i in range(1, nlevels + 1):
    for j in col:
        theNames.append(str(j) + str(i))

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
# 2 - Features definition
# =============================================================================
#----- 2.1 Define Start & Stop timestamp for all files (run it once for all for each file)
for f in theOrderBookFiles:
    print(f + ' **** ' + datetime.now().strftime("%H:%M:%S")) # used for prtotyping only
    
    theName = f[0:15] + '_startAndStop.csv'
     
    if (f[0:4] == 'INTC'):
        os.chdir(r'/home/arno/work/research/lobster/data/INTC')
    
    if (f[0:4] == 'TSLA'):
        os.chdir(r'/home/arno/work/research/lobster/data/TSLA') 
    
    dt = pd.read_csv(f, names = theNames)
    #dt.head(5)
    
    theMessageFile = f[0:15] + '_34200000_57600000_message_10.csv'
    mf = pd.read_csv(theMessageFile, names = ['timeStamp','EventType','Order ID','Size','Price','Direction'], float_precision='round_trip')
    dfTimeStamp = mf['timeStamp']
    theIndex = dfTimeStamp.index.values 
    timeStampInSeconds = dfTimeStamp.round(0)  
    maxLookBack = sum(timeStampInSeconds.value_counts().iloc[:6])      
    timeStamp = dfTimeStamp.to_frame().values  
    
    start = [] 
    
    for j in range(len(timeStamp)):
        if (j == 0):
            start.append(j)
            
        elif (j <= maxLookBack):
            theIndexValue = int(abs(timeStamp[:j] - (timeStamp[j] - 5)).argmin())
            start.append(theIndexValue)
            
        elif (j > maxLookBack):
            a = j - maxLookBack
            bb = np.column_stack((theIndex[a:j],timeStamp[a:j])) 
            theIndexValue = int(bb[abs(bb[:,1] - (timeStamp[j] - 5)).argmin(),0])
            start.append(theIndexValue)
    
    stop = list(range(len(timeStamp)))
    
    startAndStop = pd.concat([dfTimeStamp, pd.Series(start), pd.Series(stop)], axis= 1)
    startAndStop.columns = ['timeStamp','start','stop']
    
    startAndStop.to_csv(r'/home/arno/work/research/lobster/data/features/' + theName, header = True, index = False)



#----- 2.2 Imbalance Level and Derivative per level
for f in theOrderBookFiles:
    print(f + ' **** ' + datetime.now().strftime("%H:%M:%S")) # used for prtotyping only
    
    theNameLevel = f[0:15] + '_imbalanceLevel_' + str(nlevels) + '.csv'
    theNameDerivative = f[0:15] + '_imbalanceDerivative_' + str(nlevels) + '.csv'
    
    if (f[0:4] == 'INTC'):
        os.chdir(r'/home/arno/work/research/lobster/data/INTC')
    
    if (f[0:4] == 'TSLA'):
        os.chdir(r'/home/arno/work/research/lobster/data/TSLA') 
    
    #--- Open Order Book file
    theOrderBookFile = pd.read_csv(f, names = theNames)
        
    #--- Open Messages file
    theMessageFile = f[0:15] + '_34200000_57600000_message_10.csv'
    mf = pd.read_csv(theMessageFile, names = ['timeStamp','EventType','Order ID','Size','Price','Direction'], float_precision='round_trip')
    timeStamp = mf['timeStamp'].to_frame().values
    
    #--- Open Start & Stop file
    os.chdir(r'/home/arno/work/research/lobster/data/features/')
    sSFile = [g for g in os.listdir() if ((g[-17:] == '_startAndStop.csv') and (g[:15] == f[:15]))]
    startStopFile = pd.read_csv(sSFile[0], names = ['timeStamp','start','stop'], header = 0)
    start =  np.array(startStopFile['start'])
    stop = np.array(startStopFile['stop'])
    
    imbLevel = pd.DataFrame()
    imbDerivative = pd.DataFrame()
       
    for i in range(1,nlevels+1):
        #--- Define Imbalance
        nameOne = 'Bid Size '+ str(i)
        nameTwo = 'Ask Size '+ str(i)
        colName = 'imb '+ str(i)
        lev = (theOrderBookFile[nameTwo] - theOrderBookFile[nameOne])/(theOrderBookFile[nameTwo] + theOrderBookFile[nameOne])
        theLevel = round(10 * (lev - lev.min()) / (lev.max() - lev.min()),0) # Deciles
        lev = np.array(lev)

        #--- Define Imbalance Derivative
#        print('**** ' + datetime.now().strftime("%H:%M:%S"))
        theDerivative = [round(stats.percentileofscore(lev[start[k]:k],lev[k])/10,0) for k in range(len(timeStamp))] # Deciles
#        print('**** ' + datetime.now().strftime("%H:%M:%S"))
        
        imbLevel = pd.concat([imbLevel,theLevel],axis=1)
        imbDerivative = pd.concat([imbDerivative,pd.Series(theDerivative)],axis=1)
    
    imbLevel.columns = ['imbLevel1','imbLevel2','imbLevel3','imbLevel4','imbLevel5','imbLevel6','imbLevel7','imbLevel8','imbLevel9','imbLevel10']
    imbLevel.to_csv(r'/home/arno/work/research/lobster/data/features/' + theNameLevel, header = True, index = False)
    
    imbDerivative.columns = ['imbDer1','imbDer2','imbDer3','imbDer4','imbDer5','imbDer6','imbDer7','imbDer8','imbDer9','imbDer10']
    imbDerivative.to_csv(r'/home/arno/work/research/lobster/data/features/' + theNameDerivative, header = True, index = False)
    
    os.chdir(r'/home/arno/work/research/lobster/data')       
    
    
    
#----- 2.3 Mid price, Spread, Volume Spread derivatives
for f in theOrderBookFiles:
    print(f + ' **** ' + datetime.now().strftime("%H:%M:%S")) # used for prtotyping only
    theName = f[0:15] + '_misc_' + str(nlevels) + '.csv'
    
    if (f[0:4] == 'INTC'):
        os.chdir(r'/home/arno/work/research/lobster/data/INTC')
    
    if (f[0:4] == 'TSLA'):
        os.chdir(r'/home/arno/work/research/lobster/data/TSLA') 
    
    dt = pd.read_csv(f, names = theNames)
    
    #--- Open Start & Stop file
    os.chdir(r'/home/arno/work/research/lobster/data/features/')
    sSFile = [g for g in os.listdir() if ((g[-17:] == '_startAndStop.csv') and (g[:15] == f[:15]))]
    startStopFile = pd.read_csv(sSFile[0], names = ['timeStamp','start','stop'], header = 0)
    start = np.array(startStopFile['start'])
    stop = np.array(startStopFile['stop'])
    
    aa = pd.DataFrame()
    newNames = []
    
    for i in range(1,nlevels+1):
        newNames.extend(['midDer'+ str(i),'spreadDer'+ str(i),'volumeDer'+ str(i)])  
        
        nameOne = 'Bid Price '+ str(i)
        nameTwo = 'Ask Price '+ str(i)
        nameThree = 'Bid Size '+ str(i)
        nameFour = 'Ask Size '+ str(i)
        
        #--- Define Levels
        mid = np.array((dt[nameTwo] + dt[nameOne])/2)
        spread = np.array(dt[nameTwo] - dt[nameOne])
        volumeSpread = np.array(dt[nameFour] - dt[nameThree])
        
        #--- Define Derivatives
        theMidDerivative = [round(stats.percentileofscore(mid[start[k]:k],mid[k])/10,0) for k in range(len(startStopFile))] # Deciles
        theSpreadDerivative = [round(stats.percentileofscore(spread[start[k]:k],spread[k])/10,0) for k in range(len(startStopFile))] # Deciles
        theVolumeSpreadDerivative = [round(stats.percentileofscore(volumeSpread[start[k]:k],volumeSpread[k])/10,0) for k in range(len(startStopFile))] # Deciles

        aa = pd.concat([aa,pd.Series(theMidDerivative),pd.Series(theSpreadDerivative),pd.Series(theVolumeSpreadDerivative)],axis=1)
 
    aa = pd.concat([startStopFile['timeStamp'],aa],axis=1)
    newNames.insert(0,'timeStamp')
    aa.columns = newNames
    aa.to_csv(r'/home/arno/work/research/lobster/data/features/' + theName, header = True, index = False)

    os.chdir(r'/home/arno/work/research/lobster/data')



#----- 2.4 Price differences
for f in theOrderBookFiles:
    print(f + ' **** ' + datetime.now().strftime("%H:%M:%S")) # used for prtotyping only
    theName = f[0:15] + '_priceDiff_' + str(nlevels) + '.csv'
    
    if (f[0:4] == 'INTC'):
        os.chdir(r'/home/arno/work/research/lobster/data/INTC')
    
    if (f[0:4] == 'TSLA'):
        os.chdir(r'/home/arno/work/research/lobster/data/TSLA') 
    
    dt = pd.read_csv(f, names = theNames)
    
    #--- Open Start & Stop file
    os.chdir(r'/home/arno/work/research/lobster/data/features/')
    sSFile = [g for g in os.listdir() if ((g[-17:] == '_startAndStop.csv') and (g[:15] == f[:15]))]
    startStopFile = pd.read_csv(sSFile[0], names = ['timeStamp','start','stop'], header = 0)
    start = np.array(startStopFile['start'])
    stop = np.array(startStopFile['stop'])
    
    newNames = []
    cc = pd.DataFrame()
    
    for i in range(1,nlevels):
        # i = 1
        nameOne = 'Ask Price '+ str(i)
        nameTwo = 'Ask Price '+ str(i+1)
        aa = np.array(abs(dt[nameTwo] + dt[nameOne]))
        theAskDiffDerivative = [round(stats.percentileofscore(aa[start[k]:k],aa[k])/10,0) for k in range(len(startStopFile))] # Deciles
        
        nameTree = 'Bid Price '+ str(i)
        nameFour = 'Bid Price '+ str(i+1)
        bb = np.array(abs(dt[nameFour] + dt[nameTree]))
        theBidDiffDerivative = [round(stats.percentileofscore(bb[start[k]:k],bb[k])/10,0) for k in range(len(startStopFile))] # Deciles
        
        newNames.extend(['AskPriceDiff '+ str(i),'BidPriceDiff '+ str(i)])
        
        cc = pd.concat([cc,pd.Series(theAskDiffDerivative),pd.Series(theBidDiffDerivative)],axis=1)
    
    cc = pd.concat([startStopFile['timeStamp'],cc],axis=1)
    newNames.insert(0,'timeStamp')
    cc.columns = newNames
        
    cc.to_csv(r'/home/arno/work/research/lobster/data/features/' + theName, header = True, index = False)

    os.chdir(r'/home/arno/work/research/lobster/data')



#----- 2.5 Mean Price and Volume
for f in theOrderBookFiles:
    print(f + ' **** ' + datetime.now().strftime("%H:%M:%S")) # used for prtotyping only
    theName = f[0:15] + '_meanPriceAndVolume.csv'
    
    if (f[0:4] == 'INTC'):
        os.chdir(r'/home/arno/work/research/lobster/data/INTC')
    
    if (f[0:4] == 'TSLA'):
        os.chdir(r'/home/arno/work/research/lobster/data/TSLA') 
    
    dt = pd.read_csv(f, names = theNames)
    
    #--- Open Start & Stop file
    os.chdir(r'/home/arno/work/research/lobster/data/features/')
    sSFile = [g for g in os.listdir() if ((g[-17:] == '_startAndStop.csv') and (g[:15] == f[:15]))]
    startStopFile = pd.read_csv(sSFile[0], names = ['timeStamp','start','stop'], header = 0)
    start = np.array(startStopFile['start'])
    stop = np.array(startStopFile['stop'])
    
    for i in range(1,nlevels+1):
        #i = 1
        accAskName = 'Ask Price '+ str(i)
        accBidName = 'Bid Price '+ str(i)
        accAskVolName = 'Ask Size '+ str(i)
        accBidVolName = 'Bid Size '+ str(i)
        
        if i == 1:
            dtAccAsk = dt[accAskName]
            dtAccBid = dt[accBidName]
            dtAccAskVol = dt[accAskVolName]
            dtAccBidVol = dt[accBidVolName]
        
        if i != 1:
            dtAccAsk = dtAccAsk + dt[accAskName]
            dtAccBid = dtAccBid + dt[accBidName]
            dtAccAskVol = dtAccAskVol + dt[accAskVolName]
            dtAccBidVol = dtAccBidVol + dt[accBidVolName]
        
    meanAsk = np.array(dtAccAsk/nlevels) 
    meanBid = np.array(dtAccBid/nlevels)
    meanAskSize = np.array(dtAccAskVol/nlevels)
    meanBidSize = np.array(dtAccBidVol/nlevels)
    
    aa = [round(stats.percentileofscore(meanAsk[start[k]:k],meanAsk[k])/10,0) for k in range(len(startStopFile))] # Deciles
    bb = [round(stats.percentileofscore(meanBid[start[k]:k],meanBid[k])/10,0) for k in range(len(startStopFile))] # Deciles
    cc = [round(stats.percentileofscore(meanAskSize[start[k]:k],meanAskSize[k])/10,0) for k in range(len(startStopFile))] # Deciles
    dd = [round(stats.percentileofscore(meanBidSize[start[k]:k],meanBidSize[k])/10,0) for k in range(len(startStopFile))] # Deciles
    
    ee = pd.concat([pd.Series(startStopFile['timeStamp']),pd.Series(aa),pd.Series(bb),pd.Series(cc),pd.Series(dd)],axis=1)
    ee.columns = ['timeStamp','averageAsk','averageBid','averageAskSize','averageBidSize']
    
    ee.to_csv(r'/home/arno/work/research/lobster/data/features/' + theName, header = True, index = False)

    os.chdir(r'/home/arno/work/research/lobster/data')



#----- 2.6 Accumulated differences Price and Volume
for f in theOrderBookFiles:
    print(f + ' **** ' + datetime.now().strftime("%H:%M:%S")) # used for prtotyping only
    theName = f[0:15] + '_accDiffPriceAndVolume.csv'
    
    if (f[0:4] == 'INTC'):
        os.chdir(r'/home/arno/work/research/lobster/data/INTC')
    
    if (f[0:4] == 'TSLA'):
        os.chdir(r'/home/arno/work/research/lobster/data/TSLA') 
    
    dt = pd.read_csv(f, names = theNames)
    
    #--- Open Start & Stop file
    os.chdir(r'/home/arno/work/research/lobster/data/features/')
    sSFile = [g for g in os.listdir() if ((g[-17:] == '_startAndStop.csv') and (g[:15] == f[:15]))]
    startStopFile = pd.read_csv(sSFile[0], names = ['timeStamp','start','stop'], header = 0)
    start = startStopFile['start']
    stop = startStopFile['stop']
    
    aa = pd.DataFrame()
    
    # i = 1
    aa['AccPriceDiff'] = (dt ['Bid Price 1'] + dt ['Bid Price 2'] + dt ['Bid Price 3'] + dt ['Bid Price 4'] + dt ['Bid Price 5'] +
    dt ['Bid Price 6'] - dt ['Bid Price 7'] + dt ['Bid Price 8'] + dt ['Bid Price 9'] + dt ['Bid Price 10'] -
    dt ['Ask Price 1'] - dt ['Ask Price 2'] - dt ['Ask Price 3'] - dt ['Ask Price 4'] - dt ['Ask Price 5'] -
    dt ['Ask Price 6'] - dt ['Ask Price 7'] - dt ['Ask Price 8'] - dt ['Ask Price 9'] - dt ['Ask Price 10'])
    
    aa['AccSizeDiff'] = (dt ['Bid Size 1'] + dt ['Bid Size 2'] + dt ['Bid Size 3'] + dt ['Bid Size 4'] + dt ['Bid Size 5'] +
    dt ['Bid Size 6'] - dt ['Bid Size 7'] + dt ['Bid Size 8'] + dt ['Bid Size 9'] + dt ['Bid Size 10'] -
    dt ['Ask Size 1'] - dt ['Ask Size 2'] - dt ['Ask Size 3'] - dt ['Ask Size 4'] - dt ['Ask Size 5'] -
    dt ['Ask Size 6'] - dt ['Ask Size 7'] - dt ['Ask Size 8'] - dt ['Ask Size 9'] - dt ['Ask Size 10'])
    
    accPrideDiff = np.array(aa['AccPriceDiff'])
    accSizeDiff = np.array(aa['AccSizeDiff'])
    
    bb = [round(stats.percentileofscore(accPrideDiff[start[k]:k],accPrideDiff[k])/10,0) for k in range(len(startStopFile))] # Deciles
    cc = [round(stats.percentileofscore(accSizeDiff[start[k]:k],accSizeDiff[k])/10,0) for k in range(len(startStopFile))] # Deciles
    
    accDiffPriceAndVolume = pd.concat([pd.Series(startStopFile['timeStamp']),pd.Series(bb),pd.Series(cc)],axis=1)
    accDiffPriceAndVolume.columns = ['timeStamp','accPriceDiff','accSizeDiff']
        
    accDiffPriceAndVolume.to_csv(r'/home/arno/work/research/lobster/data/features/' + theName, header = True, index = False)

    os.chdir(r'/home/arno/work/research/lobster/data')



#----- 2.7 Price and Volume Derivatives
for f in theOrderBookFiles:
    print(f + ' **** ' + datetime.now().strftime("%H:%M:%S"))
    
    theName = f[0:15] + '_priceAndVolumeDerivatives.csv'
    
    if (f[0:4] == 'INTC'):
        os.chdir(r'/home/arno/work/research/lobster/data/INTC')
    
    if (f[0:4] == 'TSLA'):
        os.chdir(r'/home/arno/work/research/lobster/data/TSLA') 
   
    #--- The message file
    theMessageFile = f[0:15] + '_34200000_57600000_message_10.csv'
    mf = pd.read_csv(theMessageFile, names = ['timeStamp','EventType','Order ID','Size','Price','Direction'], float_precision='round_trip')
    dfTimeStamp = mf['timeStamp']
    timeStampInSeconds = dfTimeStamp.round(0)  
    maxLookBack = max(timeStampInSeconds.value_counts()) + 5     
    timeStamp = mf['timeStamp'].to_frame().values 
    
    #--- The order book file
    theDataFile = f[0:15] + '_34200000_57600000_orderbook_10.csv'
    df = pd.read_csv(theDataFile, names = theNames)
    dfArray = df.values
    
    #--- Open Start & Stop file
    os.chdir(r'/home/arno/work/research/lobster/data/features/')
    sSFile = [g for g in os.listdir() if ((g[-17:] == '_startAndStop.csv') and (g[:15] == f[:15]))]
    startStopFile = pd.read_csv(sSFile[0], names = ['timeStamp','start','stop'], header = 0)
    start = np.array(startStopFile['start'])
    stop = np.array(startStopFile['stop'])
    
    begin = [] 
    
#    start_time = time.time() # used only for profiling    
    for i in range(len(timeStamp)):
        # i = 223200
        if i == 0:
            begin.append(0)
            
        elif i < maxLookBack:
            theIndexValue = abs(timeStamp[:i,0] - (timeStamp[i,0] - 1)).argmin()
            begin.append(theIndexValue)
            
        elif i >= maxLookBack:    
            a = i - maxLookBack
            bb = np.column_stack((np.array(dfTimeStamp.iloc[a:i].index),timeStamp[a:i,0])) 
            theIndexValue = bb[abs(bb[:,1] - (timeStamp[i,0] - 1)).argmin(),0]
            begin.append(int(theIndexValue))
    
    end = np.array(range(len(timeStamp)))
    begin = np.array(begin)
#    print("--- %s seconds ---" % round(time.time() - start_time,2))  # used only for profiling           
    
    theBigArray = np.empty([len(end), df.shape[1]])
    for i in range(df.shape[1]):
        #   i = 0
        #   j = 0
        aa = np.array([(dfArray[end[j],i]/dfArray[begin[j],i] - 1) for j in range(len(end))])
        bb = [round(stats.percentileofscore(dfArray[start[k]:k,i],dfArray[k,i])/10,0) for k in range(len(startStopFile))]
        theBigArray[:,i] = np.array(bb)
       
    theBigArray = pd.DataFrame(theBigArray)
    theBigArray.columns = theNames
    
    theBigArray.to_csv(r'/home/arno/work/research/lobster/data/features/' + theName, header = True, index = False)

    os.chdir(r'/home/arno/work/research/lobster/data')



#----- 2.8 Average intensity (1 sec.)
#--- Only the following event types are selected (See LOBSTER page for the exect definition): 1(Submission of a new limit order) and 3(Deletion (total deletion of a limit order)      
for f in theMessagesFiles:
    #f = 'INTC_2015-01-02_34200000_57600000_message_10.csv'
    print(f + ' **** ' + datetime.now().strftime("%H:%M:%S"))
    
    theName = f[0:15] + '_tradeIntensity.csv'
    
    if (f[0:4] == 'INTC'):
        os.chdir(r'/home/arno/work/research/lobster/data/INTC')
    
    if (f[0:4] == 'TSLA'):
        os.chdir(r'/home/arno/work/research/lobster/data/TSLA') 
    
    theMessageFile = f[0:15] + '_34200000_57600000_message_10.csv'
    
    mf = pd.read_csv(theMessageFile, names = ['timeStamp','EventType','Order ID','Size','Price','Direction'], float_precision='round_trip')
    dfTimeStamp = mf['timeStamp']
    timeStampInSeconds = dfTimeStamp.round(0) # used to define max lookback period 
    maxLookBack = sum(timeStampInSeconds.value_counts().iloc[:2]) # used to define max lookback period 
    timeStamp = mf['timeStamp'].to_frame().values 
    mf = mf.values
    
    #--- Open Start & Stop file
    os.chdir(r'/home/arno/work/research/lobster/data/features/')
    sSFile = [g for g in os.listdir() if ((g[-17:] == '_startAndStop.csv') and (g[:15] == f[:15]))]
    startStopFile = pd.read_csv(sSFile[0], names = ['timeStamp','start','stop'], header = 0)
    start = np.array(startStopFile['start'])
    stop = np.array(startStopFile['stop'])
    
    eventsCount = np.empty((len(mf),3)) 
    
#    start_time = time.time() # For profiling only        
    for i in range(len(timeStamp)):
        if i == 0:
            eventsCount[0,0] = 0 
            eventsCount[0,1] = 0
            eventsCount[0,2] = 0
        
        elif i < maxLookBack:
            startIndexValue = abs(timeStamp[:i,0] - (timeStamp[i,0] - 1)).argmin()
            mfSmall = mf[startIndexValue:i,:]
            eventsCount[i,0] = np.count_nonzero(mfSmall[:,1] == 1)
            eventsCount[i,1] = np.count_nonzero(mfSmall[:,1] == 2)
            eventsCount[i,2] = np.count_nonzero(mfSmall[:,1] == 3)
            
        elif i >= maxLookBack:    
            a = i - maxLookBack
            bb = np.column_stack((np.array(dfTimeStamp.iloc[a:i].index),timeStamp[a:i,0]))
            startIndexValue = int(bb[abs(bb[:,1] - (timeStamp[i,0] - 1)).argmin(),0])
            mfSmall = mf[startIndexValue:i,:]
            eventsCount[i,0] = np.count_nonzero(mfSmall[:,1] == 1)
            eventsCount[i,1] = np.count_nonzero(mfSmall[:,1] == 2)
            eventsCount[i,2] = np.count_nonzero(mfSmall[:,1] == 3)
#    print("--- %s seconds ---" % round(time.time() - start_time,2))  # For profiling only  
    
    cc = [round(stats.percentileofscore(eventsCount[start[k]:k,0],eventsCount[k,0])/10,0) for k in range(len(startStopFile))]
    dd = [round(stats.percentileofscore(eventsCount[start[k]:k,1],eventsCount[k,1])/10,0) for k in range(len(startStopFile))]
    ee = [round(stats.percentileofscore(eventsCount[start[k]:k,2],eventsCount[k,2])/10,0) for k in range(len(startStopFile))]
    
    ff = pd.concat([dfTimeStamp,pd.Series(cc),pd.Series(dd),pd.Series(ee),pd.DataFrame(eventsCount)],axis=1)
    ff.columns = ['timeStamp','decEventType_1','decEventType_2','decEventType_3','levelEventType_1','levelEventType_2','levelEventType_3']
    
    ff.to_csv(r'/home/arno/work/research/lobster/data/features/' + theName, header = True, index = False)

    os.chdir(r'/home/arno/work/research/lobster/data')



#----- 2.9 Relative intensity - step 1 - (10 sec.) 
for f in theMessagesFiles:
    print(f + ' **** ' + datetime.now().strftime("%H:%M:%S"))
    
    theName = f[0:15] + '_relativeIntensity10s.csv'
    
    if (f[0:4] == 'INTC'):
        os.chdir(r'/home/arno/work/research/lobster/data/INTC')
    
    if (f[0:4] == 'TSLA'):
        os.chdir(r'/home/arno/work/research/lobster/data/TSLA') 
    
    theMessageFile = f[0:15] + '_34200000_57600000_message_10.csv'
    
    mf = pd.read_csv(theMessageFile, names = ['timeStamp','EventType','Order ID','Size','Price','Direction'], float_precision='round_trip')
    dfTimeStamp = mf['timeStamp']
    timeStampInSeconds = dfTimeStamp.round(0)
    maxLookBack = sum(timeStampInSeconds.value_counts().iloc[:11]) 
    timeStampInSeconds = timeStampInSeconds.values 
    timeStamp = mf['timeStamp'].to_frame().values 
    mf = mf.values
    
    eventsCount10s = np.empty((len(mf),3)) 

#    start_time = time.time() # For profiling only        
    for i in range(len(timeStamp)):
        if (i == 0):
            eventsCount10s[i,0] = 0
            eventsCount10s[i,1] = 0
            eventsCount10s[i,2] = 0
            
        if (i <= maxLookBack):
            mfSmall = mf[:i,1]
            eventsCount10s[i,0] = np.count_nonzero(mfSmall == 1)
            eventsCount10s[i,1] = np.count_nonzero(mfSmall == 2)
            eventsCount10s[i,2] = np.count_nonzero(mfSmall == 3) 
            
        elif (i > maxLookBack):
            a = i - maxLookBack
            aa = np.column_stack((np.array(dfTimeStamp.iloc[a:i].index),timeStamp[a:i,0]))
            startIndexValue = int(aa[abs(aa[:,1] - (timeStamp[i,0] - 10)).argmin(),0])
            mfSmall = mf[startIndexValue:i,1]
            eventsCount10s[i,0] = np.count_nonzero(mfSmall == 1)
            eventsCount10s[i,1] = np.count_nonzero(mfSmall == 2)
            eventsCount10s[i,2] = np.count_nonzero(mfSmall == 3)    
#    print("--- %s seconds ---" % round(time.time() - start_time,2))  # For profiling only  
    
    eventsCount10s = pd.DataFrame(eventsCount10s)
    eventsCount10s.columns = ['EventType_1','EventType_2','EventType_3']
    
    # Add time stamp
    eventsCount10s = pd.concat([dfTimeStamp,eventsCount10s],axis=1)
    eventsCount10s.columns = ['timeStamp','EventType_1','EventType_2','EventType_3']
    eventsCount10s.to_csv(r'/home/arno/work/research/lobster/data/features/' + theName, header = True, index = False)

    os.chdir(r'/home/arno/work/research/lobster/data')



#----- 2.10 Relative intensity - step 2 - (900 sec.) 
for f in theMessagesFiles:
    print(f + ' **** ' + datetime.now().strftime("%H:%M:%S"))
    
    theName = f[0:15] + '_relativeIntensity900s.csv'
    
    if (f[0:4] == 'INTC'):
        os.chdir(r'/home/arno/work/research/lobster/data/INTC')
    
    if (f[0:4] == 'TSLA'):
        os.chdir(r'/home/arno/work/research/lobster/data/TSLA') 
    
    theMessageFile = f[0:15] + '_34200000_57600000_message_10.csv'
    
    mf = pd.read_csv(theMessageFile, names = ['timeStamp','EventType','Order ID','Size','Price','Direction'], float_precision='round_trip')
    dfTimeStamp = mf['timeStamp']
    timeStampInSeconds = dfTimeStamp.round(0)
    maxLookBack = sum(timeStampInSeconds.value_counts().iloc[:901]) 
    timeStampInSeconds = timeStampInSeconds.values 
    timeStamp = mf['timeStamp'].to_frame().values 
    mf = mf.values
    
    eventsCount900s = np.empty((len(mf),3)) 

#    start_time = time.time() # For profiling only        
    for i in range(len(timeStamp)):
        if (i == 0):
            eventsCount900s[i,0] = 0
            eventsCount900s[i,1] = 0
            eventsCount900s[i,2] = 0
            
        if (i <= maxLookBack):
            mfSmall = mf[:i,1]
            eventsCount900s[i,0] = np.count_nonzero(mfSmall == 1)
            eventsCount900s[i,1] = np.count_nonzero(mfSmall == 2)
            eventsCount900s[i,2] = np.count_nonzero(mfSmall == 3) 
            
        elif (i > maxLookBack):
            a = i - maxLookBack
            aa = np.column_stack((np.array(dfTimeStamp.iloc[a:i].index),timeStamp[a:i,0]))
            startIndexValue = int(aa[abs(aa[:,1] - (timeStamp[i,0] - 900)).argmin(),0])
            mfSmall = mf[startIndexValue:i,1]
            eventsCount900s[i,0] = np.count_nonzero(mfSmall == 1)
            eventsCount900s[i,1] = np.count_nonzero(mfSmall == 2)
            eventsCount900s[i,2] = np.count_nonzero(mfSmall == 3) 
#    print("--- %s seconds ---" % round(time.time() - start_time,2))  # For profiling only  
    
    eventsCount900s = pd.DataFrame(eventsCount900s)
    eventsCount900s.columns = ['EventType_1','EventType_2','EventType_3']
    
    # Add time stamp
    eventsCount900s = pd.concat([dfTimeStamp,eventsCount900s],axis=1)
    eventsCount900s.columns = ['timeStamp','EventType_1','EventType_2','EventType_3']
    eventsCount900s.to_csv(r'/home/arno/work/research/lobster/data/features/' + theName, header = True, index = False)

    os.chdir(r'/home/arno/work/research/lobster/data')



#----- 2.11 Relative intensity - Step 3 - Put it all together   
os.chdir(r'/home/arno/work/research/lobster/data/features/')
allFiles = os.listdir()

files900s = [sl for sl in allFiles if "relativeIntensity900s" in sl and 'TSLA' in sl]
files900s.sort()

files10s = [sk for sk in allFiles if "relativeIntensity10s" in sk and 'TSLA' in sk]
files10s.sort()

for i in range(len(files10s)):
    print(files10s[i] + ' *** ' + datetime.now().strftime("%H:%M:%S"))
    theName = files10s[i][0:15] + '_relativeIntensity.csv' 
        
    the10sFile = pd.read_csv(files10s[i], float_precision='round_trip')
    the900sFile = pd.read_csv(files900s[i], float_precision='round_trip')
    
    #--- Open Start & Stop file
    os.chdir(r'/home/arno/work/research/lobster/data/features/')
    sSFile = [g for g in os.listdir() if ((g[-17:] == '_startAndStop.csv') and (g[:15] == files10s[i][:15]))]
    startStopFile = pd.read_csv(sSFile[0], names = ['timeStamp','start','stop'], header = 0)
    start = np.array(startStopFile['start'])
    stop = np.array(startStopFile['stop'])

    relInt1 = np.array(the10sFile['EventType_1']/the900sFile['EventType_1'])
    relInt2 = np.array(the10sFile['EventType_2']/the900sFile['EventType_2'])
    relInt3 = np.array(the10sFile['EventType_3']/the900sFile['EventType_3'])
    
    aa = [round(stats.percentileofscore(relInt1[start[k]:k],relInt1[k])/10,0) for k in range(len(startStopFile))]
    bb = [round(stats.percentileofscore(relInt2[start[k]:k],relInt2[k])/10,0) for k in range(len(startStopFile))]
    cc = [round(stats.percentileofscore(relInt3[start[k]:k],relInt3[k])/10,0) for k in range(len(startStopFile))]
    
    relInt = pd.concat([the10sFile['timeStamp'],pd.Series(aa),pd.Series(bb),pd.Series(cc)], axis = 1)
    relInt.columns = ['timeStamp','relIntEventType_1','relIntEventType_2','relIntEventType_3']
    
    relInt.to_csv(r'/home/arno/work/research/lobster/data/features/' + theName, header = True, index = False)


