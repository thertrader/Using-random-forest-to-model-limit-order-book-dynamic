#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calibration and test of the Random Forest model 

1. Basic parameters
2. Features pre processing - Run it only once
3. Calibration & test

@author: thertrader@gmail.com
@Date: Mar 2020
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
#from sklearn import metrics

sys.path.append('/home/arno/work/programming/python') # add to the list of python paths
import useful


# =============================================================================
# 1 - Basic parameters
# =============================================================================
tickers = ['TSLA'] 

#----- Lists of all files 
theFiles = []  
for tk in tickers:
    # tk = 'INTC'
    os.chdir(r"/home/arno/work/research/lobster/data/" + tk)
    theFiles.extend(sorted(os.listdir()))

#----- Unique dates
theDates = []
for i in range(0,len(theFiles)):
    theDates.append(theFiles[i][5:15])

theDates = sorted(list(set(theDates)))

#--- To limit the effect of Open & Close auctions I removed the first and last 15 minutes. Not sure what the real impact is...  
top = 35100 # starts @ 9:45
bottom = 56700 # starts @ 15:45


# =============================================================================
# 2 - Features pre processing - Run it only once
# =============================================================================
# tk = 'TSLA'
# f = 'TSLA_2015-01-02_34200000_57600000_orderbook_10.csv'
# m = '2015-01-05'
# g = 'TSLA_2015-01-02_Y10Sec.csv'
os.chdir(r'/home/arno/work/research/lobster/data/features/') 
for tk in tickers:    
    for m in theDates:
        print(m + ' **** ' + datetime.now().strftime("%H:%M:%S")) # for prototyping only
        thisStockFiles = [f for f in os.listdir() if (f[:4] == tk and f[5:15] == m)]
        xFiles = sorted([f for f in thisStockFiles if ((f[16] != 'Y') and (f[16:36] != 'relativeIntensity10s') and (f[16:37] != 'relativeIntensity900s') and (f[16:28] != 'startAndStop'))])
        yFiles = sorted([f for f in thisStockFiles if f[16] == 'Y'])
        
        #--- Define start, stop lines and index to keep
        dd = pd.read_csv(yFiles[0])
        timeStamp = dd['timeStamp']
        t = abs(timeStamp - top).idxmin()
        b = abs(timeStamp - bottom).idxmin()
        
        #--- Y variables
        yVar = pd.DataFrame(dtype = 'int')        
        for g in yFiles:
            dt = pd.read_csv(g)
            dt = dt.drop(columns = ['timeStamp'])
            yVar = pd.concat([yVar, dt], axis = 1)
        yVar = yVar.iloc[t:(b+1),:]
        yVar.columns = ['midRtn_10','midRtn_20']
        yVar = yVar.astype(int)
        
        #--- X variables    
        xVar = pd.DataFrame()
        for f in xFiles:
            dt = pd.read_csv(f)
            xVar = pd.concat([xVar, dt], axis = 1)    

        xVar = xVar.drop(columns = ['timeStamp','decEventType_2','levelEventType_2','relIntEventType_2'])
        xVar = xVar[[f for f in xVar.columns if (f[-1] in list('12345') or f[:3] == 'acc' or f[:3] == 'ave')]] # Drop order book levels > 5
        xVar = xVar.iloc[t:(b+1),:] # remove first and last 15 min.
        print(m + ' **** '+ str(len(list(xVar.columns))))
        
        yVar.to_csv(r'/home/arno/work/research/lobster/data/calibration/yVar_' + tk + '_' + m + '.csv', header = True, index = False)
        xVar.to_csv(r'/home/arno/work/research/lobster/data/calibration/xVar_' + tk + '_' + m + '.csv', header = True, index = False)
        
        del xVar,yVar
        
        
# =============================================================================
# 3 - Calibration & test
# =============================================================================
#--- Create RFC      
rfc = RandomForestClassifier(
                            n_jobs = 4, # How many processors is it allowed to use. -1 means there is no restriction, 1 means it can only use one processor
                            n_estimators = 200, # The number of trees in the forest
                            min_samples_leaf = 200, # Minimum number of observations (i.e. samples) in terminal leaf
                            min_samples_split = 300, # represents the minimum number of samples (i.e. observations) required to split an internal node. 
                            oob_score = True, # This is a random forest cross validation method. It is very similar to leave one out validation technique 
                            max_depth = None, # The maximum depth of the tree
                            verbose  = 1, # To check progress of the estimation
                            max_features = 'sqrt') # The number of features to consider when looking for the best split. None = no limit    

 
#--- Set parameters to run the model 
os.chdir(r'/home/arno/work/research/lobster/data/calibration')
listOfFiles = os.listdir()

tk = 'TSLA'
th = 0.50
lookBackPeriod = 4
indepVar = 'midRtn_10'

gatherResults = pd.DataFrame(index = range((20-lookBackPeriod)), 
                             columns = ['date','IS-HR-U','#obs.1','IS-HR-S','#obs.2','IS-HR-D','#obs.3','OOS-HR-U','#obs.4','OOS-HR-S','#obs.5','OOS-HR-D','#obs.6',
                                        'IS-HR-U55','#obs.155','IS-HR-S55','#obs.255','IS-HR-D55','#obs.355','OOS-HR-U55','#obs.455','OOS-HR-S55','#obs.555','OOS-HR-D55','#obs.655',
                                        'IS-HR-U60','#obs.160','IS-HR-S60','#obs.260','IS-HR-D60','#obs.360','OOS-HR-U60','#obs.460','OOS-HR-S60','#obs.560','OOS-HR-D60','#obs.660',
                                        'overallScore'])

#--- Run the model 
for i in range(len(theDates)):
    if (i >= lookBackPeriod):    
        print(theDates[i] + ' **** ' + datetime.now().strftime("%H:%M:%S")) # for prototyping only    
        for tk in tickers:        
            outputName = tk + '_' + theDates[i] + '.csv'
            
            #--- Define features & labels calibration dataset (4 days)             
            x = pd.DataFrame()
            xFiles = sorted([f for f in listOfFiles if f[5:9] == tk and f[0] == 'x'])
            xFiles = xFiles[(i-lookBackPeriod):i]
            for g in xFiles:
                dt = pd.read_csv(g)
                x = pd.concat([x, dt], axis = 0)  
            
            y = pd.DataFrame()
            yFiles = sorted([f for f in listOfFiles if f[5:9] == tk and f[0] == 'y'])
            yFiles = yFiles[(i-lookBackPeriod):i]
            for h in yFiles:
                du = pd.read_csv(h)
                y = pd.concat([y, du], axis = 0)         
            
            #--- Define features & labels test dataset (1 day)
            xTestFile = 'xVar_' + tk + '_' + theDates[i] + '.csv'
            xTest = pd.read_csv(xTestFile)

            yTestFile = 'yVar_' + tk + '_' + theDates[i] + '.csv'
            yTest = pd.read_csv(yTestFile)
            
            #--- Random forest estimation
            theFit = rfc.fit(x.values, y[indepVar].values)
            
            #---- Extract Decision Rules from in sample estimateions
            decisionsRules = pd.Series(useful.getDecisionRules(theFit))
            decisionsRules.to_csv(r'/home/arno/work/research/lobster/results/decisionsRules_' + outputName, header = False, index = False)
            
            #---- Feature importance ranked by score
            featureImportance = pd.concat([pd.Series(x.columns),pd.Series(theFit.feature_importances_)], axis = 1)
            featureImportance.columns = ['feature','score']
            featureImportance = featureImportance.sort_values(by=['score'], ascending=False)
            featureImportance.to_csv(r'/home/arno/work/research/lobster/results/featuresImportance_' + outputName, header = True, index = False)
            
            #---- Calibration diagnostic
            featureOverallScore = theFit.score(x.values, y[indepVar].values)
                        
            #---- Calibration; Hit Ratio per label
            pp = theFit.predict_proba(x) # tricky calculation (see scikit doc).....
            newDt = np.concatenate([pp,y],axis= 1)
            newDt = newDt[:,[0,1,2,3]]
            
            #---- Test: Hit Ratio per label
            qq = theFit.predict_proba(xTest)
            newDtTest = np.concatenate([qq,yTest],axis= 1)
            newDtTest = newDtTest[:,[0,1,2,3]]
            
            #--- Store results 
            k = i - lookBackPeriod
            gatherResults.iloc[k,0] = int(theDates[i][:4] + theDates[i][5:7] + theDates[i][8:10])
            #--- th = 0.5
            gatherResults.iloc[k,1] = np.where((newDt[:,0] > th) & (newDt[:,3] == -1),1,0).sum()/np.where(newDt[:,0] > th,1,0).sum()
            gatherResults.iloc[k,2] = np.where(newDt[:,0] > th,1,0).sum()
            gatherResults.iloc[k,3] = np.where((newDt[:,1] > th) & (newDt[:,3] == 0),1,0).sum()/np.where(newDt[:,1] > th,1,0).sum()
            gatherResults.iloc[k,4] = np.where(newDt[:,1] > th,1,0).sum()
            gatherResults.iloc[k,5] = np.where((newDt[:,2] > th) & (newDt[:,3] == 1),1,0).sum()/np.where(newDt[:,2] > th,1,0).sum()
            gatherResults.iloc[k,6] = np.where(newDt[:,2] > th,1,0).sum()
            
            gatherResults.iloc[k,7] = np.where((newDtTest[:,0] > th) & (newDtTest[:,3] == -1),1,0).sum()/np.where(newDtTest[:,0] > th,1,0).sum()
            gatherResults.iloc[k,8] = np.where(newDtTest[:,0] > th,1,0).sum()
            gatherResults.iloc[k,9] = np.where((newDtTest[:,1] > th) & (newDtTest[:,3] == 0),1,0).sum()/np.where(newDtTest[:,1] > th,1,0).sum()
            gatherResults.iloc[k,10] = np.where(newDtTest[:,1] > th,1,0).sum()
            gatherResults.iloc[k,11] = np.where((newDtTest[:,2] > th) & (newDtTest[:,3] == 1),1,0).sum()/np.where(newDtTest[:,2] > th,1,0).sum()
            gatherResults.iloc[k,12] = np.where(newDtTest[:,2] > th,1,0).sum()
            
            #--- th = 0.55
            gatherResults.iloc[k,13] = np.where((newDt[:,0] > (th+0.05)) & (newDt[:,3] == -1),1,0).sum()/np.where(newDt[:,0] > (th+0.05),1,0).sum()
            gatherResults.iloc[k,14] = np.where(newDt[:,0] > (th+0.05),1,0).sum()
            gatherResults.iloc[k,15] = np.where((newDt[:,1] > (th+0.05)) & (newDt[:,3] == 0),1,0).sum()/np.where(newDt[:,1] > (th+0.05),1,0).sum()
            gatherResults.iloc[k,16] = np.where(newDt[:,1] > (th+0.05),1,0).sum()
            gatherResults.iloc[k,17] = np.where((newDt[:,2] > (th+0.05)) & (newDt[:,3] == 1),1,0).sum()/np.where(newDt[:,2] > (th+0.05),1,0).sum()
            gatherResults.iloc[k,18] = np.where(newDt[:,2] > (th+0.05),1,0).sum()
            
            gatherResults.iloc[k,19] = np.where((newDtTest[:,0] > (th+0.05)) & (newDtTest[:,3] == -1),1,0).sum()/np.where(newDtTest[:,0] > (th+0.05),1,0).sum()
            gatherResults.iloc[k,20] = np.where(newDtTest[:,0] > (th+0.05),1,0).sum()
            gatherResults.iloc[k,21] = np.where((newDtTest[:,1] > (th+0.05)) & (newDtTest[:,3] == 0),1,0).sum()/np.where(newDtTest[:,1] > (th+0.05),1,0).sum()
            gatherResults.iloc[k,22] = np.where(newDtTest[:,1] > (th+0.05),1,0).sum()
            gatherResults.iloc[k,23] = np.where((newDtTest[:,2] > (th+0.05)) & (newDtTest[:,3] == 1),1,0).sum()/np.where(newDtTest[:,2] > (th+0.05),1,0).sum()
            gatherResults.iloc[k,24] = np.where(newDtTest[:,2] > (th+0.05),1,0).sum()
            
            #--- th = 0.6
            gatherResults.iloc[k,25] = np.where((newDt[:,0] > (th+0.1)) & (newDt[:,3] == -1),1,0).sum()/np.where(newDt[:,0] > (th+0.1),1,0).sum()
            gatherResults.iloc[k,26] = np.where(newDt[:,0] > (th+0.1),1,0).sum()
            gatherResults.iloc[k,27] = np.where((newDt[:,1] > (th+0.1)) & (newDt[:,3] == 0),1,0).sum()/np.where(newDt[:,1] > (th+0.1),1,0).sum()
            gatherResults.iloc[k,28] = np.where(newDt[:,1] > (th+0.1),1,0).sum()
            gatherResults.iloc[k,29] = np.where((newDt[:,2] > (th+0.1)) & (newDt[:,3] == 1),1,0).sum()/np.where(newDt[:,2] > (th+0.1),1,0).sum()
            gatherResults.iloc[k,30] = np.where(newDt[:,2] > (th+0.1),1,0).sum()
            
            gatherResults.iloc[k,31] = np.where((newDtTest[:,0] > (th+0.1)) & (newDtTest[:,3] == -1),1,0).sum()/np.where(newDtTest[:,0] > (th+0.1),1,0).sum()
            gatherResults.iloc[k,32] = np.where(newDtTest[:,0] > (th+0.1),1,0).sum()
            gatherResults.iloc[k,33] = np.where((newDtTest[:,1] > (th+0.1)) & (newDtTest[:,3] == 0),1,0).sum()/np.where(newDtTest[:,1] > (th+0.1),1,0).sum()
            gatherResults.iloc[k,34] = np.where(newDtTest[:,1] > (th+0.1),1,0).sum()
            gatherResults.iloc[k,35] = np.where((newDtTest[:,2] > (th+0.1)) & (newDtTest[:,3] == 1),1,0).sum()/np.where(newDtTest[:,2] > (th+0.1),1,0).sum()
            gatherResults.iloc[k,36] = np.where(newDtTest[:,2] > (th+0.1),1,0).sum()
            gatherResults.iloc[k,37] = featureOverallScore
            
gatherResults.to_csv(r'/home/arno/work/research/lobster/results/results.csv', header = True, index = False)










        







