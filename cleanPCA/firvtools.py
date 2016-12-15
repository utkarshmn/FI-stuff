import quandl

import numpy as np
from numpy import array
import pandas as pd
from operator import sub
import pickle
import math
import itertools as iter
import datetime as dt
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, rc
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA, KernelPCA

from IPython.display import display, HTML
import json
def pretty(obj):
    return json.dumps(obj, sort_keys=True, indent=2)

## Directories
cwd = os.getcwd()
dataDir = cwd+"/depo/Data/"
saveDir = cwd+"/depo/Outputs/"

##PV01s

pv01par = pd.read_csv(dataDir+"pv01s.csv",thousands=",").set_index('tenor')["pv01"].T.to_dict()
pv01d = {}

def computeFwds(data, pv01s = pv01d):
    
    newParCols = [x[3:]+"y" for x in data.columns.values]
    par = data.copy()
    par.columns = newParCols

    fwds = pd.DataFrame(par[newParCols[0]]*100, index=par.index.values)
    parCols = par.columns.values
    allFwds = [parCols[i]+str(int(parCols[i+1][:-1])-int(parCols[i][:-1]))+"y" for i in range(len(parCols)-1)]

    for i in range(len(allFwds)):
        t1, t2 = str(parCols[i]), str(parCols[i+1])
        p1, p2 = pv01s[t1], pv01s[t2]
        r1, r2 = par[t1], par[t2]
        fwds[allfwds[i]] = 100*(r2*p2-r1*p1)/(p2-p1)
    
    return fwds

def calcFwdRate(r1, r2, t1, t2, pv01s=pv01d):
    p1 = pv01s[t1]
    p2 = pv01s[t2]
    return (p2*r2-p1*r1)/(p2-p1)


def computeFwdPV01s(pv01s = pv01par):
    result = pv01s
    parCols = list(pv01s.keys())

    parCols.sort(key=lambda x: int(x[:-1]))
    allFwds = [parCols[i]+str(int(parCols[i+1][:-1])-int(parCols[i][:-1]))+"y" for i in range(len(parCols)-1)]

    for i in range(len(allFwds)):
        t1, t2 = str(parCols[i]), str(parCols[i+1])
        p1, p2 = pv01s[t1], pv01s[t2]
        result[allFwds[i]] = p2-p1

    return result

pv01d = {
  "10y": 914.99,
  "10y2y": 158.26,
  "12y": 1073.25,
  "12y3y": 222.14,
  "15y": 1295.39,
  "15y5y": 334.49,
  "1y": 99.42,
  "1y1y": 98.8,
  "20y": 1629.88,
  "20y5y": 295.58,
  "25y": 1925.46,
  "25y5y": 262.28,
  "2y": 198.22,
  "2y1y": 96.44,
  "30y": 2187.74,
  "30y10y": 442.05,
  "3y": 294.66,
  "3y1y": 94.64,
  "40y": 2629.79,
  "4y": 389.3,
  "4y1y": 92.94,
  "5y": 482.24,
  "5y1y": 90.84,
  "6y": 573.08,
  "6y1y": 88.71,
  "7y": 661.79,
  "7y3y": 253.2
}


freqRules = {
  "A-DEC": "A",
  "AS-JAN": "AS",
  "B": "WEEKDAY",
  "BA-APR": "A@APR",
  "BA-AUG": "A@AUG",
  "BA-DEC": "BA",
  "BA-FEB": "A@FEB",
  "BA-JAN": "A@JAN",
  "BA-JUL": "A@JUL",
  "BA-JUN": "A@JUN",
  "BA-MAR": "A@MAR",
  "BA-MAY": "A@MAY",
  "BA-NOV": "A@NOV",
  "BA-OCT": "A@OCT",
  "BA-SEP": "A@SEP",
  "BAS-JAN": "BAS",
  "BM": "EOM",
  "BQ-FEB": "Q@FEB",
  "BQ-JAN": "Q@JAN",
  "BQ-MAR": "Q@MAR",
  "L": "ms",
  "Q-DEC": "Q",
  "T": "Min",
  "U": "us",
  "W-FRI": "W@FRI",
  "W-MON": "W@MON",
  "W-SAT": "W@SAT",
  "W-SUN": "W",
  "W-THU": "W@THU",
  "W-TUE": "W@TUE",
  "W-WED": "W@WED"
}

freqString = {
    "BA-":"Daily",
    "W-":"Weekly",
    "BQ-":"Quarterly",
    "A":"Annual"
}
###### PCA TOOLS ######

def getPV01(maturity):
    return bpv[mat.index(maturity)]

def kfacReconstruct(data, evTable, k=3, cols = [], auto= 0):
    ## Auto reorients EVs and PCs
    ## Pick securities to receive data for. If no input, do all. 
    if cols ==[]: 
        cols = data.columns.values
    
    # Returns k-factor reconstruction when given the data and Eigenvectors
    
    result = {}
    
    totFactors = len(evTable.columns.values)
    if totFactors < k:
        print ("Error! Total factors are less than k.")
        return
    
    # get demeaned data
    meanVals = data.mean()
    demeaned = data - meanVals
    
    #reconstruct historical factor time series
    factorTS = demeaned.dot(evTable)
    if auto != 0:
        reOrient = pcReorient(data, factorTS, tol=auto)[:totFactors]
        newEVs = evTable.copy()
        newFactors = factorTS.copy()
        for i in range(totFactors):
            newEVs.loc[:, evTable.columns[i]] *= reOrient[i]
            newFactors.loc[:, factorTS.columns[i]] *= reOrient[i]
        factorTS = newFactors
        evTable = newEVs

    #inverse of eigenvectors
    invEV = pd.DataFrame(np.linalg.pinv(evTable.values), evTable.columns, evTable.index)

    #drop columns to adjust for k factors
    factorTS.drop(factorTS.columns[range(len(factorTS.columns.values))[k:]], axis=1, inplace=True)
    
    #drop rows to adjust for k factors
    invEV.drop(invEV.index[range(len(invEV.index.values))[k:]], axis=0, inplace=True)
    
    #### Reconstruction using k factors
    kRebuild = factorTS.dot(invEV)
    kResiduals = demeaned - kRebuild
    reRaw = kRebuild + meanVals
    
    result["factorTS"] = factorTS
    result["rebuildRaw"] = reRaw[cols]
    result["residuals"] = kResiduals[cols]
    
    return result, evTable
    
def pcReorient(data, factors, tol=2):
    lenData = len(data.columns.values)
    numCurves = lenData//2
    numFlies = comb(lenData, 3)    
    
    pc1corr, pc2corr, pc3corr = 1, 1, 1
        
    if tol > numCurves or tol > numFlies:
        print ("Error.")
        return
    numFactors = len(factors.columns.values)
    if numFactors >=1:
        ## check pc1
        split = lenData//tol
        pc1check = pd.concat([data.ix[:,j*split] for j in range(tol)], axis=1)
        pc1corr = pc1check.corrwith(factors["PC1"]).mean()
    if numFactors >=2:
        ## check pc2
        pc2check = pd.concat([data.ix[:,-i]-data.ix[:,i] for i in range(numCurves)], axis=1)
        pc2corr = pc2check.corrwith(factors["PC2"]).mean()
    if numFactors >=3:
        ## check pc3
        pc3check = data.ix[:,numCurves]*2 - data.ix[:,0] - data.ix[:,-1]
        pc3corr = factors["PC3"].corr(pc3check).mean()
    
    
    return np.sign([pc1corr, pc2corr, pc3corr])

def staticPCA(data, n=3, freq=1, corrW = 12, autoOrient=2, plot=True):
    '''
    This function returns a dictionary with the following key value combinations:
    key    | Value
    raw    | DF with raw data (as resampled)
    covM   | Covariance matrix of raw data
    evals  | n eigenvalues
    evecs  | n eigenvectors
    facTS  | time series of reconstructed factors using raw data. 
    reRaw  | Rebuilt raw data from n EVs, and accompanying residuals
    resid  | Residuals (Actual - Reconstructed)
    facCR  | Rolling cross correlations between factors
    '''
    results = {}
    
    ## resample data based on freq and calc demeaned data
    raw = data.iloc[::freq, :]
    results["rawDat"] = raw
    meanVals = raw.mean()
    demeaned = raw - meanVals
    
    ## Covariance Matrix
    covM = raw.cov()
    results["covM"] = covM
    
    ## PCA
    evals, evecs = np.linalg.eig(covM)
    epairs = [(np.abs(evals[i]), evecs[:,i]) for i in range(len(evals))]
    epairs.sort(key=lambda x: x[0], reverse=True)
    evals = sorted([i*100/sum(evals) for i in evals])[::-1][:n]
    results["evals"] = evals

    evTable = pd.DataFrame(index=covM.index)
    for i in range(n):
        evTable.insert(i, "PC"+str(i+1), epairs[i][1])

    ## Reconstruct
    reConResult = kfacReconstruct(raw, evTable, n, auto=autoOrient)
    reconstructed = reConResult[0]
    evTable = reConResult[1].copy()
    evTable.index = covM.index
    results["evecs"] = evTable
    resid = reconstructed["residuals"]
    facTS = reconstructed["factorTS"]
    reRaw = reconstructed["rebuildRaw"]
    
    results["resid"] = resid
    results["facTS"] = facTS
    results["reRaw"] = reRaw
    

    ## Rolling correlations of factors
    facCR = pd.DataFrame()
    combos =[facTS[list(pair)] for pair in list(iter.combinations(facTS.columns, 2))]
    for df in combos:
        cols = df.columns.values
        facCR["".join(cols)] = facTS[cols[0]].rolling(window=corrW).corr(other=facTS[cols[1]])

    results["facCR"] = facCR
    
    return results


def rollingPCA(data, lb=30, n=3, corrW=12, skip=1):
    """
    For now, this function returns a data frame with time series of eigenvalues and eigenvectors of a rolling PCA. 
    """

    rollResult = {}
    
    assets = data.columns.values
    pcCols = ["PC"+str(i+1) for i in range(n)]
    eVecCols = [pc+asset for pc in pcCols for asset in assets]
    
    # Create dataframe for results. 
    accumEvals = pd.DataFrame(columns = range(1, n+1), index=data.index.values[lb:])
    accumEvecs = dict.fromkeys(pcCols, pd.DataFrame(columns=assets, index=data.index.values[lb:]))
    alleVectors = pd.DataFrame(index=data.index.values[lb:], columns = eVecCols)
    
    ### Rolling PCA - Loop and save data.
    
    for i in range(0, len(data.index)-lb, skip):
        
        temp = data[i:lb+i]
        currDate = data.index.values[lb+i]
        res = staticPCA(temp, n=n, corrW=corrW)

        ## Save eigenvalues
        eigenvalues = res["evals"]
        for j in range(len(eigenvalues)):
            accumEvals.set_value(currDate,j+1,eigenvalues[j])

        ## Save eigenvectors
        eigenvectors = res["evecs"]
        tempDict = {}
        for pc in pcCols:
            factor = eigenvectors[pc]
            #print (i, pc, factor.tolist())
            for k in range(len(assets)):
                alleVectors.set_value(currDate, pc+assets[k], factor[k])
    
    #### Split data into PC1, PC2, PC3 and drop NAs.
    
    accumEvals = accumEvals.dropna()

    grouped = alleVectors.groupby(lambda x: x[:3], axis=1)
    
    for pc in pcCols:
        accumEvecs[pc] = grouped.get_group(pc).dropna()
        accumEvecs[pc].rename(columns=lambda x: x[3:], inplace=True)
    accumEvals.columns = pcCols
    rollResult["evectors"] = accumEvecs
    rollResult["evalues"] = accumEvals
    
    ## Secret
    rollResult["master"] = alleVectors.dropna()
    
    return rollResult

def cleanPCs(input, smoothing=0):
	#### Function to help out the output of rollingPCAs. 
    newdf = pd.DataFrame(index=input.index.values, columns=input.columns.values)
    flag = "flipped"

    for i in range(1, len(input)):
        if i ==1:
            prevfactor = input.iloc[i-1]
        else:
            prevfactor = newdf.iloc[i-1]
        factor = input.iloc[i]
        negfactor = [-i for i in factor.tolist()]

        orig = sum([np.abs(x) for x in (factor-prevfactor).tolist()])
        new = sum([np.abs(x) for x in (-factor-prevfactor).tolist()])
        
        if orig > new:
            for s in range(len(input.columns.values)):
                newdf.set_value(newdf.index.values[i], newdf.columns.values[s], -1*input.iloc[i, s])
                flag = "flipped"
        else:
            for s in range(len(input.columns.values)):
                newdf.set_value(newdf.index.values[i], newdf.columns.values[s], input.iloc[i, s])
                flag = "same"
    
    if smoothing==1:
        newdf = newdf[newdf.apply(lambda x: np.abs(x - x.mean()) / x.std() < 2.5).all(axis=1)]
        newdf = newdf[newdf.apply(lambda x: np.abs(x - x.mean()) / x.std() < 2.5).all(axis=1)]

        
    return newdf



####### Trade Related #######


def returnNotWgts(trade, evectors, bpvs=0):
    ### Returns notionals to use in PCA weighted butterflies. 
    tenors = sorted([int(x.strip()) for x in trade.split("s") if x])
    tenors = [str(i)+"y" for i in tenors]
    ## re-index evectors:
    newIndex = [''.join(i for i in x if i.isdigit())+"y" for x in evectors.index.values]
    evectors.index = newIndex
    
    if bpvs == 0:
        return returnRiskWgts(trade, evectors)
    
    ## curves
    if len(tenors)==2:
        ## longer tenor = 1
        short = tenors[0]
        long = tenors[1]
        es = evectors.at[short, "PC1"]
        el = evectors.at[long, "PC1"]
        return [-bpvs[long]/bpvs[short]*el/es, 1]
    
    if len(tenors)==3:
        ## get belly and wings. Assume belly = 1.
        belly = tenors[1]
        wings = [w for w in tenors if w not in belly]
        ## get eigenvector subset
        
        #LHS
        wingFactors = pd.DataFrame(index=wings)
        wingFactors = wingFactors.join(evectors).ix[:,:-1].transpose()
        wingRisk = np.diag([bpvs[x] for x in wings])
        coeff = wingFactors.dot(wingRisk)
        invcoeff = np.linalg.pinv(coeff)
        
        #RHS
        bellyFactor = pd.DataFrame(index=[belly])
        bellyFactor = bellyFactor.join(evectors).ix[:,:-1].transpose()
            
        bellyFactor.loc[:] *= -bpvs[belly]
                
        wingWeights = list(np.ravel(invcoeff.dot(bellyFactor)))
        wingWeights.insert(1, 1)
        
        
        return wingWeights
    else:
        print("Oops. Not enough, or too many instruments.")
        
        
    return

def returnRiskWgts(trade, evectors):
    ### Returns BPVs to use in PCA weighted butterflies. 
    tenors = sorted([int(x.strip()) for x in trade.split("s") if x])
    tenors = [str(i)+"y" for i in tenors]
    ## re-index evectors:
    newIndex = [''.join(i for i in x if i.isdigit())+"y" for x in evectors.index.values]
    evectors.index = newIndex

    if len(tenors)==2:
        short = tenors[0]
        long = tenors[1]
        es = evectors.at[short, "PC1"]
        el = evectors.at[long, "PC1"]
        return [-el/es, 1]
    
    if len(tenors)==3:
        belly = tenors[1]
        wings = [w for w in tenors if w not in belly]
        
        wingFactors = pd.DataFrame(index=wings)
        wingFactors = wingFactors.join(evectors).ix[:,:-1].transpose()
        bellyFactor = pd.DataFrame(index=[belly])
        bellyFactor = bellyFactor.join(evectors).ix[:,:-1].transpose()
        
        invcoeff = np.linalg.pinv(wingFactors)
        bellyFactor.loc[:] *= -1
        rhs = invcoeff.dot(bellyFactor)
        rhs = list(np.ravel(rhs))
        rhs.insert(1, 1)
        
        return rhs
    else:
        print("Oops. Not enough, or too many instruments.")
    return





######## SAVE/DISPLAY TOOLS ##########



def pcaSnapshot(dates, data, lb=30, n=3,orient=True, vec=[], tol=1.5):
    # Returns tuple of evalues and a dict of evectors.
    reorient = len(dates)*[1]
    
    pcaResults = rollingPCA(data, lb=lb, n=n)
    orig = pcaResults["master"]
    evectors = pd.DataFrame(index=dates).join(orig)
    
    evalues = pd.DataFrame(index=dates).join(pcaResults["evalues"])

    pcCols = evalues.columns.values
    
    grouped = evectors.groupby(lambda x: x[:3], axis=1)
    
    accumEvecs = dict.fromkeys(pcCols, pd.DataFrame(columns=data.columns.values, index=data.index.values[lb:]))
    flag = 0
    for pc in pcCols:
        
        if flag == 0 and orient :
            flag = 1
            temp = grouped.get_group(pc).dropna()
            temp['orient'] = temp.applymap(np.sign).sum(axis=1)
            reorient = temp["orient"].apply(lambda x: (-1)**((np.abs(x-temp["orient"].mean())/temp["orient"].std()<tol)-1))
        
        factors = grouped.get_group(pc).dropna()
            
        accumEvecs[pc] = factors.mul(reorient, axis=0)
        accumEvecs[pc].rename(columns=lambda x: x[3:], inplace=True)
    
    return evalues, accumEvecs


