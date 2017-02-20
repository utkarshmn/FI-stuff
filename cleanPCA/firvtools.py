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
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, rc
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA, KernelPCA
from scipy.misc import comb


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


## Directories
cwd = os.getcwd()
dataDir = cwd+"/depo/Data/"
saveDir = cwd+"/depo/Outputs/"


## Available Currencies


def loadCurrencies(currencies):
    fwd = {}
    par = {}
    for ccy in currencies:
        fwd[ccy] = pd.read_pickle(dataDir+"fwd"+ccy)[::-1]
        temp = pd.read_pickle(dataDir+ccy)[::-1]
        if temp.columns.values[0][-1] != "y":
            temp.rename(columns=lambda x: str(x[3:])+"y", inplace=True)
        par[ccy] = temp
        print ("Loaded "+ccy)
    return fwd, par

def matchTenors(c1, c2):
    assets = [x for x in c1.columns.values if x in c2.columns.values]
    r1 = c1.copy().reindex(columns=[assets])
    r2 = c2.copy().reindex(columns=[assets])
    return r1, r2



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
  "7y3y": 253.2,
  "5y5y":432.75,
  "10y5y":380.54,
  "10y10y":714.89,
  "20y20y":1000
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

freqDesc = {
  "A-DEC": "Annual",
  "AS-JAN": "Annual",
  "B": "Daily",
  "BA-APR": "Monthly",
  "BA-AUG": "Monthly",
  "BA-DEC": "Monthly",
  "BA-FEB": "Monthly",
  "BA-JAN": "Monthly",
  "BA-JUL": "Monthly",
  "BA-JUN": "Monthly",
  "BA-MAR": "Monthly",
  "BA-MAY": "Monthly",
  "BA-NOV": "Monthly",
  "BA-OCT": "Monthly",
  "BA-SEP": "Monthly",
  "BAS-JAN": "Semi-Annual",
  "BM": "EOM",
  "BQ-FEB": "Quarterly",
  "BQ-JAN": "Quarterly",
  "BQ-MAR": "Quarterly",
  "L": "ms",
  "Q-DEC": "Quarterly",
  "T": "Min",
  "U": "us",
  "W-FRI": "Weekly",
  "W-MON": "Weekly",
  "W-SAT": "Weekly",
  "W-SUN": "Weekly",
  "W-THU": "Weekly",
  "W-TUE": "Weekly",
  "W-WED": "Weekly"
}


###### PCA TOOLS ######


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

def staticPCA(data, n=3, freq=1, corrW = 20, autoOrient=2):
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
    results["covM"] = covM[::-1]
    
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

def cleanPCs(input, smoothing=0, sdev = 2):
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
        newdf = newdf[newdf.apply(lambda x: np.abs(x - x.mean()) / x.std() < sdev).all(axis=1)]
        newdf = newdf[newdf.apply(lambda x: np.abs(x - x.mean()) / x.std() < sdev).all(axis=1)]

        
    return newdf



####### Trade Related #######

def getWgts(trade, evectors,bpvs=0):
    ### Returns risk to use in trades
    eVec = evectors.copy()
    if "-" in trade:
        tenors = trade.split("-")
        
    else:
        tenors = sorted([int(x.strip()) for x in trade.split("s") if x])
        tenors = [str(i)+"y" for i in tenors]
        ## re-index evectors:
        newIndex = [''.join(i for i in x if i.isdigit())+"y" for x in evectors.index.values]
        eVec.index = newIndex

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
        wingFactors = wingFactors.join(eVec).ix[:,:-1].transpose()
        bellyFactor = pd.DataFrame(index=[belly])
        bellyFactor = bellyFactor.join(eVec).ix[:,:-1].transpose()
        invcoeff = np.linalg.pinv(wingFactors)
        bellyFactor.loc[:] *= -1
        rhs = invcoeff.dot(bellyFactor)
        rhs = [2*x for x in list(np.ravel(rhs))]
        rhs.insert(1, 2)
        
        return rhs
    else:
        print("Oops. Not enough, or too many instruments.")
    return

def genTrade(trade, data, pcaResult):
    eVec = pcaResult["evecs"].copy()
    
    
    if "-" in trade:
        tenors = trade.split("-")
        
    else:
        tenors = sorted([int(x.strip()) for x in trade.split("s") if x])
        tenors = [str(i)+"y" for i in tenors]
        ## re-index evectors:
        newIndex = [''.join(i for i in x if i.isdigit())+"y" for x in eVec.index.values]
        eVec.index = newIndex
        
    pcaWgts = getWgts(trade, eVec)
    
    result = data[tenors].copy()
    result["PCA "+trade] = result.dot(pcaWgts)
    
    if len(tenors) == 2:
        result["Standard "+trade] = result[tenors[1]] - result[tenors[0]]
    elif len(tenors)==3:
        legs = data[tenors].copy()
        result["Standard "+trade] = legs.dot(np.array([-1, 2, -1]))
    
    residuals = pcaResult["resid"][tenors].copy()
    residuals["PCA "+trade+" Resid."] = residuals.dot(pcaWgts)
    if len(tenors) == 2:
        residuals["Standard "+trade+" Resid."] = residuals[tenors[1]] - residuals[tenors[0]]
    elif len(tenors)==3:
        rlegs = residuals[tenors].copy()
        residuals["Standard "+trade+" Resid."] = rlegs.dot(np.array([-1, 2, -1]))    
    
    
    return result, residuals

def genStdParTrade(trade, data):
    result = pd.DataFrame(index=data.index.values)
    if trade.count("y")==1:
        return data[trade]
    elif trade.count("s")==2:
        tenors = sorted([int(x.strip()) for x in trade.split("s") if x])
        tenors = [str(i)+"y" for i in tenors]
        result[trade] = data[tenors[1]] - data[tenors[0]]
    elif trade.count("s")==3:
        tenors = sorted([int(x.strip()) for x in trade.split("s") if x])
        tenors = [str(i)+"y" for i in tenors] 
        legs = data[tenors].copy()
        result[trade] = legs.dot(np.array([-1, 2, -1]))
    return (result)



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

def getDataSubset(raw, start, end, scale = 1, resample=1):
    subset = raw[start:end] *scale
    subset = subset.iloc[::resample, :]
    return subset

def plotResidual(residTS, ax, k=3):
    assets = residTS.columns.values
    xvals = range(len(assets))

    relVal = pd.DataFrame()
    relVal = residTS.ix[[-1, -5, -20, -60]]
    relVal.index = ["1d", "1w", "1m", "3m"]
    relVal = relVal.transpose()
    
    
    
    lday = ax.bar(xvals, relVal["1d"], alpha=0.5,color='#256189', align='center', width=0.5, label='1d')
    ax.set_xticks(xvals)
    ax.set_xticklabels(assets)
    ax.grid()
    s = 25
    lwk = ax.scatter(xvals, relVal["1w"], color='#B20F2C', marker = "o", s=2*s, label='1w')
    lmnth = ax.scatter(xvals, relVal["1m"], color='b', marker = "x", s=1.5*s, label="1m")
    lquart = ax.scatter(xvals, relVal["3m"], color='g', marker = "D", s=1.5*s, label="3m")
    ax.axhline(0, color="grey")
    
    ax.set_axis_bgcolor('#f8fcff')
    ax.set_ylabel("Residual (bps)")
    ax.set_title(str(k)+" Factor Residuals")

    return ([lday, lwk, lmnth, lquart])

def plotPC(pca, ax, k=3, rawPx = None, comp=None):
    import matplotlib.dates as mdates
    myFmt = mdates.DateFormatter('%b %y')
    ax.patch.set_facecolor('#FAFAFA')
    ax.patch.set_alpha(1.0)


    pcx = pca["facTS"].index.values
    pcts = pca["facTS"]["PC"+str(k)]
    plt.xticks(rotation=30)
    pcplt = ax.plot(pcx, pcts, 'g-', label="PC"+str(k)+" - "+"%0.2f"%pcts.ix[-1, 0])
    ax.xaxis.set_major_formatter(myFmt)
    ax.grid()
    ax.axhline(0, color="grey")
    ax.set_title("PC"+str(k)+" vs. "+comp)

    if len(rawPx.columns.values) > 1 and comp != None:
        plt.xticks(rotation=30)
        cTrade = genStdParTrade(comp, rawPx)
        axr = ax.twinx()
        tradePlt = axr.plot(pcx, cTrade, 'b--', label=comp+" (rhs) - "+"%0.2f"%cTrade.ix[-1,0])
        lns = tradePlt+pcplt
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0, frameon=True, framealpha=0.7)
        return

    ax.legend(frameon=True, framealpha=0.7)
    return


def plotSummary(ccy, ctype, start, end, parD, fwdD=None, 
    tenors=None, scale=1, resample=1, pcCorr=20, AR=2, cTrades = ["7y", "5s30s", "2s5s10s"]):
    par = parD
    fwd = fwdD
    import seaborn as sns


    ## Check multiCurrency ###
    if ccy.count("-")==1:
        c1, c2 = ccy.split("-")
        stDate = start
        edDate = end
        ctype = ctype.upper()

        try:
            testCcyLoad = par[c1]
            testCcyLoad = par[c2]
        except:
            print ("Currencies not loaded properly.")
            return

        if ctype == "PAR":
            try:
                rawPar1 = getDataSubset(par[c1], stDate, edDate, scale, resample)
                rawPar2 = getDataSubset(par[c2], stDate, edDate, scale, resample)
                rawPar1, rawPar2 = matchTenors(rawPar1, rawPar2)
                if tenors != None:
                    rawPar1 = rawPar1.reindex(columns=[tenors])
                    rawPar2 = rawPar2.reindex(columns=[tenors])
                rawPar = rawPar1.subtract(rawPar2)

            except:
                print ("Invalid Tenors.")
                return
            pca1 = staticPCA(rawPar, freq=1, n=1, corrW = pcCorr)
            pca2 = staticPCA(rawPar, freq=1, n=2, corrW = pcCorr)
            pca3 = staticPCA(rawPar, freq=1, n=3, corrW = pcCorr, autoOrient=AR)
            cDesc = "Par"
        elif ctype == "FWD":
            try:
                rawPar1 = getDataSubset(par[c1], stDate, edDate, scale, resample)
                rawPar2 = getDataSubset(par[c2], stDate, edDate, scale, resample)
                rawPar1, rawPar2 = matchTenors(rawPar1, rawPar2)
                rawFwd1 = getDataSubset(fwd[c1], stDate, edDate, scale, resample)
                rawFwd2 = getDataSubset(fwd[c2], stDate, edDate, scale, resample)
                rawFwd1, rawFwd2 = matchTenors(rawFwd1, rawFwd2)
                if tenors != None:
                    rawFwd1 = rawFwd1.reindex(columns=[tenors])
                    rawFwd2 = rawFwd2.reindex(columns=[tenors])
                    
                rawPar = rawPar1.subtract(rawPar2)
                rawFwd = rawFwd1.subtract(rawFwd2)
            except:
                print ("Invalid Tenors.")
                return
            pca1 = staticPCA(rawFwd, freq=1, n=1, corrW = pcCorr)
            pca2 = staticPCA(rawFwd, freq=1, n=2, corrW = pcCorr)
            pca3 = staticPCA(rawFwd, freq=1, n=3, corrW = pcCorr, autoOrient=AR)
            cDesc = "Forwards"
        else:
            print ("Incorrect Curve Type.")
            return        
    else:
        stDate = start
        edDate = end
        ctype = ctype.upper()

        try:
            testCcyLoad = par[ccy]
        except:
            print ("Currencies not loaded properly.")
            return

        if ctype == "PAR":
            try:
                rawPar = getDataSubset(par[ccy], stDate, edDate, scale, resample)
                if tenors != None:
                    rawPar = rawPar.reindex(columns=[tenors])

            except:
                print ("Invalid Tenors.")
                return
            pca1 = staticPCA(rawPar, freq=1, n=1, corrW = pcCorr)
            pca2 = staticPCA(rawPar, freq=1, n=2, corrW = pcCorr)
            pca3 = staticPCA(rawPar, freq=1, n=3, corrW = pcCorr)
            cDesc = "Par"
        elif ctype == "FWD":
            try:
                rawPar = getDataSubset(par[ccy], stDate, edDate, scale, resample)
                rawFwd = getDataSubset(fwd[ccy], stDate, edDate, scale, resample)
                print (rawFwd.columns.values)
                if tenors != None:
                    rawPar = rawPar.reindex(columns=[tenors])
            except:
                print ("Invalid Tenors.")
                return
            pca1 = staticPCA(rawFwd, freq=1, n=1, corrW = pcCorr)
            pca2 = staticPCA(rawFwd, freq=1, n=2, corrW = pcCorr)
            pca3 = staticPCA(rawFwd, freq=1, n=3, corrW = pcCorr)
            cDesc = "Forwards"
        else:
            print ("Incorrect Curve Type.")
            return
    ######################################################################
    
    
    fig = plt.figure(figsize=(16, 24), dpi=400)    
    plt.style.use('seaborn-white')

    ## Cov ax
    ax1 = plt.subplot2grid((5, 3), (4, 0), colspan=2)
    covM = pca3["covM"]
    sns.heatmap(covM/covM.values.max()*100,annot=True, linewidths=.25, fmt='.0f', 
                cmap = cm.YlGnBu, vmin=0, vmax=100, ax=ax1)
    ax1.set_title("Covariance Matrix")
    
    
    ## eigenvealues ######################################################
    ax2 = plt.subplot2grid((5, 3), (4, 2), colspan=1)
    
    nevals = pca3["evals"]
    cumExp = np.cumsum(nevals)
    
    ax2.bar(range(3), nevals, alpha=0.5, align='center', label='individual expl. var.')
    ax2.step(range(3), cumExp, where='mid', label='cumulative expl. var.')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["PC1", "PC2", "PC3"])
    ax2.set_xlabel('Principal Components')
    ax2.set_ylabel('Explained Variance')
    ax2.legend(loc="best")
    ax2.set_title("Explained Variance",)
    ax2.grid()

    rects = ax2.patches
    labels = [str(round(i, 1))+" %" for i in nevals]

    for rect,label in zip(rects, labels):
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2, height + 3, label, ha='center', va='bottom')

    ## eigenvectors ######################################################
    
    ax3 = plt.subplot2grid((5, 3), (3, 0), colspan=2)
    evTable = pca3["evecs"]
    
    ax3.grid()
    bwidth = 0.25
    assets = evTable.index.values
    adjpos = [-bwidth, 0, bwidth]
    colors = ['#2980b9', '#58D3F7', '#16a085']
    ind = np.arange(len(assets))
    for col in range(len(evTable.columns)):
        ax3.bar(ind+adjpos[col], evTable.ix[:,col],alpha=0.75, width=bwidth,color=colors[col], align="center")
    ax3.set_xlabel('Maturity')
    ax3.set_ylabel('Sensitivity')
    ax3.set_xticks(range(len(assets)))
    ax3.set_xticklabels(assets)
    ax3.legend(["PC1", "PC2", "PC3"], loc="lower right")
    ax3.set_title("Eigenvectors")
   
    ## residuals #########################################################
    
    PCAs = [pca1, pca2, pca3]

    for i in range(len(PCAs)):
        ax = plt.subplot2grid((5, 3), (len(PCAs)-i-1, 0), colspan=2, rowspan=1)
        residLines = plotResidual(PCAs[i]["resid"], ax, k=i+1)

    ###############################################################################
    import matplotlib.dates as mdates
    myFmt = mdates.DateFormatter('%b %y')

    ## Correlations
    corrs = pca3["facCR"]
    #print (corrs)
    ax7 = plt.subplot2grid((5, 3), (3, 2), colspan=1, rowspan=1)
    corrx = corrs.index.values
    ax7.plot(corrx, corrs.ix[:,0], corrx, corrs.ix[:, 1], corrx, corrs.ix[:, 2])
    ax7.grid()
    lastCorrs = [corrs.ix[-1, i] for i in range(len(corrs.columns.values))]
    lgnd = [x + " (%0.1f)"%(y) for x, y in zip(corrs.columns.values, lastCorrs)]
    ax7.legend(lgnd, loc="lower left",frameon=True, framealpha=0.7)
    ax7.patch.set_facecolor('#F4AFAB')
    ax7.patch.set_alpha(0.2)
    ax7.axhline(0, color="grey")
    ax7.set_title(str(pcCorr)+" period rolling correlation")
    plt.xticks(rotation=30)
    ax7.xaxis.set_major_formatter(myFmt)

    for i in range(len(PCAs)):
        ax = plt.subplot2grid((5, 3), (len(PCAs)-1-i, 2), colspan=1, rowspan=1)
        plotPC(pca3, ax, i+1, rawPar, cTrades[i])
        
    freq = freqDesc[pd.infer_freq(pca3["rawDat"].index)]
        
    fig.suptitle("%s PCA Summary"%
                 (ccy+" "+cDesc), fontsize=22, fontweight="bold", 
                 bbox={'facecolor':'black', 'alpha':0.2, 'pad':18})
    fig.subplots_adjust(top=0.94)

    fig.text(0.5, 0.965,'%s to %s (%s)'%(stDate.strftime("%d %b %y"), edDate.strftime("%d %b %y"), freq+" data"), 
             ha='center', va='center', transform=ax3.transAxes, fontsize=10)
    plt.figlegend(residLines, ["1p", "5p", "20p", "60p"], frameon=True,loc="upper center",
                 bbox_to_anchor=[0.095, 0.84], framealpha=0.65)
    
    
    
    return fig, pca3



