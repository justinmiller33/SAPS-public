# Modeling
# Static File to craft models off of data

import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt
from matplotlib import pyplot as plt
import random
import math
import warnings
import time
import re
import json


"""
DISCLAIMER:
This class is being modified from a preexisting model for a different data format
Many methods will not run on the public json data

OVERVIEW:
Creating a class to generalize a pipeline for different models
Includes:
0. Initializing Data Stream
1. Creating a set of function for profit definitions
2. Gathering supplemental data for feature engineering 
3. Formatting of data
3. ML Models
4. Ensembling

Notes:
- Time descriptions |t| are estimates and may vary based on the data
- CALLS details custom methods called in the function
- CALLS ignores __init__ and second-level functions
- REQS:  details methods that must be called before the function can be run
- REQS: ignores __init__

DIRECTORY:

0.0 __init__
(data,df,fData) : |4 Seconds / 100MB|
initializing data through the post df, financial data fData, and post/fData keys with data.keys
CALLS: loadJson
REQS: n/a

0.1 loadJson
(pathToJson) : |4 Seconds / 100 MB|
loading json file from local path and reserializing some pd series and dataframes
CALLS: n/a
REQS: n/a

0.2 initProfitDicts
() : |insignificant|
initializing (or resetting) profit and missed dicts to handle profit assignments
CALLS: n/a
REQS: n/a


1.0 sortByMinutes
(data,minutes) : |1 second|
getting profits for a certain time period after buying at time of post
REQS: n/a

1.1 sortByMinutesWithDrop
(data, minutes, lowDropProp, highDropProp) : |1 second|
getting profits for a certain time period after buying at time of post
REQS: n/a

1.2 sortBySellTime
(data,sellTime,buyTime) : |unknown|
getting profits for a certain range of buy to sell time
REQS: n/a

2.0 meansByMinutes
(data,maxMinutes) : |~(maxMinutes * 0.2 seconds)|
plotting and returning mean profits by time of hold from post
Internal REQS: sortByMinutes
External REQS: n/a

2.1 getSelftextLengths
(data, profits) : |milliseconds|
getting lengths of selftexts to use as a feature
Internal REQS: n/a
External REQS: sortByMinutes

3.0 splitPars
(data,ipOp) : |milliseconds|
taking properly formatted data and splitting into test and train
REQS: n/a

3.1 getSelftextWithClass
(data,profits) : |~1 minute per 1000 data points|
matching selftexts with the profit quartile and formatting for wordbag naive bayes
Internal REQS: n/a
External REQS: sortByMinutes

4.0 postLengthLinReg
(data,profits,strLen) : |~1 second per 1000 data points|
simple linear regression on length of string... plots maximum effectiveness of this policy
Internal REQS: splitPars
External REQS: sortByMinutes, getSelftextLengths

4.1 naiveBayes 
(data,stData) : |~1 minute per 1000 data points|
categorical naive bayes applied just to selftexts with profit quartiles
Internal REQS: splitPars
External REQS: getSelftextWithClass


NOTES:
selftext & text represent the same data: the body of the post in r/pennystocks.
Internal REQS: Functions called within the function.
External REQS: Functions that need to be manually called before function use

"""

class pipeline:

    """
    0.0-2
    Section for Initializing Class and Loading Data
    """
    
    # Initializing the pipeline to allow for analysis on subsets of the scraped data
    def __init__(self,pathToJson,**kwargs):
        self.data = pipeline.loadJson(pathToJson)
        self.subList = list(self.data.keys())
        self.subTemp = None
        self.finType = 'inter'
        self.profitDict = {}
        self.missedDict = {}
        
    # Default function to open master json file
    # Also processes certain data into pandas dataframes and series
    def loadJson(pathToJson):

        # Load data as pandas df
        data = pd.read_json(pathToJson)

        # Convert back to original dict
        data = data.to_dict()

        # Loop through each subreddit
        for subreddit in data.keys():

            # Convert post data into labeled dataframe
            data[subreddit]['raw']['postData'] = pd.DataFrame(data[subreddit]['raw']['postData'], columns = ["ticker","title","text","flair","unix"])

            # For each financial data type
            for finType in ["inter","intra"]:

                # Get the data and transform it into a pandas time series
                for key in list(data[subreddit]['raw'][finType].keys()):

                    newData = np.array(data[subreddit]['raw'][finType][key])
                    newData = pd.Series(data = newData[:,1], index = newData[:,0])
                    data[subreddit]['raw'][finType][key] = newData
                    
        return data

    # Initialize (or reset) profit dictionaries to assign profits for analysis
    def initProfitDicts(self):
        for sub in self.subList:
            self.profitDict[sub] = {}
            self.missedDict[sub] = []

    """
    1.0-1.1
    Minor Functions for Checks and Assertions
    """
    def assertInter(self):
        if self.finType != "inter":
            raise Exception("This Function can only be called with .finType = 'inter'")

    def assertIntra(self):
        if self.finType != "intra":
            raise Exception("This Function can only be called with .finType = 'intra'")

    """
    2.0-2
    Assigning Profits and Reorganizing within postData
    """

    # Assigning profits via hold time
    def holdForTime(self,days):

        # Assert that self.finType is inter
        self.assertInter()

        # For each subreddit
        for sub in self.subList:

            # For each key in the data
            for key in self.data[sub]['md'][self.finType]['keys']:

                # Get the profit time series for that post
                dataTemp = self.data[sub]['raw'][self.finType][key]

                # Try-except for recent posts that don't have that many days
                try:
                    # Get unique timestamps (only 1 per day)
                    dayListUnix = np.unique(dataTemp.index)
                    # Find the target stamp
                    targetUnix = dayListUnix[days]

                    # Profit is the last index for that day, assuming 4:00 sell
                    profit = 1 + dataTemp[targetUnix].iloc[-1]
                    self.profitDict[sub][key] = profit

                except:
                    # If there isn't enough data for a n-day hold, mark it as missed
                    self.missedDict[sub].append(key)

    # Function to cut postData to dataframe with profits from time
    # NOTE... can only run once per test without reloading json
    def cutMissedProfits(self):

        # Assert inter
        self.assertInter()
        
        # For each subreddit
        for sub in self.subList:

            # Convert indices to str to match keys to raw postData df
            saps.data[sub]['raw']['postData'].index = saps.data[sub]['raw']['postData'].index.astype(str)

            # Make new df keeping only posts with profits
            saps.data[sub]['raw']['postData'] = saps.data[sub]['raw']['postData'].loc[list(saps.profitDict[sub].keys())]


    # Adding profits to dataframe for easier manipulation
    def appendProfitsToDf(self):

        # Assert inter
        self.assertInter()

        # For each subreddit
        for sub in self.subList:

            # Create a new column with profit data
            saps.data[sub]['raw']['postData']['profit'] = list(saps.profitDict[sub].values())

    """
    3.0-1
    Supplementing postData and Removing Confounding Vars for ML algos
    """

    # Function to remove tickers from text to reduce confounding in naive bayes
    # Not possible to remove everything, will experience time-relevancy bias
    def removeTickersFromText(self):

        # For each subreddit
        for sub in self.subList:

            # Assign temporary dataframe
            dfTemp = saps.data[sub]['raw']['postData']

            # For each row in the df
            for rowNum in range(len(dfTemp)):

                # Removve the ticker from the text and/or title
                dfTemp['text'][rowNum] = dfTemp['text'][rowNum].replace(dfTemp.ticker[rowNum],"")
                dfTemp['title'][rowNum] = dfTemp['title'][rowNum].replace(dfTemp.ticker[rowNum],"")
            





























































    """
    1.0-1
    Section for Functions to parse which profits to test on
    Basis for modeling on different time frames for theoretical trading
    Dependent only on time of post
    """
    
    # Stand alone, sell exactly after a set amount of time
    # Theoretically cuts off buying with (minutes) left before 4PM
    def sortByMinutes(data,minutes):
        
        # Initializing blank dict
        profits = {}
        
        # For each key
        for key in data.keys:
            
            # Find profit % where datetime is (minutes) minutes more than start time
            # Declare start time as the first index in profits
            startTime = data.fData[key].index[0]

            # Declare the time target where we want to get our profit from 
            target = startTime + datetime.timedelta(0,0,0,0,minutes,0)

            # Try to find the index, if we can't find it raise exception
            # Neccesary check since yfinance has some missing minute data intervals for smaller stocks
            try:
                # Where the series indices match our target time
                index = np.where(data.fData[key].index == target)[0][0]

                # Get the profit at that index
                tempProfit = data.fData[key][index]

                # Adding that to our array of profits
                profits[key] = tempProfit

            except:
                # If np.where failed because data ran out after 4PM
                if target.hour >= 16:
                    warnings.warn("Some failed to hold for " + str(minutes) + " minutes due to market closing.")

                # If np.where failed because of inconsistent data
                else:
                    warnings.warn("Some failed to hold due to inconsistent data")

        # Warn about the number of values that failed
        warnings.warn(str(len(data.keys)-len(profits)) + " values failed out of " + str(len(data.keys)))

        # Convert profits to pd.Series
        profits = pd.Series(profits)

        # Merge profits with df
        # Make profit indices match df
        profits.index = profits.index.astype(int)

        # Make profits a labeled dataframe
        profits = pd.DataFrame(profits, columns = ['profits'])

        # Join and dropnans
        data.df = data.df.join(profits)
        data.df = data.df.dropna()

        return data.df

    # Sort by minutes but drop if price passes or dips below a price range
    def sortByMinutesWithDrop(data, minutes, lowDropProp, highDropProp):
        # Initializing blank dict
        profits = {}
        
        # For each key
        for key in data.keys:
            
            # Find profit % where datetime is (minutes) minutes more than start time
            # Declare start time as the first index in profits
            startTime = data.fData[key].index[0]

            # Declare the time target where we want to get our profit from 
            target = startTime + datetime.timedelta(0,0,0,0,minutes,0)

            # Getting series of current profits
            currentProfits = data.fData[key]
            
            # Try to find the index, if we can't find it raise exception
            # Neccesary check since yfinance has some missing minute data intervals for smaller stocks
            try:
                # Where the series indices match our target time
                index = np.where(currentProfits.index == target)[0][0]

                # Search to make sure no profits before that passed the drop price
                # At every minute
                for i in range(index):

                    # If the current profit at that time dips below drop
                    if currentProfits[i] <= lowDropProp or currenProfits[i] >= highDropProp:
                        
                        # That first recorded after drop is our profit
                        tempProfit = currentProfits[i]
                        
                        # Break to add to dict of profits
                        break

                    # If it isn't under the dropProp
                    else:
                        # Check to see if its the last index
                        if i == index - 1:

                            # If it's still running by now, put index as profit
                            tempProfit = currentProfits[index]

                # Adding that to our dict of profits
                profits[key] = tempProfit

            except:
                continue

        # Convert profits to pd.Series
        profits = pd.Series(profits)

        # Merge profits with df
        # Make profit indices match df
        profits.index = profits.index.astype(int)

        # Make profits a labeled dataframe
        profits = pd.DataFrame(profits, columns = ['profits'])

        # Join and dropnans
        data.df = data.df.join(profits)
        data.df = data.df.dropna()

        return data.df

    def buyOnCondition(data,balance):
        indices = df.index[::-1]
        balanceArray = np.array([balance])
        for indexLoc in range(len(indices)-1):
            fdat = fData[str(indices[indexLoc])]
            fdatInd = fdat.index.tz_localize(None)
            dTime = df.datetime[indices[indexLoc+1]]-fdatInd
            testCon = dTime[dTime > datetime.timedelta(seconds = 0)]

            if len(testCon)>0:
                endPriceArgBef = np.argmin(testCon)
                
            else:
                continue

            if len(dTime) > endPriceArgBef:
                endPriceArgAft = endPriceArgBef +1

            else:
                endPriceArgAft = endPriceArgAft

            try:
                endPriceAve = (fdat[endPriceArgBef] + fdat[endPriceArgAft]) /2
                #print(df.datetime[indices[indexLoc]])
                #print(fdatInd[endPriceArgBef],fdatInd[endPriceArgAft])
                #print(endPriceAve)
                balance = balance * (1 + endPriceAve)
                balanceArray = np.append(balanceArray, [balance])

            except:
                endPriceAve = fdat[-1]
                #print("STOPPED AT END OF DAY")
                #print(endPriceAve)
                balance = balance * (1 + endPriceAve)
                balanceArray = np.append(balanceArray, [balance])

        return balanceArray
            
    # Combinable, sell at the same time for all accumulated stocks before set period
    #def sortBySellTime(data,sellTime,buyTime):

    """
    2.0-1
    Supplemental Data to model
    """

    # Getting array of mean profits by minutes taken
    # To go with sortByMinutes
    def meansByMinutes(data, maxMinutes):

        # Initializing means array
        means = np.array([])

        # For each # of minutes in the range
        for i in range(maxMinutes):

            # Getting profits array from sortByMinutes
            profits = saps.sortByMinutes(i+1)

            # Appending it to means array
            means = np.append(means, [np.mean(df.profits)])

        # Show Visual
        plt.plot(means,'*')
        plt.show()

        return means
    

    
    """
    3.0-1
    Getting usable profit data
    """

    # Simple function to reshape input and output arrays into the common [len(data),2] matrix
    inputOutput = lambda data,ip,op : np.hstack((np.array(ip).reshape(len(ip),1), np.array(op).reshape(len(op),1)))
    
    # General function to split parallel arrays (in vertical matrix form [len(DATA),2]) to test (usually linreg)
    # With option to return test indices
    def splitPars(data,ipOp,rti):

        # Seperating Training Data
        # Inputting default training percentage
        trainPercent = 0.75

        # Getting range of indices
        inds = np.arange(len(ipOp))
        # Shuffling
        random.shuffle(inds)

        # Finding threshold using training percentile
        indThreshold = math.floor(trainPercent * len(inds))

        # Seperating train and test indices
        trainIndices = inds[:indThreshold]
        testIndices = inds[indThreshold:]

        # Assigning train and test data
        trainData = ipOp[trainIndices]
        testData = ipOp[testIndices]

        if rti:
            return trainData,testData,testIndices
        else:
            return trainData,testData
    
    
    # Get ticker-removed text with class for wordbag based text classification models    
    def getCategoricalClass(data):

        #Initializing category array
        cats = np.array([])

        # Using indices of profits as keys to the post data array
        for key in data.df.index:

            # Get selftext for each row (lowercased to reduce variation accounted for in other models)
            title = data.df.title[key].lower()

            # REMOVE TICKER NAME FROM SELFTEXT TO AVOID OVERFITTING
            tickerToRemove = data.df.ticker[key].lower()
            title = re.sub(tickerToRemove, 'TICKER', title)
            
            # Getting quartile markers to use to classify
            first = data.df.profits.describe()['25%']
            second = data.df.profits.describe()['50%']
            third = data.df.profits.describe()['75%']
            
            # Get profit classification as [NEGNEG, NEG, POS, POSPOS] for each row based on the quartile
            # Not neccesary positive or negative, but in that quarter of the results
            if data.df.profits[key] >= third:
                category = 'POSPOS'
            elif data.df.profits[key] >= second:
                category = 'POS'
            elif data.df.profits[key] >= first:
                category = 'NEG'
            else:
                category = 'NEGNEG'

            
            # Get row of text and category and stacking onto selftext data
            if df.title[key] != 'TICKER':
                data.df.title[key] = title
                cats = np.append(cats, [category])
            

        # Writing categoricals to new column in df
        cats = pd.DataFrame(cats, index=data.df.index, columns = ['category'])
        data.df = data.df.join(cats)
        
        return data.df
        
    
    """
    4.0-3
    Model Creation
    """
    
    # Simple regression for length of post
    # Associated with getSelfTextLenghts
    def postLengthLinReg(data):

        # Make input and output into matrix to use splitPars()
        # Helper for splitPars... brings input and output array together as vertical matrix
        
        ipOp = saps.inputOutput(data.df.textlen,data.df.profits)

        # Split into testing and training data
        trainData, testData = saps.splitPars(ipOp,False)

        """
        Data ready to train
        """
        
        # Import modules
        from sklearn.linear_model import LinearRegression

        # Define model class
        model = LinearRegression(fit_intercept=True)

        # Save training input data as x, training output data as y
        x = trainData[:,0]
        y = trainData[:,1]

        # Fit the model
        model.fit(x[:, np.newaxis], y)

        """
        Data ready to test
        """
        
        # Test the model
        # Save x values to predict
        xTest = testData[:,0]

        # Save actual results of those x values
        actuals = testData[:,1]

        # Get predicted values for the test data
        preds = model.predict(xTest[:,np.newaxis])

        # Getting general idea of feasability to use in ensemble
        # Testing how much of the reccomended policy we can use
        # Note that here it will be linear, will hardly tell us anything at all
        # But could be useful in an ADABOOST~esque ensembling method
        recPolicy = np.argsort(preds)[::-1]

        # Loop through minimum to max application of policy to find where its most effective
        meanPolicyProfit = np.array([])

        # For the entire length of our policy
        for i in range(len(recPolicy)):

            # Profit of i most recommended choices
            iRecProfit = np.mean(actuals[recPolicy[:i+1]])

            # Appending that to array
            meanPolicyProfit = np.append(meanPolicyProfit, [iRecProfit])
            
        # Plotting mean policy profits based on how much advice was taken

        plt.plot(meanPolicyProfit,'*')
        plt.show()

        return model

    def postLengthMultivarReg(data,profits,strLen,order):

        # Make input and output into matrix to use splitPars()
        # Helper for splitPars... brings input and output array together as vertical matrix
        
        ipOp = saps.inputOutput(strLen,profits)

        # Split into testing and training data
        (trainData, testData) = saps.splitPars(ipOp,False)

        """
        Data ready to train
        """

        # Importing modules
        from sklearn.preprocessing import PolynomialFeatures

        # Fittting regression model
        model = PolynomialFeatures(order)
        pol
        




    # Naive bayes classifier (pos or neg return)
    # Associated with getSelftextWithClass
    def naiveBayes(data):

        # Splitting data into test and train categories
        trainData,testData,testIndices = saps.splitPars(df[['title','category']].to_numpy().reshape((len(df),2)),True)
        
        """
        Data ready to train.
        """
        
        # Importing textblob modules
        from textblob.classifiers import NaiveBayesClassifier as NBC
        from textblob import TextBlob

        # Train... thank you textblob
        model = NBC(trainData)

        """
        Trained... moving to testing and predictions.
        """
        # Importing modules from skl
        from sklearn.metrics import confusion_matrix
        
        #Finding predictions
        preds = np.array([])
        # For each of the test data values
        for i in range(len(testData)):
            # Classification Result
            preds = np.append(preds,[model.classify(testData[i][0])])
        
        # Getting actuals to compare to predictions
        actuals = testData[:,1]

        print(confusion_matrix(actuals,preds))

        # Getting location and printing values of each predictions
        for cat in ['POSPOS','POS','NEG','NEGNEG']:
            loc = np.where(preds == cat)[0]
            indices = testIndices[loc]
            print(cat)
            print(df.profits.iloc[indices].describe())
        
        return model


    """
    5.0-4
    Building Ensemble
    # Final form, as a predict() function
    """


    """
    6.0
    Testing Model
    """
    # def simulate(testData,model):
        


# Input local data path
pathToJson = "/home/justinmiller/devel/SAPS-public/Data/saps.json"

# Initialize pipeline with json formatted data
saps = pipeline(pathToJson)

# Initialize profit dicts and metadata
saps.initProfitDicts()

# Get profits for a certain length hold
saps.holdForTime(10)

# Cut missed rows from dataframe and append profits
saps.cutMissedProfits()
saps.appendProfitsToDf()

# Remove tickers from text
saps.removeTickersFromText()

# Assign main minute value for testing
# minutes = 60

# Get profits appended ontp df
# df = saps.sortByMinutes(minutes)
# profs = saps.sortByMinutesWithDrop(minutes,-0.04)
"""
NEED TO REINITIALIZE PIPELINE WHENEVER DF IS UPDATED
"""
# saps = pipeline(df,fData)
# print(df.groupby('linked').describe())
"""
# Naive bayes only
df = saps.getCategoricalClass()
saps = pipeline(df,fData)
print('data organized, starting bayes')

model = saps.naiveBayes()
"""

# Linear for postlen only
# For showing lin reg policy effectiveness
# strLen = saps.getSelftextLengths(profits)
# model = saps.postLengthLinReg()
#plt.plot(np.cumsum(profits[np.argsort(strLen)][::-1]))
#plt.show()



