# Sentiment Analysis of Public Stockholders (SAPS) Data Extraction Software #
# ------------------------------------------------------------------------- #
# Justin Miller: College of Engineering, Northeastern University.
# In affiliation with Matthew Katz: Carroll School of Management, Boston College.

# Modules
import time
import numpy as np
import yfinance as yf
from datetime import datetime as dt
import datetime
import requests
import pandas as pd
import re
import pickle

# SAPS Class
class Saps:
    
    """
    Class to facilitate the extraction, matching, and organizing
    of stock reccomendations from various social media channels.
    Repeated on monthly intervals using crontab to update main
    dataset. (TODO) Link with jsonHandler.py and Kaggle API to
    implement a fully automated publising pipeline. 

    Properties:
        
    Saps.recentData:
        Actively declared dict of most recent json data
        Used when updating dataset

    (TODO)
    Saps.pathConfig:
        Dict of paths to facilitate use on external systems

    Methods:

    --Post-Related--
    crawlPage(subreddit, lastPage):
        crawls one page of reddits pushshift API call
    crawlSubreddit(subreddit, maxSubmissions):
        appends a page crawl onto previoius page calls with a
        set stop at the maximum submission number.
    extractTickers(string):
        extracts potential ticker callouts from string
    extractPost(num, channel, testName):
        extracts num recent posts from channel and saves 
        relevant information as df'testName'.npy

    --Financial-Related--
    getIntraday(rowNum,df):
        gets intraday trading data for the rowNum'th post in the dataframe
    getInterday(rowNum,df):
        gets intraday trading data for the rowNum'th post in the dataframe
    extractFinData(df):
        gets interday & intraday data for all posts in dataframe
        
    --Miscellaneous--
    loadDf(pathToFile)
        loads previously created subreddit dataframe .npy file
    
    Input Options:
    Reddit - Subreddits
    

    """
    # Function to crawl 1 page of the reddit api
    def crawlPage(subreddit: str, lastPage = None):

        # Initialize parameters
        params = {"subreddit": subreddit, "size": 500, "sort": "desc", "sort_type": "created_utc"}

        # If not first run through
        if lastPage is not None:

            # If last page had content
            if len(lastPage) > 0:
                # Continue scrape from 1 step ahead of last page
                params["before"] = lastPage[-1]["created_utc"]
                
            else:
              # Return no more data
              return []

        # Make request(returns just json)
        url = "https://api.pushshift.io/reddit/search/submission"
        results = requests.get(url, params)

        # If scrape fails raise exception
        if not results.ok:
            print("Server returned status code {}".format(results.status_code))
            print("Sleeping 1 minute then retrying")
            time.sleep(60)
            lastPage = Saps.crawlPage(subreddit, lastPage)
            return lastPage
        
        # Return relevant json
        return results.json()["data"]

    # Function to loop through multiple pushshift pages to crawl subreddit
    def crawlSubreddit(subreddit, maxSubmissions,killCond):

        # Initialize new submissions as empty list
        submissions = []
        # Reset lastPage
        lastPage = None

        # While temp page var isn't empty and we haven't hit the submission limit
        while lastPage != [] and len(submissions) < maxSubmissions:

            # Crawl a pushshift page
            lastPage = Saps.crawlPage(subreddit, lastPage)
            # Add lastPage to running submission list
            submissions += lastPage
            # Let API rest, update progress
            time.sleep(3)
            print("Minimum Progress: "+ str(100* len(submissions[:maxSubmissions])/maxSubmissions) + "%")

            # Running list of results
            ls = submissions[:maxSubmissions]
        
            # If the last result violates the killCond
            if ls[-1]['created_utc'] < killCond:

                # Getting the start point for our backtrack to find the killCond
                lenToLoop = len(ls) - 2
            
                # Work backwards from the list until the last adherent result is found
                while ls[lenToLoop]['created_utc'] < killCond:

                    # Iterate one up the list
                    lenToLoop = lenToLoop - 1

                # Trimming values after killCond off of ls
                ls = ls[:lenToLoop]

                lastPage = []
        
        return ls

    # Extracting all tickers in title of post
    # Identifying series' of three or four capital letters as potential tickers
    # Validation is done later in the process
    def extractTickers(string):

        # First step, getting all substrings with 3 or 4 letters ending in a space and starting with space or $
        tickers = np.array(re.findall(r'[$]\w{4}\b',string) + re.findall(r'[ ]\w{4}\b',string))
        tick3 = np.array(re.findall(r'[$]\w{3}\b',string) + re.findall(r'[ ]\w{3}\b',string))

        # print(tickers)
        
        # For each length, dropping the $ or _ taken from the regex
        tickers = np.array(list(map(lambda x:x[1:], tickers)))
        tick3 = np.array(list(map(lambda x:x[1:], tick3)))

        # print(tickers)
        # For each length, only taking the substrings that are fully capitilized
        tickers = tickers[list(map(lambda x:x.isupper(),tickers))]
        tick3 = tick3[list(map(lambda x:x.isupper(),tick3))]

        # Of those, only taking unique
        tickers = np.unique(tickers)
        tick3 = np.unique(tick3)

        # Initializing an array to delete subset redundancies
        toDelete = np.array([])

        # Loop through arrays finding where tick3 is a substring of a member of tickers
        for i in range(len(tick3)):
            for j in range(len(tickers)):
                if tick3[i] in tickers[j]:
                    toDelete = np.append(toDelete, i)

        # Delete redundancies from tick3
        tick3 = np.delete(tick3, toDelete.astype(int))

        # Appeding tick3 to tickers list
        tickers = np.append(tickers, tick3)
        
        # Update Output
        #print("Completed Ticker Extraction")
        #print(str(len(tickers)) + " Tickers")

        return tickers

        
    # Time convert function (utc to NYC)

    tc = lambda utc : dt.utcfromtimestamp(utc) - datetime.timedelta(hours = 4)

    """ Ticker validation and data collection """
    # Seeing if ticker is registered in yfinance API
    fetchTicker = lambda ticker : yf.Ticker(ticker)
    # Checking to see if time series data for ticker exists
    validateTicker = lambda ticker : np.sum(ticker.history().to_numpy()) !=0
    # Getting data for a ticker in a given time period from current date
    getData = lambda ticker,prd: ticker.history(period = prd)

    # Function to extract a number of posts from a given subreddit
    def extractPosts(num, channel, testName):

        # Starting timer
        start = time.time()

        # Setting a unix time as a killCond
        killCond = getKillCond(testName)
        
        # Getting data from subreddit
        ls = Saps.crawlSubreddit(channel,num,killCond)
        
        # Time taken to complete reddit scrape
        print(time.time()-start)
        
        # Extraction of singly titled posts
        # Initiating arrays for each data field
        tickerList = np.array([])
        titleList =  np.array([])
        textList =  np.array([])
        flairList =  np.array([])
        datetimeList = np.array([])

        
        # Loop through each post
        for i in range(len(ls)):

            # Getting possible tickers in string
            tickers = Saps.extractTickers(ls[i]["title"])
            # Remove repeat tickers
            tickers = np.unique(tickers)
            
            # For each ticker mentioned in that post
            for j in range(len(tickers)):

              # Trying to get ticker from yfinance
                try:
                    ticker = Saps.fetchTicker(tickers[j])

                # If it doesn't exist, continue to the next ticker    
                except:
                    continue

                #if ticker is invalid, continue to the next ticker
                if not Saps.validateTicker(ticker):
                    continue

                # Getting usable data: ticker, title, selftext, link_flair_text, time, time until open (hours) 
                # For consistency, only take titled posts, else continue
                if ls[i]['title']:
                    titleList = np.append(titleList, ls[i]['title'])
                    
                else:
                    continue
                
                # Appending ticker
                tickerList = np.append(tickerList, tickers[j])
              
                # Appending selftext, can be none
                try:
                    textList = np.append(textList, ls[i]['selftext'])
                except:
                    textList = np.append(textList, "None")
                
                # Appending flair, can be none
                try:
                    flairList = np.append(flairList, ls[i]['link_flair_text'])
                except:
                    flairList = np.append(flairList, "None")
                
                # Appending time as unix
                unix = ls[i]['created_utc']
                datetimeList = np.append(datetimeList, [Saps.tc(unix)])


        # Creating dataframe from np arrays
        df = pd.DataFrame(columns = ['ticker','title','text','type','datetime'])
        df.ticker = tickerList
        df.title = titleList
        df.text = textList
        df.type = flairList
        df.datetime = datetimeList

        # Saving dataframe with user defined file name
        np.save('/home/justinmiller/devel/SAPS-public/Data/df'+testName+'.npy',df)
        # Update
        print("loaded with " + str(len(df)) + " tickers... move to finData extraction.")

        return df

    # Function to load a dataframe with info on a subreddit
    def loadDf(pathToFile):
        df = np.load(pathToFile, allow_pickle = True)
        df = pd.DataFrame(df, columns = ['ticker','title','text','type','datetime'])

        return df

    # Function to load pickle files
    def readPickle(pathToFile):
        file = open(pathToFile, "rb")
        fData = pickle.load(file)
        file.close()

        return fData

    # Function to get intraday financial data for one post
    def getIntraday(rowNum,df):
        
        # Getting ticker object
        ticker = yf.Ticker(df.ticker[rowNum])

        # Getting the endtime for that day
        endTime = datetime.datetime(df.datetime[rowNum].year, df.datetime[rowNum].month, df.datetime[rowNum].day, 16,0,0)

        # Getting the minimum time of post to get data for that post
        minStart = datetime.datetime(df.datetime[rowNum].year, df.datetime[rowNum].month, df.datetime[rowNum].day, 9,30,0)
        
        # Getting that days history (5m intervals)
        try:

            # Only get data if post at or after 9:30
            if df.datetime[rowNum] >= minStart:
                
                # Get history between time of post and market closure
                hist = ticker.history(period = '1d', interval = '5m', start = df.datetime[rowNum], end = endTime)

                # Declaring start price.. round minute up for theoretical implementation
                startPrice = hist.Open[1]

                # Getting series of profits
                profits = (hist.Close - startPrice) / startPrice
                # Removing first profit (since we started on second minute)
                # Removing last profit due to glitch in yfinance
                # profits = profits.iloc[1:len(profits)-1]

            # If not, return no profits
            else:
                profits = np.array([False])

        # If fails, it's either missing profits or posted after market's closed. Return no profits.
        except:
            profits = np.array([False])

        return profits

        # Function to get intraday financial data for one post
    def getInterday(rowNum,df):
        
        # Getting ticker object
        ticker = yf.Ticker(df.ticker[rowNum])

        # Getting that days history (1h intervals)
        try:

            # Get history between time of post and market closure
            hist = ticker.history(period = 'max', interval = '1h', start = df.datetime[rowNum])

            # Declaring start price.. round minute up for theoretical implementation
            startPrice = hist.Open[1]

            # Getting series of profits
            profits = (hist.Close - startPrice) / startPrice
            # Removing first profit (since we started on second minute)
            # Removing last profit due to glitch in yfinance
            # profits = profits.iloc[1:len(profits)-1]

        
        # If fails, it's either missing profits or posted after market's closed. Return no profits.
        except:
            profits = np.array([False])

        return profits

    # Function to save dictionary of financial data as a pickle file
    def savePickle(fData,pathToFile):
        file = open("/home/justinmiller/devel/SAPS-public/Data/fData" + pathToFile + ".pkl","wb")
        pickle.dump(fData,file)
        file.close()
        
    # Extracting financial data for intraday and interday trades
    def extractFinData(df, fileName):
        fDataIntra = {}
        
        # For each row in the dataframe
        for rowNum in range(len(df)):
        
            # If profits gathered, add profits
            profits = Saps.getIntraday(rowNum,df)
            if profits.any():
                fDataIntra[str(rowNum)] = profits

            # Updating progress
            if rowNum % 100 == 50:
                print(str(rowNum/len(df)) + " of Intraday Loaded")
                print(len(fDataIntra))

        Saps.savePickle(fDataIntra, "Intra" + fileName)
        print("Intraday Saved")
        
        fDataInter = {}
        
        # For each row in the dataframe
        for rowNum in range(len(df)):
        
            # If profits gathered, add profits
            profits = Saps.getInterday(rowNum,df)
            if profits.any():
                fDataInter[str(rowNum)] = profits

            # Updating progress
            if rowNum % 100 == 50:
                print(str(rowNum/len(df)) + " of Interday Loaded")
                print(len(fDataInter))

        Saps.savePickle(fDataInter, "Inter" + fileName)
        print("Interday Saved")
        
        return fDataIntra, fDataInter

    # Check to kill processes once a certain datapoint is reached
    # NOTE: Will only be used in 'update' mode
    def getKillCond(testName):

        # WARNING: CURRENTLY USES LOCAL PATH TO DATA AS IT CAN NOT BE HELD ON GITHUB
        # Not currently a problem as project is being run on a single server
        pathToJson = "/home/justinmiller/Documents/OfflineDatasets/sapsRecent.json
        # Getting recent data
        # ERROR CHECK... want to fail here if data not found
        Saps.recentData = Saps.loadJson(pathToJson)

        # Try Except CHECK... shouldn't fail here if there isn't this dictpath
        # Just return 0 for the killCond unix
        try:
            # String in yyyy-mm-dd hh-mm-ss
            endDate = recentData[testName]['md']['postData']['endDate']

            # Converting to datetime
            endDate = dt.strptime(endDate, '%Y-%m-%d %H:%M:%S')

            # Getting killCond as unix
            killCond = int(endDate.timestamp())

        # Unix of killCond would mean we'd get data back until 01/01/1970
        except:
            killCond = 0

        return killCond
        
    # Default function to open master json file
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
        
    # Main loop to extract post and financial data
    def runExtract(num, channel, name):

        # Extract and save post data df
        df = Saps.extractPosts(num, channel, name)

        # Extract and save financial data pickles
        fDataIntra, fDataInter = Saps.extractFinData(df, name)

        return df, fDataIntra, fDataInter

""" ACTION """

# Max number of posts to scrape
num = 100000

# List of subreddits
#subList = ["stocks","StockMarket","daytrading","robinhood","RobinHoodPennyStocks"]
subList = ['investing']
# Naming list for each output
#nameList = ["Stocks","StockMarket","Daytrading","Robinhood","RobinHoodPennyStocks"]
nameList = ['Investing']


"""
# For each subreddit
for i in range(len(subList)):

    df,fDataIntra, fDataInter = Saps.runExtract(num, subList[i], nameList[i])
    
    print("------------------------------------")
    print(str(len(fDataIntra)) + " + " + str(len(fDataInter)) + " posts scraped for r/" + subList[i]) 
    print("------------------------------------")
    print(i)
"""
    

