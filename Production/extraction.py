# SAPS Data Extraction Software
# Justin Miller

# Modules
import time
import numpy as np
import yfinance as yf
from datetime import datetime as dt
import datetime
import requests
import pandas as pd
import re

# SAPS Class
class Saps:
    
    """
    Class to facilitate the extraction, matching, and organizing
    of stock data from various social media channels and accounts


    Functionality:
    
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
        

    Input Options:
    Reddit - Subreddits
    Twitter - Users, Hashtags (TODO)

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
            raise Exception("Server returned status code {}".format(results.status_code))

        # Return relevant json
        return results.json()["data"]

    # Function to loop through multiple pushshift pages to crawl subreddit
    def crawlSubreddit(subreddit, maxSubmissions):

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

        return submissions[:maxSubmissions]

    # Extracting all tickers in title of post
    # Identifying series' of three or four capital letters as potential tickers
    # Validation is done later in the process
    def extractTickers(string):

        # First step, getting all substrings with 3 or 4 letters ending in a space
        tickers = np.array(re.findall(r'\w{4}\b',string))
        tick3 = np.array(re.findall(r'\w{3}\b',string))

        # For each length, only taking the substrings that are fully capitilized
        tickers = tickers[list(map(lambda x:x.isupper(),tickers))]
        tick3 = tick3[list(map(lambda x:x.isupper(),tick3))]

        # Of those, only taking unique
        tickers = np.unique(tickers)
        tick3 = np.unique(tick3)

        # Initializing array to delete
        toDelete = np.array([])

        # Loop through arrays finding where tick3 is a substring of a member of tickers
        for i in range(len(tick3)):
            for j in range(len(tickers)):
                if tick3[i] in tickers[j]:
                    toDelete = np.append(toDelete, i)

        # Delete redundancies from tick3
        tick3 = np.delete(tick3, toDelete.astype(int))

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

        # Getting data from subreddit
        ls = Saps.crawlSubreddit(channel,num)
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
        np.save('df'+testName+'.npy',df)
        # Update
        print("loaded with " + str(len(df)) + " tickers... move to finData extraction.")

        return df

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

    # Extracting financial data for intraday and interday trades
    def extractFinData(df):
        fDataIntra = {}
        
        # For each row in the dataframe
        for rowNum in range(len(df)):
        
            # If profits gathered, add profits
            profits = Saps.getIntraday(rowNum,df)
            if profits.any():
                fDataIntra[str(rowNum)] = profits

            # Updating progress
            if rowNum % 100 == 50:
                print(rowNum/len(df))
                print(len(fDataIntra))
                
        fDataInter = {}
        
        # For each row in the dataframe
        for rowNum in range(len(df)):
        
            # If profits gathered, add profits
            profits = Saps.getInterday(rowNum,df)
            if profits.any():
                fDataInter[str(rowNum)] = profits

            # Updating progress
            if rowNum % 100 == 50:
                print(rowNum/len(df))
                print(len(fDataInter))

        return fDataIntra, fDataInter
