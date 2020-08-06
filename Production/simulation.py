# Simulation
# Live File to Simulate Stock Purcahses


class simpy:
    
    def __init__(self, balance, **kwargs):
        
        # Importing modules Modules
        import numpy as np
        import pandas as pd
        import time
        import math
        import sched
        import robin_stocks
        from datetime import datetime
        from twilio.rest import Client
        import os

        self.balance = balance
        self.smsUpdate = kwargs.get('smsUpdate',None)
        self.webUpdate = kwargs.get('webUpdate',None)
        self.currentHold = ""
        self.history = {'balance':balance, 'datetime',datetime.now()}

    # def loadPost():

        
    # Authenticating Robinhood profile
    def robinhoodLogin():
        global password
    
        # Authenticate
        robin_stocks.authentication.login(username='JustinM6711', password=password)

    # Stock update via sms
    def sendSmsUpdate(text):
        account_sid = 'ACa5e090aa63a4eacc1cb426134bc6f06f'
        auth_token = '22ce3e77765ac854ec64f46a97561530'
        client = Client(account_sid, auth_token)
        message = client.messages.create(from_ = '+12055092802',to = '+17742668896', body = text)

    # Function to buy stock
    def buy(ticker):
        global balance
        global shares
        global leftoverBalance
        global currentHold
            
        # Warn if holding
        if currentHold:
            print("Warning: Currently Holding " + currentHold)

        # Login
        robinhoodLogin()

        # Get stock price
        boughtPrice = float(robin_stocks.stocks.get_latest_price(str(ticker))[0])

        # Calculate max number of shares
        shares = math.floor(balance/boughtPrice)
        # Calculate leftover price from that exchange
        leftoverBalance = balance%boughtPrice

        # Send smsUpdate
        if smsUpdate:
            sendSmsUpdate("PURCHASE UPDATE: " + str(shares) + " of " + str(ticker) + ". Processed at " + str(datetime.now().strftime("%H:%M")) + " for $" + str(boughtPrice) + " per share. Portfolio Valuation is: $" + str(boughtPrice * shares + leftoverBalance))

        # Send webUpdate
        if webUpdate:
            sendWebUpdate()
        
        # Logout
        robin_stocks.authentication.logout()

    # Function to Sell All
    def sell():

        # Login
        robinhoodLogin()

        # Get sold price
        soldPrice = float(robin_stocks.stocks.get_latest_price(currentHold)[0])
        # Get new balance
        balance = shares * soldPrice + leftoverBalance
        
        # Logout
        robin_stocks.authentication.logout()
        

    # Function to update history arrays
    def updateHistory(balance):
        global histBalance
        global histTime
        
        history['balance'] = np.append(history['balance'], [balance])
        history['datetime'] = np.append(history['time'], [datetime.now()])



    # Function to return if market is open or not
    def marketIsOpen():
        # Getting times of open and close
        # (Temporary, will need to account for wknds/holidays)
        op = datetime(2020,1,1,9,30,0).time()
        cl = datetime(2020,1,1,16,0,0).time()

        # Getting current time
        currentTime = datetime.now().time()

        # Checking to see if time is in range
        isOpen = bool(op < currentTime and cl > currentTime)

        return isOpen  
