import dash
import numpy as np
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import pandas_datareader as pdr
import datetime
from datetime import date
import requests
import math
import plotly.graph_objects as go
import plotly
from bs4 import BeautifulSoup
import yfinance as yf

def Profile(selected_dropdown_value):
    CompanyCode = selected_dropdown_value    
    Code = yf.Ticker(selected_dropdown_value)
    Info = Code.info
    outputlist = []
    outputlist.append(Info['longBusinessSummary'])
    return outputlist

def Valuations(selected_dropdown_value, EBITDA_Growth, DiscountRate, FreeCashGrowth):
    outputlist = []
    CompanyCode = selected_dropdown_value
    Code = yf.Ticker(selected_dropdown_value)
    Cashflow = Code.cashflow
    EBITDA_Growth = float(EBITDA_Growth/100)
    DiscountRate = float(DiscountRate/100)
    FreeCashGrowth = float(FreeCashGrowth/100)
    try:
        EBITDA_Multiple = round(Code.info['enterpriseValue']/Code.info['ebitda'],3)
        EBITDA_Forward = Code.info['ebitda']*(1+EBITDA_Growth)
        Net_Debt = Code.info['totalDebt'] - Code.info['totalCash']
        ENT = (EBITDA_Multiple*EBITDA_Forward) - Net_Debt
        Value = round(ENT/Code.info['sharesOutstanding'],3)
        outputlist.append(("The (EBITDA) valuation for ", CompanyCode, " is: ",Value))
        outputlist.append(("The current price is: ",Code.info['regularMarketPrice']))
        outputlist.append(("The return of the (EBITDA) valuation is: ", round(((Value-Code.info['regularMarketPrice'])/Code.info['regularMarketPrice'])*100,3),"%"))
    except:
        outputlist.append(("Financial Services and Banks cannot have EBITDA models"))
    FreeCashFlow = Code.info['freeCashflow']
    if FreeCashFlow == None:
        FreeCashFlow = Cashflow.loc['Total Cash From Operating Activities'][0]-Cashflow.loc['Capital Expenditures'][0]
    FreeCashValue = round(((FreeCashFlow*(1+FreeCashGrowth))/(DiscountRate-FreeCashGrowth))/Code.info['sharesOutstanding'],3)
    outputlist.append(("The (FCF) valuation for ", CompanyCode, " is: ",FreeCashValue))
    outputlist.append(("------- Further Ratios Below ----------"))
    outputlist.append(("The PE Ratio is: ", round(Code.info['forwardPE'],3)))
    try:
        outputlist.append(("The payout ratio is: ", round(Code.info['payoutRatio'],3)))
    except:
        outputlist.append(("Cannot obtain the payout ratio"))
    try:
        outputlist.append(("Institutional holders make up: ", round(Code.info['heldPercentInstitutions'],3)))
    except:
        outputlist.append(("Cannot obtain the institutional holder value"))
    try:
        outputlist.append(("Insider holders make up: ", round(Code.info['heldPercentInsiders'],3)))
    except:
        outputlist.append(("Cannot obtain the insider holder value"))
    try:
        outputlist.append(("The percentage of shares short is: ",round(Code.info['sharesPercentSharesOut'],4)))
    except:
        outputlist.append(("Cannot obtain the short percentage"))
    return outputlist

def Fundamentals(selected_dropdown_value):
    CompanyCode = selected_dropdown_value
    try:
        stocktoday = pdr.get_data_yahoo(selected_dropdown_value,start=date.today(), end=date.today())
    except:
        stocktoday = pdr.get_data_yahoo(selected_dropdown_value,start=(datetime.datetime.now() - datetime.timedelta(days=4)), end=(datetime.datetime.now() - datetime.timedelta(days=4)))
    CurrentPrice = stocktoday['Close'][0]
    
    Code = yf.Ticker(selected_dropdown_value)
    Financials = Code.financials
    Balances = Code.balance_sheet
    Cashflow = Code.cashflow
    Info = Code.info
    count = 0
    Number_Of_Shares = []
    Net_Income = []
    Shareholder_Equity = []
    Total_Assets = []
    Total_Liability = []
    Gross_Margin = []
    Operating_Cashflow = []
    Freecashflow = []
    EBITDA = []
    EV = []
    Net_Debt = []
    ROE = []
    ROA = []
    DTE = []
    EPS = []
    dex = []
    Interest_Expense = []
    Long_Debt = []
    Longer_Debt = []
    PS_DIV = []
    while count < 4:
        if Financials.loc['Interest Expense'][count] == None:
            Interest_Expense.append(0)
        else:
            Interest_Expense.append(Financials.loc['Interest Expense'][count])
        try:
            Long_Debt.append(Balances.loc['Short Long Term Debt'][count])
        except:
            Long_Debt.append(0)
        try:
            Longer_Debt.append(Balances.loc['Long Term Debt'][count])
        except:
            Longer_Debt.append(0)
        Number_Of_Shares.append(round(float(Info['sharesOutstanding']/1000),2))
        Net_Income.append(float(Financials.loc['Net Income'][count]/1000))
        Shareholder_Equity.append(float(Balances.loc['Total Stockholder Equity'][count]/1000))
        Total_Assets.append(Balances.loc['Total Assets'][count])
        Total_Liability.append(Balances.loc['Total Liab'][count])
        Gross_Margin.append(round((((Financials.loc['Total Revenue'][count]-Financials.loc['Cost Of Revenue'][count])/Financials.loc['Total Revenue'][count])),3))
        Operating_Cashflow.append(float(Cashflow.loc['Total Cash From Operating Activities'][count]/1000))
        Freecashflow.append(float(((Operating_Cashflow[count]*1000)+Cashflow.loc['Capital Expenditures'][count])/1000))
        EBITDA.append(float(((Net_Income[count]*1000)-Interest_Expense[count]+Financials.loc['Income Tax Expense'][count]+Cashflow.loc['Depreciation'][count])/1000))
        Net_Debt.append(Long_Debt[count]+Longer_Debt[count]-Balances.loc['Cash'][count])
        ROE.append(round(Net_Income[count]/Shareholder_Equity[count],3))
        ROA.append(round((Net_Income[count]*1000)/Total_Assets[count],3))
        DTE.append(round(Total_Liability[count]/(Shareholder_Equity[count]*1000),3))
        EPS.append(round(Net_Income[count]/Number_Of_Shares[count],3))
        dex.append(date.today().year-count)
        try:
            PS_DIV.append(round(abs(Cashflow.loc['Dividends Paid'][count]/(Number_Of_Shares[count]*1000)),3))
        except:
            PS_DIV.append(0)
        count = count + 1
    df = pd.DataFrame({CompanyCode: dex, 'EPS': EPS,'ROE': ROE, 'ROA': ROA, 'Net Income ("000)': Net_Income,
                       'Debt to Equity':DTE, 'Shareholder Equity ("000)': Shareholder_Equity,'Gross Margin':Gross_Margin,
                       'Operating Cashflow ("000)':Operating_Cashflow, 'Free Cashflow ("000)': Freecashflow, 'EBITDA ("000)': EBITDA, 'Dividend P/S':PS_DIV})
    return df

#This is the part of the code that takes buy points and displays them
def FibonacciGrapher(CompanyCode, SellDate, SellPrice, SellDate1, SellPrice1, BuyPrice, BuyDate): 
    CompanyCode = CompanyCode
    x = 0
    y = 0
    w = 0
    differenceidentify = 0
    count = 125
    High = []
    Low = []
    HighValue = []
    LowValue = []
    HighValue2 = []
    LowValue2 = []
    LowValue3 = []
    HighValue3 = []
    stock = pdr.get_data_yahoo(CompanyCode,start=datetime.datetime(2018,2,2), end=date.today(),interval="m")
    days = stock['Close'].count()
    close_prices = stock['Close']
    df1 = pd.DataFrame(stock, columns=['Close','Open','High','Low'])
    periods = math.floor(days / 100)
    breh = 0
    maxposition = 0
    while x < periods:
        breh = close_prices[(days -1 - (x+1)*100):((days-1) - (x * 100))]
        maxposition = np.where(breh == breh.max())
        maxposition = maxposition[0][0]
        if maxposition > 94:
            differenceidentify = 100 - maxposition + 2
        else:
            differenceidentify = 0
        High.append(max(close_prices[(days - 1 - (x+1)*100):((days-1) - (x * 100) + differenceidentify)]))
        Low.append(min(close_prices[(days -1 - (x+1)*100):((days-1) - (x * 100))]))
        x = x + 1
        if ((x+1)*100 == days):
            x = x+2
        differenceidentify = 0
    df1 = df1.dropna()
    while y < (days-count):
        HighValue.append(None)
        LowValue.append(None)
        y = y + 1
    while w < count:
        HighValue.append(round(float(High[0]), 2))
        LowValue.append(round(float(Low[0]), 2))
        w = w + 1
    y = 0
    w = 0
    while y < (days - (2*count)):
        HighValue2.append(None)
        LowValue2.append(None)
        y = y + 1
    while w < count*2:
        HighValue2.append(round(float(High[1]), 2))
        LowValue2.append(round(float(Low[1]), 2))
        w = w + 1
    y = 0
    while y < (days):
        LowValue3.append(round(float(Low[2]), 2))
        HighValue3.append(round(float(High[2]), 2))
        y = y + 1
    df1['Low3'] = LowValue3
    df1['High3'] = HighValue3
    df1['HFib'] = HighValue
    df1['LFib'] = LowValue
    df1['HFib2'] = HighValue2
    df1['LFib2'] = LowValue2
    df1['MMA'] = df1.rolling(window=50).mean()['Close']
    df1['SMA'] = df1.rolling(window=20).mean()['Close']
    df1['LMA'] = df1.rolling(window=200).mean()['Close']
    df1['20 Day Volatility'] = df1['Close'].rolling(window=20).std()
    df1['Top Bollinger Band']=df1['SMA']+2*df1['20 Day Volatility']
    df1['Bottom Bollinger Band']=df1['SMA']-2*df1['20 Day Volatility']
    df1['Bollinger Band Difference']=df1['Top Bollinger Band']-df1['Bottom Bollinger Band']
    x = 1
    BolRet = [0]
    while x < days:
        BolRet.append((df1['Bollinger Band Difference'][x]-df1['Bollinger Band Difference'][x-1])/df1['Bollinger Band Difference'][x-1])
        x = x + 1
    x = 1
    Daily_Return = [0]
    while x < days:
        Daily_Return.append((df1['Close'][x]-df1['Close'][x-1])/df1['Close'][x-1])
        x = x+1
    df1['Daily Return']=Daily_Return
    df1['252 Day Volatility'] = df1['Daily Return'].rolling(window=252).std()
    df1['180 Day Volatility'] = df1['Daily Return'].rolling(window=180).std()
    df1['60 Day Volatility'] = df1['Daily Return'].rolling(window=60).std()
    df1['30 Day Volatility'] = df1['Daily Return'].rolling(window=20).std()
    df1['Annual_Volatility252'] = (df1['252 Day Volatility'])*(252**(1/2))
    df1['Annual_Volatility180'] = (df1['180 Day Volatility'])*(252**(1/2))
    df1['Annual_Volatility60'] = (df1['60 Day Volatility'])*(252**(1/2))
    df1['Annual_Volatility20'] = (df1['30 Day Volatility'])*(252**(1/2))
    df1['Bollinger Diff Return'] = BolRet
    df1['CUNT']=df1['Bollinger Diff Return'].mean()
    df1['BUTT']=df1['CUNT']-df1['Bollinger Diff Return'].std()
    x = 25
    BollingerDate = []
    Under = []
    Over = []
    Bollinger = []
    while x < days-5:
        if df1['Annual_Volatility20'][x]<df1['Annual_Volatility60'][x]:
            if df1['Annual_Volatility20'][x]<df1['Annual_Volatility180'][x]:
                if df1['Annual_Volatility20'][x]<df1['Annual_Volatility252'][x]:
                    if df1['Annual_Volatility20'][x-1]>df1['Annual_Volatility60'][x-1] or df1['Annual_Volatility20'][x-1]>df1['Annual_Volatility180'][x-1] or df1['Annual_Volatility20'][x-1]>df1['Annual_Volatility252'][x-1]:
                        Under = df1['Bollinger Diff Return'][x-5:x+1]
                        Underposition = np.where(Under == Under.min())
                        Underposition = Underposition[0][0]
                        Over = df1['Bollinger Diff Return'][x-4+Underposition:x+2+Underposition]
                        Overposition = np.where(Over == Over.max())
                        Overposition = Overposition[0][0]
                        if df1['Bollinger Diff Return'][x-5+Underposition]<df1['BUTT'][x]:
                               if df1['Bollinger Diff Return'][x-4+Underposition+Overposition]>0:
                                    BollingerDate.append(df1.index.date[x-4+Underposition+Overposition])
                                    Bollinger.append((df1['High'][x-4+Underposition+Overposition])*1.05)
        Under = df1['Bollinger Diff Return'][x-5:x+1]
        Underposition = np.where(Under == Under.min())
        Underposition = Underposition[0][0]
        Over = df1['Bollinger Diff Return'][x-4+Underposition:x+2+Underposition]
        Overposition = np.where(Over == Over.max())
        Overposition = Overposition[0][0]
        if df1['Bollinger Diff Return'][x-5+Underposition]<df1['BUTT'][x]:
               if df1['Bollinger Diff Return'][x-4+Underposition+Overposition]>0:
                   if min(df1['Annual_Volatility20'][x-4:x-4+Underposition+Overposition+1])<min(df1['Annual_Volatility60'][x-4:x-4+Underposition+Overposition+1]):
                       if min(df1['Annual_Volatility20'][x-4:x-4+Underposition+Overposition+1])<min(df1['Annual_Volatility180'][x-4:x-4+Underposition+Overposition+1]):
                           if min(df1['Annual_Volatility20'][x-4:x-4+Underposition+Overposition+1])<min(df1['Annual_Volatility252'][x-4:x-4+Underposition+Overposition+1]):
                                BollingerDate.append(df1.index.date[x-4+Underposition+Overposition])
                                Bollinger.append((df1['High'][x-4+Underposition+Overposition])*1.05)
        x = x + 1
    fig = go.Figure()
    king = 0
    if "." in CompanyCode:
        king = ('Australian Market - '+ CompanyCode)
    else:
        king = ('US Market - '+ CompanyCode)
    fig = go.Figure(data=[go.Candlestick(x=df1.index,open=df1['Open'],high=df1['High'],low=df1['Low'],close=df1['Close'])])
    fig.add_trace(go.Scatter(x=df1.index,y=df1['Bottom Bollinger Band'], mode = 'lines',marker=dict(size=1, color="purple"),showlegend=False))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['Top Bollinger Band'], mode = 'lines',fill='tonexty',fillcolor='rgba(173,204,255,0.2)',marker=dict(size=1, color="purple"),showlegend=False))
    fig.update_layout(xaxis_rangeslider_visible=False, width = 1000, height = 700,title=king, showlegend=False)
    df2 = pd.DataFrame(data = {'Dates':BuyDate,'BuyPrice':BuyPrice})
    fig.add_trace(go.Scatter(x=df2['Dates'],y=df2['BuyPrice'], mode = 'markers',marker=dict(size=12, color="green"),showlegend=False))
    df3 = pd.DataFrame(data = {'Dates':SellDate,'SellPrice':SellPrice})
    fig.add_trace(go.Scatter(x=df3['Dates'],y=df3['SellPrice'], mode = 'markers',marker=dict(size=12, color="Red"),showlegend=False))
    df4 = pd.DataFrame(data = {'Dates1':SellDate1,'SellPrice1':SellPrice1})
    fig.add_trace(go.Scatter(x=df4['Dates1'],y=df4['SellPrice1'], mode = 'markers',marker=dict(size=12, color="Orange"),showlegend=False))
    fig.update_xaxes(dtick="M2",tickformat="%d\n%b\n%Y")
    fig.add_trace(go.Scatter(x=df1.index,y=df1['LMA'], mode = 'lines',marker=dict(size=1, color="orange"),showlegend=False))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['SMA'], mode = 'lines',marker=dict(size=1, color="dark grey"),showlegend=False))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['HFib'], mode = 'lines',marker=dict(size=1, color="purple"),showlegend=False))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['HFib2'], mode = 'lines',marker=dict(size=1, color="purple"),showlegend=False))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['High3'], mode = 'lines',marker=dict(size=1, color="purple"),showlegend=False))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['LFib'], mode = 'lines',marker=dict(size=1, color="black"),showlegend=False))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['LFib2'], mode = 'lines',marker=dict(size=1, color="black"),showlegend=False))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['Low3'], mode = 'lines',marker=dict(size=1, color="black"),showlegend=False))
    df4 = pd.DataFrame(data = {'BolDates':BollingerDate,'BolPrice':Bollinger})
    fig.add_trace(go.Scatter(x=df4['BolDates'],y=df4['BolPrice'], mode = 'markers',marker=dict(size=12, color="black"), marker_symbol = "arrow-bar-down",showlegend=False))
    return fig


#This is the part of the code that takes buy points and displays them
def BollingerBands(selected_dropdown_value): 
    CompanyCode = selected_dropdown_value
    stock = pdr.get_data_yahoo(CompanyCode,start=datetime.datetime(2018,1,1), end=date.today())
    days = stock['Close'].count()
    close_prices = stock['Close']
    df1 = pd.DataFrame(stock, columns=['Close','Open','Low','High'])
    df1['20 Day Volatility'] = df1['Close'].rolling(window=20).std()
    df1['LMA'] = df1.rolling(window=20).mean()['Close']
    df1['Top Bollinger Band']=df1['LMA']+2*df1['20 Day Volatility']
    df1['Bottom Bollinger Band']=df1['LMA']-2*df1['20 Day Volatility']
    df1['Bollinger Band Difference']=df1['Top Bollinger Band']-df1['Bottom Bollinger Band']
    df1['Bol Dif MA'] = df1['Bollinger Band Difference'].rolling(window=200).mean()
    df1['Bol Dif STD'] = df1['Bollinger Band Difference'].rolling(window=200).mean()+ df1['Bollinger Band Difference'].rolling(window=200).std()
    df1['Bol Dif NEG STD'] = df1['Bollinger Band Difference'].rolling(window=200).mean() - df1['Bollinger Band Difference'].rolling(window=200).std()
    x = 1
    BolRet = [0]
    while x < days:
        BolRet.append((df1['Bollinger Band Difference'][x]-df1['Bollinger Band Difference'][x-1])/df1['Bollinger Band Difference'][x-1])
        x = x + 1
    x = 1
    Daily_Return = [0]
    while x < days:
        Daily_Return.append((df1['Close'][x]-df1['Close'][x-1])/df1['Close'][x-1])
        x = x+1
    df1['Daily Return']=Daily_Return
    df1['252 Day Volatility'] = df1['Daily Return'].rolling(window=252).std()
    df1['180 Day Volatility'] = df1['Daily Return'].rolling(window=180).std()
    df1['60 Day Volatility'] = df1['Daily Return'].rolling(window=60).std()
    df1['30 Day Volatility'] = df1['Daily Return'].rolling(window=20).std()
    df1['Annual_Volatility252'] = (df1['252 Day Volatility'])*(252**(1/2))
    df1['Annual_Volatility180'] = (df1['180 Day Volatility'])*(252**(1/2))
    df1['Annual_Volatility60'] = (df1['60 Day Volatility'])*(252**(1/2))
    df1['Annual_Volatility20'] = (df1['30 Day Volatility'])*(252**(1/2))
    df1['Bollinger Diff Return'] = BolRet
    df1['CUNT']=df1['Bollinger Diff Return'].mean()
    df1['BUTT']=df1['CUNT']-df1['Bollinger Diff Return'].std()
    x = 25
    BuyDate = []
    Under = []
    Over = []
    Buy = []
    while x < days-5:
        if df1['Annual_Volatility20'][x]<df1['Annual_Volatility60'][x]:
            if df1['Annual_Volatility20'][x]<df1['Annual_Volatility180'][x]:
                if df1['Annual_Volatility20'][x]<df1['Annual_Volatility252'][x]:
                    if df1['Annual_Volatility20'][x-1]>df1['Annual_Volatility60'][x-1] or df1['Annual_Volatility20'][x-1]>df1['Annual_Volatility180'][x-1] or df1['Annual_Volatility20'][x-1]>df1['Annual_Volatility252'][x-1]:
                        Under = df1['Bollinger Diff Return'][x-5:x+1]
                        Underposition = np.where(Under == Under.min())
                        Underposition = Underposition[0][0]
                        Over = df1['Bollinger Diff Return'][x-4+Underposition:x+2+Underposition]
                        Overposition = np.where(Over == Over.max())
                        Overposition = Overposition[0][0]
                        if df1['Bollinger Diff Return'][x-5+Underposition]<df1['BUTT'][x]:
                               if df1['Bollinger Diff Return'][x-4+Underposition+Overposition]>0:
                                    BuyDate.append(df1.index.date[x-4+Underposition+Overposition])
                                    Buy.append(df1['Bollinger Band Difference'][x-4+Underposition+Overposition])
        Under = df1['Bollinger Diff Return'][x-5:x+1]
        Underposition = np.where(Under == Under.min())
        Underposition = Underposition[0][0]
        Over = df1['Bollinger Diff Return'][x-4+Underposition:x+2+Underposition]
        Overposition = np.where(Over == Over.max())
        Overposition = Overposition[0][0]
        if df1['Bollinger Diff Return'][x-5+Underposition]<df1['BUTT'][x]:
               if df1['Bollinger Diff Return'][x-4+Underposition+Overposition]>0:
                   if min(df1['Annual_Volatility20'][x-4:x-4+Underposition+Overposition+1])<min(df1['Annual_Volatility60'][x-4:x-4+Underposition+Overposition+1]):
                       if min(df1['Annual_Volatility20'][x-4:x-4+Underposition+Overposition+1])<min(df1['Annual_Volatility180'][x-4:x-4+Underposition+Overposition+1]):
                           if min(df1['Annual_Volatility20'][x-4:x-4+Underposition+Overposition+1])<min(df1['Annual_Volatility252'][x-4:x-4+Underposition+Overposition+1]):
                                BuyDate.append(df1.index.date[x-4+Underposition+Overposition])
                                Buy.append(df1['Bollinger Band Difference'][x-4+Underposition+Overposition])
        x = x + 1
    fig = go.Figure()                                                                                                            
    king = 0
    if "." in CompanyCode:
        king = ('Bollinger Band Difference - '+ CompanyCode)
    else:
        king = ('Bollinger Band Difference - '+ CompanyCode)
    fig.add_trace(go.Scatter(x=df1.index,y=df1['Bollinger Band Difference'], mode = 'lines',marker=dict(size=12, color="blue"), marker_symbol = "star-diamond",showlegend=False))
    fig.update_layout(xaxis_rangeslider_visible=False, width = 1000, height = 400,title=king, showlegend=False)
    fig.update_xaxes(dtick="M2",tickformat="%d\n%b\n%Y")
    fig.update_yaxes(title = "Bollinger Band Difference")
    fig.add_trace(go.Scatter(x=df1.index,y=df1['Bol Dif MA'], mode = 'lines',marker=dict(size=12, color="green"),showlegend=True,name="200 Bol MA"))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['Bol Dif STD'], mode = 'lines',marker=dict(size=12, color="purple"),showlegend=True,name="200 Bol STD"))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['Bol Dif NEG STD'], mode = 'lines',marker=dict(size=12, color="purple"),showlegend=True,name="200 Bol NEG STD"))
    df2 = pd.DataFrame(data = {'Dates':BuyDate,'BuyPrice':Buy})
    fig.add_trace(go.Scatter(x=df2['Dates'],y=df2['BuyPrice'], mode = 'markers',marker=dict(size=12, color="red"), marker_symbol = "arrow-bar-down",showlegend=True,name="Significants"))
    return fig

def Oppie_Calc(selected_dropdown_value,input1, input2, input3, input4, input5, input6, input7,stockers):
    strike = float(input2)
    risk = float(input4)
    Annual_Volatility = float(input7)
    Stock_Price = stockers
    delta_time = (input3/252)/input6
    u = math.exp(Annual_Volatility*(delta_time**(1/2)))
    d = 1/u
    p = (math.exp(risk*delta_time)-d)/(u-d)
    S_Far = []
    S_Near = []
    F_Far = []
    F_Near = []
    x = 0
    Option_Value = 0
    Early_Exercise_Value = 0
    while x <= input6:
        S_Far.append(Stock_Price*(u**(input6-x))*(d**x))
        if input1.upper() == 'C':
            F_Far.append(max((S_Far[x] - strike),0))
        if input1.upper() == 'P':
            F_Far.append(max((strike - S_Far[x]),0))
        x = x + 1
    x = 1
    counter = 0
    delta = 0
    while x <= input6:
        while counter <= (input6 - x):
            S_Near.append(Stock_Price*(u**(input6-x-counter))*(d**counter))
            Option_Value = math.exp(-risk*delta_time)*(p*F_Far[counter]+(1-p)*F_Far[counter+1])
            if input1.upper() == 'C' and input5.upper() == 'A':
                Early_Exercise_Value = S_Near[counter] - strike
            if input1.upper() == 'P' and input5.upper() == 'A':
                Early_Exercise_Value = strike - S_Near[counter]
            else:
                Early_Exercise_Value = 0
            F_Near.append(max(Option_Value,Early_Exercise_Value,0))
            counter = counter + 1
        delta = (F_Far[0]-F_Far[1])/((Stock_Price*u)-(Stock_Price*d))
        S_Near = []
        F_Far = F_Near
        F_Near = []
        counter = 0
        x = x + 1
    Premium = F_Far[0]
    return Premium


#This is the part of the app for producing the main info and buy/sell points
def TradingAlgo(selected_dropdown_value, junky, signalinput):
    factor = junky
    CompanyCode = selected_dropdown_value
    stock = pdr.get_data_yahoo(CompanyCode,start=datetime.datetime(2018,2,2), end=date.today(),interval="m")
    days = stock['Close'].count()
    df1 = pd.DataFrame(stock, columns=['Close','Open','High','Low','Volume'])
    x = 1
    Daily_Return = [0]
    while x < days:
        Daily_Return.append((df1['Close'][x]-df1['Close'][x-1])/df1['Close'][x-1])
        x = x+1
    df1['Daily Return']=Daily_Return
    df1['30 Day Volatility'] = df1['Daily Return'].rolling(window=30).std()
    Annual_Volatility30 = (df1['30 Day Volatility'][len(df1['30 Day Volatility'])-1])*(252**(1/2))
    df1['Annual_Volatility30'] = (df1['30 Day Volatility'])*(252**(1/2))
    df1['26 EMA'] = df1.ewm(span = 26, min_periods = 26).mean()['Close']
    df1['12 EMA'] = df1.ewm(span = 12, min_periods = 12).mean()['Close']
    df1['MACD'] = df1['12 EMA'] - df1['26 EMA']
    df1['MACD Ave'] = df1['MACD'].mean()
    df1['Signal Line'] = df1.ewm(span = 9, min_periods = 9).mean()['MACD']
    AbsTP = []
    x = 0
    y = 0
    z = 0
    w = 0
    PosRatio = 0
    NegRatio = 0
    Positive = []
    Negative = []
    MFR = []
    Equat = 0
    MFI = [50,50,50,50,50,50,50,50,50,50,50,50,50,50]
    df1['Typical Price'] = (df1['Close'] + df1['High'] + df1['Low'])/3
    AbsTP.append(df1['Typical Price'].iloc[x])
    while x < (days - 1):
        if df1['Typical Price'].iloc[(x+1)] > df1['Typical Price'].iloc[x]:
            AbsTP.append(df1['Typical Price'].iloc[(x+1)])
        else:
            AbsTP.append((df1['Typical Price'].iloc[(x+1)])*(-1))
        x = x + 1
    df1['Abs TP'] = AbsTP
    df1['Raw Money'] = df1['Abs TP'] * df1['Volume']
    while y < days:
        if df1['Raw Money'].iloc[y] > 0:
            Positive.append(df1['Raw Money'].iloc[y])
            Negative.append(0)
        else:
            Negative.append(df1['Raw Money'].iloc[y])
            Positive.append(0)
        y = y + 1
    while z < 14:
        PosRatio = PosRatio + Positive[z]
        NegRatio = NegRatio + Negative[z]
        z = z + 1
    while z < days:
        MFR.append((PosRatio/(-1*NegRatio)))
        PosRatio = PosRatio - Positive[(z - 14)] + Positive[z]
        NegRatio = NegRatio - Negative[(z - 14)] + Negative[z]
        z = z + 1
    while w < len(MFR):
        Equat = 100 - (100/(1+MFR[w]))
        MFI.append(Equat)
        w = w + 1
    df1['MFI'] = MFI
    df1['Mid Line'] = df1['MFI'].mean()
    df1['SELL'] = df1['Mid Line'] + df1['MFI'].std()
    df1['BUYER'] = df1['Mid Line'] - df1['MFI'].std()
    df1['SMA'] = df1.rolling(window=20).mean()['Close']
    df1['LMA'] = df1.rolling(window=200).mean()['Close']
    df1['20 Day Volatility'] = df1['Close'].rolling(window=20).std()
    df1['Top Bollinger Band']=df1['SMA']+2*df1['20 Day Volatility']
    df1['Bottom Bollinger Band']=df1['SMA']-2*df1['20 Day Volatility']
    df1['Midway'] = (df1['Top Bollinger Band']+df1['SMA'])/2
    x = 1
    MACD_Return = [0]
    Signal_Return = [0]
    while x < days:
        MACD_Return.append((df1['MACD'][x]-df1['MACD'][x-1])/df1['MACD'][x-1])
        Signal_Return.append((df1['Signal Line'][x]-df1['Signal Line'][x-1])/1)
        x = x+1
    df1['MACD Ret'] = MACD_Return
    df1['Signal Ret'] = Signal_Return
    x = 30
    SellDate = []
    SellPrice = []
    SellDate1 = []
    SellPrice1 = []
    BuyDate = []
    BuyPrice = []
    outputlist = []
    BuyCounter = 0
    BuyReturn = 0
    OtherBuyReturn = 0
    OtherBuyCounter = 0
    SellReturn1 = 0
    SellCounter1 = 0
    SellReturn2 = 0
    SellCounter2 = 0
    SellReturn3 = 0
    SellCounter3 = 0
    SellReturn1P = 0
    minposition = 0
    maxposition = 0
    BuyC = []
    BuyC2 = []
    SellP = []
    SellP2 = []
    while x < days:
        if df1['Open'][x]>df1['Top Bollinger Band'][x]:
            if df1['Low'][x]<df1['Top Bollinger Band'][x]:
                if df1['Close'][x] <= df1['Open'][x]:
                    if df1['MACD'][x]>=df1['Signal Line'][x]:
                        SellDate1.append(df1.index.date[x])
                        SellPrice1.append(df1['Close'][x])
                        if x + 11>days:
                            print("")
                        else:
                            SellReturn1 = SellReturn1 + ((min(df1['Low'][x+1:x+9])-df1['High'][x])/df1['High'][x])
                            Valnear = df1['Low'][x+1:x+9]
                            minposition = np.where(Valnear == Valnear.min())
                            minposition = minposition[0][0]
                            initial_premium = Oppie_Calc(CompanyCode,"P",df1['High'][x],30,0.02,"A",30,df1['Annual_Volatility30'][x],df1['High'][x])
                            high_premium = Oppie_Calc(CompanyCode,"P",df1['High'][x],(30-minposition),0.02,"A",30,df1['Annual_Volatility30'][(x+minposition)],df1['Low'][x+minposition])
                            SellP.append((high_premium-initial_premium)/initial_premium)
                            SellP2.append(SellCounter1)
                            SellCounter1 = SellCounter1 + 1
        if df1['MACD'][x]>df1['Signal Line'][x]:
            if df1['MACD'][x]<df1['MACD'][x-1]:
                if df1['MACD'][x]>df1['MACD Ave'][x]:
                    if df1['MFI'][x]>df1['SELL'][x]:
                        if (abs(df1['MACD'][x])-abs(df1['Signal Line'][x]))<(abs(df1['MACD'][x-1])-abs(df1['Signal Line'][x-1])):
                            if (abs(df1['MACD'][x-1])-abs(df1['Signal Line'][x-1]))<(abs(df1['MACD'][x-2])-abs(df1['Signal Line'][x-2])):
                                if df1['Signal Ret'][x]<df1['Signal Ret'].std()+df1['Signal Ret'].mean():
                                    if df1['Close'][x]>df1['Midway'][x]:
                                        SellDate.append(df1.index.date[x])
                                        SellPrice.append(df1['Close'][x])
                                        if x + 11>days:
                                            print("")
                                        else:
                                            SellReturn2 = SellReturn2 + ((min(df1['Low'][x+1:x+9])-df1['High'][x])/df1['High'][x])
                                            SellCounter2 = SellCounter2 + 1
        if df1['Low'][x-1]>df1['Top Bollinger Band'][x-1]:
            if df1['High'][x]<df1['Top Bollinger Band'][x]:
                if df1['Close'][x] <= df1['Open'][x]:
                    if df1['MFI'][x]>=df1['Mid Line'][x]:
                        SellDate.append(df1.index.date[x])
                        SellPrice.append(df1['Close'][x])
                        if x + 11>days:
                            print("")
                        else:
                            SellReturn3 = SellReturn3 + ((min(df1['Low'][x+1:x+9])-df1['High'][x])/df1['High'][x])
                            SellCounter3 = SellCounter3 + 1
                        
        if df1['Low'][x]<df1['Bottom Bollinger Band'][x]:
            if df1['Low'][x]<(df1['Low'][x-1])*0.99:
                if df1['Low'][x]<(df1['Low'][x-2])*0.99:
                    BuyDate.append(df1.index.date[x])
                    BuyPrice.append(df1['Low'][x])
                    if x + 11>days:
                        print("")
                    else:
                        BuyReturn = BuyReturn + ((max(df1['High'][x+1:x+9])-df1['Low'][x])/df1['Low'][x])
                        Valnear = df1['High'][x+1:x+9]
                        maxposition = np.where(Valnear == Valnear.max())
                        maxposition = maxposition[0][0]
                        initial_premium = Oppie_Calc(CompanyCode,"C",df1['Low'][x],30,0.02,"A",30,df1['Annual_Volatility30'][x],df1['Low'][x])
                        high_premium = Oppie_Calc(CompanyCode,"C",df1['Low'][x],(30-maxposition),0.02,"A",30,df1['Annual_Volatility30'][(x+maxposition)],df1['High'][x+maxposition])
                        BuyC.append((high_premium-initial_premium)/initial_premium)
                        BuyC2.append(BuyCounter)
                        BuyCounter = BuyCounter + 1

        x = x + 1
    x = 30
    while x < days-30:
        high = max(df1['High'][x-29:x])
        low = min(df1['Low'][x-29:x])
        if df1['Close'][x] < high*0.9:
            OtherBuyReturn = OtherBuyReturn + ((max(df1['High'][x+1:x+9])-df1['Close'][x])/df1['Close'][x])
            OtherBuyCounter = OtherBuyCounter + 1
            x = x + 29
        x = x + 1
    df5 = pd.DataFrame({'Counter':BuyC2,'Call Return':BuyC})
    df6 = pd.DataFrame({'Counter':SellP2,'Put Return':SellP})
    outputlist.append(("""The buy strategy works off buying the low when it goes below the bottom bollinger band. As long as
                        the low is 1% lower than the last 2 lows."""))
    try:
        outputlist.append(("Buy Strategy (green dots):  ",round(BuyReturn/BuyCounter,4)))
        outputlist.append(("Buy Strategy (green dots) with call options:  ",round(df5['Call Return'].mean(),4)))
        outputlist.append(("Buy Strategy (green dots) with call options (0.25 Quartile):  ",round(df5['Call Return'].quantile(q=0.25),4)))
    except:
        outputlist.append(("Buy Strategy (green dots):  N/A"))
    try:
        outputlist.append(("Buy/Hold Strategy - Buy every 30 days if 10% under last 30 day high:  ",round(OtherBuyReturn/OtherBuyCounter,4)))
    except:
        outputlist.append(("Buy/Hold Strategy - Buy every 30 days if 10% under last 30 day high:  N/A"))
    try:
        outputlist.append(("(Orange) Sell strategy where the sell is triggered whenever the open is above the top band and the low is below the top band. The next 10 day low return is:  ",round(SellReturn1/SellCounter1,4)," and count: ",SellCounter1))
        outputlist.append(("(Orange) Put options return in the next 10 days is (Mean):  ",round(df6['Put Return'].mean(),4)))
        outputlist.append(("(Orange) Put options return in the next 10 days is (0.25 Quartile):  ",round(df6['Put Return'].quantile(q=0.25),4)))
    except:
        outputlist.append(("Sell strategy where the sell is triggered whenever the open is above the top band and the low is below the top band. The next 10 day low return is:  N/A"))
    try:  
        outputlist.append(("Sell strategy where the MACD and MFI are combined along with the use of the bolinger bands. The next 10 day low return is:  ",round(SellReturn2/SellCounter2,4)," and count: ",SellCounter2))
    except:
        outputlist.append(("Sell strategy where the MACD and MFI are combined along with the use of the bolinger bands. The next 10 day low return is:  N/A"))
    try:
        outputlist.append(("Sell strategy where yesterday the tick was totally above the top bollinger band and today the high was below. The next 10 day low return is:  ",round(SellReturn3/SellCounter3,4)," and count: ",SellCounter3))
    except:
        outputlist.append(("Sell strategy where yesterday the tick was totally above the top bollinger band and today the high was below. The next 10 day low return is: N/A"))
    
    if factor == 'bitch': 
        bloop = FibonacciGrapher(CompanyCode, SellDate, SellPrice, SellDate1, SellPrice1, BuyPrice, BuyDate)
        return bloop
    else:
        return outputlist

    
def MACD_BuySignal_graphed(selected_dropdown_value):
    CompanyCode = selected_dropdown_value
    stock = pdr.get_data_yahoo(CompanyCode,start=datetime.datetime(2019,9,1), end=date.today())
    days = stock['Close'].count()
    df2 = pd.DataFrame(stock, columns = ['Close'])
    df2['26 EMA'] = df2.ewm(span = 26, min_periods = 26).mean()['Close']
    df2['12 EMA'] = df2.ewm(span = 12, min_periods = 12).mean()['Close']
    df2['MACD'] = df2['12 EMA'] - df2['26 EMA']
    df2['Bro Line'] = df2['MACD'].mean()
    df2['Signal Line'] = df2.ewm(span = 9, min_periods = 9).mean()['MACD']
    df2 = df2.dropna()
    fig = go.Figure()
    if "." in CompanyCode:
        king = ('MACD Graph - '+ CompanyCode)
    else:
        king = ('MACD Graph - '+ CompanyCode)
    fig.add_trace(go.Scatter(x=df2.index,y=df2['MACD'], mode = 'lines',marker=dict(size=1, color="red"),showlegend=False))
    fig.update_xaxes(dtick="M2",tickformat="%d\n%b\n%Y")
    fig.add_trace(go.Scatter(x=df2.index,y=df2['Signal Line'], mode = 'lines',marker=dict(size=1, color="green"),showlegend=False))
    fig.add_trace(go.Scatter(x=df2.index,y=df2['Bro Line'], mode = 'lines',marker=dict(size=1, color="dark red"),showlegend=False))
    fig.update_layout(title=king,xaxis_title="Time",yaxis_title="MACD Value", width=750, height = 550)
    return fig 


def MoneyFlowIndex(selected_dropdown_value):
    AbsTP = []
    x = 0
    y = 0
    z = 0
    w = 0
    PosRatio = 0
    NegRatio = 0
    Positive = []
    Negative = []
    MFR = []
    Equat = 0
    MFI = [50,50,50,50,50,50,50,50,50,50,50,50,50,50]
    CompanyCode = selected_dropdown_value
    stock = pdr.get_data_yahoo(CompanyCode,start=datetime.datetime(2019,9,1), end=date.today())
    days = stock['Close'].count()
    df2 = pd.DataFrame(stock)
    df2['Typical Price'] = (df2['Close'] + df2['High'] + df2['Low'])/3
    AbsTP.append(df2['Typical Price'].iloc[x])
    while x < (days - 1):
        if df2['Typical Price'].iloc[(x+1)] > df2['Typical Price'].iloc[x]:
            AbsTP.append(df2['Typical Price'].iloc[(x+1)])
        else:
            AbsTP.append((df2['Typical Price'].iloc[(x+1)])*(-1))
        x = x + 1
    df2['Abs TP'] = AbsTP
    df2['Raw Money'] = df2['Abs TP'] * df2['Volume']
    while y < days:
        if df2['Raw Money'].iloc[y] > 0:
            Positive.append(df2['Raw Money'].iloc[y])
            Negative.append(0)
        else:
            Negative.append(df2['Raw Money'].iloc[y])
            Positive.append(0)
        y = y + 1
    while z < 14:
        PosRatio = PosRatio + Positive[z]
        NegRatio = NegRatio + Negative[z]
        z = z + 1
    while z < days:
        MFR.append((PosRatio/(-1*NegRatio)))
        PosRatio = PosRatio - Positive[(z - 14)] + Positive[z]
        NegRatio = NegRatio - Negative[(z - 14)] + Negative[z]
        z = z + 1
    while w < len(MFR):
        Equat = 100 - (100/(1+MFR[w]))
        MFI.append(Equat)
        w = w + 1
    df2['MFI'] = MFI
    df2['Mid Line'] = df2['MFI'].mean()
    df2['SELL'] = df2['Mid Line'] + df2['MFI'].std()
    df2['BUYER'] = df2['Mid Line'] - df2['MFI'].std()
    fig = go.Figure()
    if "." in CompanyCode:
        king = ('Money Flow Index - '+ CompanyCode)
    else:
        king = ('Money Flow Index - '+ CompanyCode)
    fig.add_trace(go.Scatter(x=df2.index,y=df2['MFI'], mode = 'lines',marker=dict(size=1, color="blue"),showlegend=False))
    fig.update_xaxes(dtick="M2",tickformat="%d\n%b\n%Y")
    fig.add_trace(go.Scatter(x=df2.index,y=df2['BUYER'], mode = 'lines',marker=dict(size=1, color="green"),showlegend=False))
    fig.add_trace(go.Scatter(x=df2.index,y=df2['SELL'], mode = 'lines',marker=dict(size=1, color="red"),showlegend=False))
    fig.add_trace(go.Scatter(x=df2.index,y=df2['Mid Line'], mode = 'lines',marker=dict(size=1, color="black"),showlegend=False))
    fig.update_layout(title=king,xaxis_title="Time",yaxis_title="MFI Value", width=750, height = 550)
    return fig


def ReturnCalculator(selected_dropdown_value):
    outputlist = []
    CompanyCode = selected_dropdown_value
    outputlist.append(('The returns (div incl) - '+ CompanyCode))
    try:
        stocktoday = pdr.get_data_yahoo(CompanyCode,start=date.today(), end=date.today())
    except KeyError:
        stocktoday = pdr.get_data_yahoo(CompanyCode,start=(date.today() - datetime.timedelta(days=3)), end=(date.today() - datetime.timedelta(days=3)))
    try:
        stock30 = pdr.get_data_yahoo(CompanyCode,start=(date.today() - datetime.timedelta(days=30)), end=(date.today() - datetime.timedelta(days=30)))
        outputlist.append(('The 1 Month Return is: ',round((((stocktoday['Adj Close'][0]-stock30['Adj Close'][0])/stock30['Adj Close'][0])*100),2),'%'))
    except KeyError:
        print('fuck')
        stock30 = pdr.get_data_yahoo(CompanyCode,start=(date.today() - datetime.timedelta(days=30+3)), end=(date.today() - datetime.timedelta(days=30+3)))
        outputlist.append(('The 1 Month Return is: ',round((((stocktoday['Adj Close'][0]-stock30['Adj Close'][0])/stock30['Adj Close'][0])*100),2),'%'))
    try:
        stock90 = pdr.get_data_yahoo(CompanyCode,start=(date.today() - datetime.timedelta(days=90)), end=(date.today() - datetime.timedelta(days=90)))
        outputlist.append(('The 3 Month Return is: ',round((((stocktoday['Adj Close'][0]-stock90['Adj Close'][0])/stock90['Adj Close'][0])*100),2),'%'))
    except KeyError:
        stock90 = pdr.get_data_yahoo(CompanyCode,start=(date.today() - datetime.timedelta(days=90+2)), end=(date.today() - datetime.timedelta(days=90+2)))
        outputlist.append(('The 3 Month Return is: ',round((((stocktoday['Adj Close'][0]-stock90['Adj Close'][0])/stock90['Adj Close'][0])*100),2),'%'))
    try:
        stock180 = pdr.get_data_yahoo(CompanyCode,start=(date.today() - datetime.timedelta(days=180)), end=(date.today() - datetime.timedelta(days=180)))
        outputlist.append(('The 6 Month Return is: ',round((((stocktoday['Adj Close'][0]-stock180['Adj Close'][0])/stock180['Adj Close'][0])*100),2),'%'))
    except KeyError:
        stock180 = pdr.get_data_yahoo(CompanyCode,start=(date.today() - datetime.timedelta(days=180+2)), end=(date.today() - datetime.timedelta(days=180+2)))
        outputlist.append(('The 6 Month Return is: ',round((((stocktoday['Adj Close'][0]-stock180['Adj Close'][0])/stock180['Adj Close'][0])*100),2),'%'))
    try:
        stock365 = pdr.get_data_yahoo(CompanyCode,start=(date.today() - datetime.timedelta(days=365)), end=(date.today() - datetime.timedelta(days=365)))
        outputlist.append(('The 1 Year Return is: ',round((((stocktoday['Adj Close'][0]-stock365['Adj Close'][0])/stock365['Adj Close'][0])*100),2),'%'))
    except KeyError:
        try:
            stock365 = pdr.get_data_yahoo(CompanyCode,start=(date.today() - datetime.timedelta(days=365+2)), end=(date.today() - datetime.timedelta(days=365+2)))
            outputlist.append(('The 1 Year Return is: ',round((((stocktoday['Adj Close'][0]-stock365['Adj Close'][0])/stock365['Adj Close'][0])*100),2),'%'))
        except KeyError:
            outputlist.append('The company is younger than 1 year')
    try:
        stock1095 = pdr.get_data_yahoo(CompanyCode,start=(date.today() - datetime.timedelta(days=1095)), end=(date.today() - datetime.timedelta(days=1095)))
        outputlist.append(('The 3 Year Return is: ',round((((stocktoday['Adj Close'][0]-stock1095['Adj Close'][0])/stock1095['Adj Close'][0])*100),2),'%'))
    except KeyError:
        try:
            stock1095 = pdr.get_data_yahoo(CompanyCode,start=(date.today() - datetime.timedelta(days=1095+2)), end=(date.today() - datetime.timedelta(days=1095+2)))
            outputlist.append(('The 3 Year Return is: ',round((((stocktoday['Adj Close'][0]-stock1095['Adj Close'][0])/stock1095['Adj Close'][0])*100),2),'%'))
        except KeyError:
            outputlist.append('The company is younger than 3 years')
    try:
        stock1825 = pdr.get_data_yahoo(CompanyCode,start=(date.today() - datetime.timedelta(days=1825)), end=(date.today() - datetime.timedelta(days=1825)))
        outputlist.append(('The 5 Year Return is: ',round((((stocktoday['Adj Close'][0]-stock1825['Adj Close'][0])/stock1825['Adj Close'][0])*100),2),'%'))
    except KeyError:
        try:
            stock1825 = pdr.get_data_yahoo(CompanyCode,start=(date.today() - datetime.timedelta(days=1825+2)), end=(date.today() - datetime.timedelta(days=1825+2)))
            outputlist.append(('The 5 Year Return is: ',round((((stocktoday['Adj Close'][0]-stock1825['Adj Close'][0])/stock1825['Adj Close'][0])*100),2),'%'))
        except KeyError:
            outputlist.append('The company is younger than 5 years')
    return outputlist
    

def Option_Calculator(selected_dropdown_value,input1, input2, input3, input4, input5, input6, input7):
    strike = float(input2)
    risk = float(input4)
    Annual_Volatility = float(input7)
    stock = pdr.get_data_yahoo(selected_dropdown_value,start=datetime.datetime(2018,1,1), end=date.today())
    days = stock['Close'].count()
    Stock_Price = round(stock['Close'][len(stock['Close'])-1],3)
    df1 = pd.DataFrame(stock, columns=['Close','Open','Low','High'])
    delta_time = (input3/252)/input6
    u = math.exp(Annual_Volatility*(delta_time**(1/2)))
    d = 1/u
    p = (math.exp(risk*delta_time)-d)/(u-d)
    S_Far = []
    S_Near = []
    F_Far = []
    F_Near = []
    x = 0
    Option_Value = 0
    Early_Exercise_Value = 0
    while x <= input6:
        S_Far.append(Stock_Price*(u**(input6-x))*(d**x))
        if input1.upper() == 'C':
            F_Far.append(max((S_Far[x] - strike),0))
        if input1.upper() == 'P':
            F_Far.append(max((strike - S_Far[x]),0))
        x = x + 1
    x = 1
    counter = 0
    delta = 0
    while x <= input6:
        while counter <= (input6 - x):
            S_Near.append(Stock_Price*(u**(input6-x-counter))*(d**counter))
            Option_Value = math.exp(-risk*delta_time)*(p*F_Far[counter]+(1-p)*F_Far[counter+1])
            if input1.upper() == 'C' and input5.upper() == 'A':
                Early_Exercise_Value = S_Near[counter] - strike
            if input1.upper() == 'P' and input5.upper() == 'A':
                Early_Exercise_Value = strike - S_Near[counter]
            else:
                Early_Exercise_Value = 0
            F_Near.append(max(Option_Value,Early_Exercise_Value,0))
            counter = counter + 1
        delta = (F_Far[0]-F_Far[1])/((Stock_Price*u)-(Stock_Price*d))
        S_Near = []
        F_Far = F_Near
        F_Near = []
        counter = 0
        x = x + 1
    Premium = F_Far[0]
    return Stock_Price, Premium, delta


def VolatilityTable(selected_dropdown_value):
    outputlist = []
    CompanyCode = selected_dropdown_value
    stock = pdr.get_data_yahoo(CompanyCode,start=datetime.datetime(2018,1,1), end=date.today())
    days = stock['Close'].count()
    df1 = pd.DataFrame(stock, columns=['Close','Open','Low','High'])
    x = 1
    Daily_Return = [0]
    while x < days:
        Daily_Return.append((df1['Close'][x]-df1['Close'][x-1])/df1['Close'][x-1])
        x = x+1
    df1['Daily Return']=Daily_Return
    df1['252 Day Volatility'] = df1['Daily Return'].rolling(window=252).std()
    df1['180 Day Volatility'] = df1['Daily Return'].rolling(window=180).std()
    df1['60 Day Volatility'] = df1['Daily Return'].rolling(window=60).std()
    df1['30 Day Volatility'] = df1['Daily Return'].rolling(window=30).std()
    Annual_Volatility252 = (df1['252 Day Volatility'][len(df1['252 Day Volatility'])-1])*(252**(1/2))
    Annual_Volatility180 = (df1['180 Day Volatility'][len(df1['180 Day Volatility'])-1])*(252**(1/2))
    Annual_Volatility60 = (df1['60 Day Volatility'][len(df1['60 Day Volatility'])-1])*(252**(1/2))
    Annual_Volatility30 = (df1['30 Day Volatility'][len(df1['30 Day Volatility'])-1])*(252**(1/2))
    outputlist.append(("The code is: ",selected_dropdown_value))
    outputlist.append(("The annual volatility is (252 period): ",round(Annual_Volatility252,4)))
    outputlist.append(("The annual volatility is (180 period): ",round(Annual_Volatility180,4)))
    outputlist.append(("The annual volatility is (60 period): ",round(Annual_Volatility60,4)))
    outputlist.append(("The annual volatility is (30 period): ",round(Annual_Volatility30,4)))
    return outputlist

def VolatilityGrapher(selected_dropdown_value):
    CompanyCode = selected_dropdown_value
    stock = pdr.get_data_yahoo(CompanyCode,start=datetime.datetime(2018,1,1), end=date.today())
    days = stock['Close'].count()
    df1 = pd.DataFrame(stock, columns=['Close','Open','Low','High'])
    x = 1
    Daily_Return = [0]
    while x < days:
        Daily_Return.append((df1['Close'][x]-df1['Close'][x-1])/df1['Close'][x-1])
        x = x+1
    df1['Daily Return']=Daily_Return
    df1['252 Day Volatility'] = df1['Daily Return'].rolling(window=252).std()
    df1['180 Day Volatility'] = df1['Daily Return'].rolling(window=180).std()
    df1['60 Day Volatility'] = df1['Daily Return'].rolling(window=60).std()
    df1['30 Day Volatility'] = df1['Daily Return'].rolling(window=30).std()
    df1['Annual_Volatility252'] = (df1['252 Day Volatility'])*(252**(1/2))
    df1['Annual_Volatility180'] = (df1['180 Day Volatility'])*(252**(1/2))
    df1['Annual_Volatility60'] = (df1['60 Day Volatility'])*(252**(1/2))
    df1['Annual_Volatility30'] = (df1['30 Day Volatility'])*(252**(1/2))
    title_graph = "Historical Volatility Grapher - "+CompanyCode
    fig = go.Figure()
    fig.update_layout(title=title_graph, width=1000, height = 400)
    fig.update_yaxes(title = "Annual Volatility Equivalent")
    fig.add_trace(go.Scatter(x=df1.index,y=df1['Annual_Volatility252'], mode = 'lines',marker=dict(size=1, color="blue"),showlegend=True,name="252 Day"))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['Annual_Volatility180'], mode = 'lines',marker=dict(size=1, color="orange"),showlegend=True,name="180 Day"))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['Annual_Volatility60'], mode = 'lines',marker=dict(size=1, color="green"),showlegend=True,name="60 Day"))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['Annual_Volatility30'], mode = 'lines',marker=dict(size=1, color="black"),showlegend=True,name="30 Day"))
    fig.update_xaxes(dtick="M2",tickformat="%d\n%b\n%Y")
    return fig

#this creates the app -- imports the stylesheet
app = dash.Dash(__name__,meta_tags=[{'property':'og:image','content':'https://i.ibb.co/P5RkK55/James-Charge-1.png'}])
server = app.server

#This sets the apps basic colours
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

#This is the total layout of the app.
app.layout = html.Div([
    html.Div([
        html.A("Link to ETH/AUD Trading dashboard", href='https://eth-trading-bot.herokuapp.com/', target="_blank")
        ],style={'width': '65%','display': 'inline-block'}),
    #breaking it down this way means so far there will be 2 sections to the app
    html.Div([
        html.H4('Chart with Buy/Sell Signals'),
        dcc.Input(id='input', value='AAPL', type='text', debounce=True),
        dcc.Input(id='signalinput', value=0.09, type='text', debounce=True),
        html.Button('Submit', id='btn-nclicks-1', n_clicks=0),
        dcc.Graph(id='my-graph')
        
        ],style={'width': '65%','display': 'inline-block'}),
    #this is the 2nd section which will have a table of technical results
    html.Div([
        html.H4('Display of the Technical Results'),
        html.Table(id = 'my-table')
        
        ],style={'width': '30%', 'float': 'right','display': 'inline-block'}),
    
    html.Div([
        dcc.Graph(id='macd-graph')
        
        ],style={'width': '50%', 'float': 'left','display': 'inline-block'}),
    
    html.Div([
        dcc.Graph(id='mfi-graph')
        
        ],style={'width': '50%', 'float': 'right','display': 'inline-block'}),

    html.Div([
        html.H4('Returns'),
        html.Table(id = 'my-returns',style={'padding-bottom': '15%'}),
        html.H4('Option Pricing'),
        dcc.Input(id='input-1-state', type='text', placeholder='C or P'),
        dcc.Input(id='input-2-state', type='number', placeholder='Strike Price'),
        dcc.Input(id='input-3-state', type='number', placeholder='Days to Expiry'),
        dcc.Input(id='input-4-state', type='number', placeholder='Risk Free Rate'),
        dcc.Input(id='input-5-state', type='text', placeholder='A or E'),
        dcc.Input(id='input-6-state', type='number', placeholder='Binomial Steps'),
        dcc.Input(id='input-7-state', type='number', placeholder='Volatility'),
        html.Button(id='submit-button-state', n_clicks=0, children='Calculate'),
        html.Div(id='output-state',style={'padding-bottom': '15%'}),
        html.H4('Volatility'),
        html.Table(id = 'my-volatility')
        
        ],style={'width': '20%', 'float': 'left','display': 'inline-block','padding-left':'5%','padding-bottom':'2%'}),
    html.Div([
        html.H4('Fundamentals'),
        html.Table(id = 'my-fundamentals'),
        html.Table(id = 'my-profile')
        
        ],style={'width': '70%', 'float': 'right','display': 'inline-block','padding-right':'2%','padding-bottom':'2%'}),
    html.Div([
        html.H4('Further Ratios and Valuations'),
        dcc.Input(id='input-EBITDA-state', type='number', placeholder='Expected EBITDA Growth'),
        dcc.Input(id='input-Discount-state', type='number', placeholder='Discount Rate'),
        dcc.Input(id='input-FCF-state', type='number', placeholder='Expected Free Cashflow Growth'),
        html.Button(id='submit-button1-state', n_clicks=0, children='Calculate'),
        html.Table(id = 'my-valuations')
        
        ],style={'width': '70%', 'float': 'right','display': 'inline-block','padding-right':'2%','padding-bottom':'2%'}),

    html.Div([
        dcc.Graph(id='Bollinger-graph'),
        dcc.Graph(id='Volatility-graph')
        ],style={'width': '70%', 'float': 'right','display': 'inline-block','padding-right':'2%','padding-bottom':'2%'})

])

#This app callback updates the graph as per the relevant company
@app.callback(Output('my-graph','figure'),[Input('input','value'),Input('signalinput','value')])
def update_graph(selected_dropdown_value, signalinput):
    fig = TradingAlgo(selected_dropdown_value, 'bitch', signalinput)
    return fig


# for the output-list
@app.callback(Output('my-table', 'children'), [Input('input', 'value'),Input('signalinput','value')])
def generate_output_list(selected_dropdown_value, signalinput):
    outputlist = TradingAlgo(selected_dropdown_value, 'dog', signalinput)
    # Header
    return html.Table([html.Tr(html.Th('Output List'))] + [html.Tr(html.Td(output)) for output in outputlist],style={'border-spacing': '13px'})

@app.callback(Output('macd-graph','figure'),[Input('input','value')])
def update_macd(selected_dropdown_value):
    fig = MACD_BuySignal_graphed(selected_dropdown_value)
    return fig

@app.callback(Output('mfi-graph','figure'),[Input('input','value')])
def update_mfi(selected_dropdown_value):
    fig = MoneyFlowIndex(selected_dropdown_value)
    return fig

# for the output-list
@app.callback(Output('my-returns', 'children'), [Input('input', 'value')])
def generate_output_list(selected_dropdown_value):
    outputlist = ReturnCalculator(selected_dropdown_value)
    # Header
    return [html.Tr(html.Th('Returns of Product (Overall)'))] + [html.Tr(html.Td(output)) for output in outputlist]

# for the fundamentals table
@app.callback(Output('my-fundamentals', 'children'), [Input('input', 'value')])
def generate_fundamentaltable(selected_dropdown_value):
    table = Fundamentals(selected_dropdown_value)
    # Header
    return html.Table([html.Tr([html.Th(col) for col in table.columns])] + [html.Tr([html.Td(table.iloc[i][col]) for col in table.columns]) for i in range(0,len(table.EPS))],style={'border-spacing': '13px'})

@app.callback(Output('my-profile', 'children'), [Input('input', 'value')])
def generate_profile(selected_dropdown_value):
    outputlist = Profile(selected_dropdown_value)
    # Header
    return [html.Tr(html.Th('Company Profile'))] + [html.Tr(html.Td(output)) for output in outputlist]


@app.callback(Output('output-state', 'children'),
              Input('submit-button-state', 'n_clicks'),
              State('input', 'value'),
              State('input-1-state', 'value'),
              State('input-2-state', 'value'),
              State('input-3-state', 'value'),
              State('input-4-state', 'value'),
              State('input-5-state', 'value'),
              State('input-6-state', 'value'),
              State('input-7-state', 'value'))
def update_output(n_clicks, selected_dropdown_value, input1, input2, input3, input4, input5, input6, input7):
    if n_clicks == 0:
        return u'''
                Option calculation pending'''
    else:
        try:
            Stock_Price, Premium, Delta = Option_Calculator(selected_dropdown_value, input1, input2, input3, input4, input5, input6, input7)
            return "The company code is: ",selected_dropdown_value, ". The current price is: ",round(Stock_Price,2),". the premium is: ",round(Premium,4),". the delta is: ",round(Delta,4)
        except:
            return "The company code is: ",selected_dropdown_value," ...Enter valid details"

@app.callback(Output('my-valuations', 'children'),
              Input('submit-button1-state', 'n_clicks'),
              State('input', 'value'),
              State('input-EBITDA-state', 'value'),
              State('input-Discount-state', 'value'),
              State('input-FCF-state', 'value'))
def update_ratios(n_clicks, selected_dropdown_value, inputEBITDA, inputDiscount, inputFCF):
    if n_clicks == 0:
        return u'''
                Valuation Calculation Pending'''
    else:
        outputlist = Valuations(selected_dropdown_value,inputEBITDA,inputDiscount,inputFCF)
        return [html.Tr(html.Td(output)) for output in outputlist]



# for the output-list
@app.callback(Output('my-volatility', 'children'), [Input('input', 'value')])
def generate_volatility_list(selected_dropdown_value):
    outputlist = VolatilityTable(selected_dropdown_value)
    # Header
    return [html.Tr(html.Th('Volatility - Each Period'))] + [html.Tr(html.Td(output)) for output in outputlist]

@app.callback(Output('Volatility-graph','figure'),[Input('input','value')])
def update_stonker(selected_dropdown_value):
    fig = VolatilityGrapher(selected_dropdown_value)
    return fig

@app.callback(Output('Bollinger-graph','figure'),[Input('input','value')])
def update_stonker(selected_dropdown_value):
    fig = BollingerBands(selected_dropdown_value)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
