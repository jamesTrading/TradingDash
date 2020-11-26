import dash
import numpy as np
from dash.dependencies import Input, Output
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

def Fundamentals(selected_dropdown_value):
    CompanyCode = selected_dropdown_value
    try:
        stocktoday = pdr.get_data_yahoo(CompanyCode,start=date.today(), end=date.today())
    except KeyError:
        stocktoday = pdr.get_data_yahoo(CompanyCode,start=(datetime.datetime.now() - datetime.timedelta(days=2)), end=(datetime.datetime.now() - datetime.timedelta(days=2)))
    CurrentPrice = stocktoday['Close'][0]
    
    urlcashflow = 'https://www.finance.yahoo.com/quote/'+CompanyCode+'/cash-flow?p='+CompanyCode
    cashsoup = BeautifulSoup((requests.get(urlcashflow).text),"lxml")

    urlincome = 'https://www.finance.yahoo.com/quote/'+CompanyCode+'/financials?p='+CompanyCode
    incomesoup = BeautifulSoup((requests.get(urlincome).text),"lxml")

    urlbalancesheet = 'https://www.finance.yahoo.com/quote/'+CompanyCode+'/balance-sheet?p='+CompanyCode
    balancesoup = BeautifulSoup((requests.get(urlbalancesheet).text),"lxml")
    
    titleBalancesheet = balancesoup.findAll('div', {'class': "D(tbc) Ta(start) Pend(15px)--mv2 Pend(10px) Bxz(bb) Py(8px) Bdends(s) Bdbs(s) Bdstarts(s) Bdstartw(1px) Bdbw(1px) Bdendw(1px) Bdc($seperatorColor) Pos(st) Start(0) Bgc($lv2BgColor) fi-row:h_Bgc($hoverBgColor) Pstart(15px)--mv2 Pstart(10px)"})
    for title in titleBalancesheet:
        if 'Common Stock Equity' in title.text:
            shareholderequity = ([div.text for div in title.findNextSiblings(attrs={'data-test': 'fin-col'})])
            shareholderequity = [int(item.replace(',','')) for item in shareholderequity]
        if 'Total Liabilities Net Minority Interest' in title.text:
            Liability = ([div.text for div in title.findNextSiblings(attrs={'data-test': 'fin-col'})])
            Liability = [int(item.replace(',','')) for item in Liability]
        if 'Total Assets' in title.text:
            totalassets = ([div.text for div in title.findNextSiblings(attrs={'data-test': 'fin-col'})])
            totalassets = [int(item.replace(',','')) for item in totalassets]
        if 'Ordinary Shares Number' in title.text:
            Sharecount = ([div.text for div in title.findNextSiblings(attrs={'data-test': 'fin-col'})])
            Sharecount = [int(item.replace(',','')) for item in Sharecount]

    titleFinancial = incomesoup.findAll('div', {'class': "D(tbc) Ta(start) Pend(15px)--mv2 Pend(10px) Bxz(bb) Py(8px) Bdends(s) Bdbs(s) Bdstarts(s) Bdstartw(1px) Bdbw(1px) Bdendw(1px) Bdc($seperatorColor) Pos(st) Start(0) Bgc($lv2BgColor) fi-row:h_Bgc($hoverBgColor) Pstart(15px)--mv2 Pstart(10px)"})
    for title in titleFinancial:
        if 'Normalized EBITDA' in title.text:
            EBITDA = ([div.text for div in title.findNextSiblings(attrs={'data-test': 'fin-col'})])
            EBITDA.pop(0)
            EBITDA = [int(item.replace(',','')) for item in EBITDA]
        if 'Net Income Common Stockholders' in title.text:
            NetIncome = ([div.text for div in title.findNextSiblings(attrs={'data-test': 'fin-col'})])
            NetIncome.pop(0)
            NetIncome = [int(item.replace(',','')) for item in NetIncome]
            
    titleCashflow = cashsoup.findAll('div', {'class': "D(tbc) Ta(start) Pend(15px)--mv2 Pend(10px) Bxz(bb) Py(8px) Bdends(s) Bdbs(s) Bdstarts(s) Bdstartw(1px) Bdbw(1px) Bdendw(1px) Bdc($seperatorColor) Pos(st) Start(0) Bgc($lv2BgColor) fi-row:h_Bgc($hoverBgColor) Pstart(15px)--mv2 Pstart(10px)"})
    for title in titleCashflow:
        if 'Operating Cash Flow' in title.text:
            OperatingCashflow2 = ([div.text for div in title.findNextSiblings(attrs={'data-test': 'fin-col'})])
            OperatingCashflow2.pop(0)
            if OperatingCashflow2[0] == '-':
                OperatingCashflow2 = ['-' for item in OperatingCashflow2]
            else:
                OperatingCashflow2 = [int(item.replace(',','')) for item in OperatingCashflow2]
        if 'Cash Flows from Used in Operating Activities Direct' in title.text:
            OperatingCashflow1 = ([div.text for div in title.findNextSiblings(attrs={'data-test': 'fin-col'})])
            OperatingCashflow1.pop(0)
            OperatingCashflow1 = [int(item.replace(',','')) for item in OperatingCashflow1]
        if 'Free Cash Flow' in title.text:
            FreeCashflow = ([div.text for div in title.findNextSiblings(attrs={'data-test': 'fin-col'})])
            FreeCashflow.pop(0)
            FreeCashflow = [int(item.replace(',','')) for item in FreeCashflow]
    EPS = []
    y = 0
    OperatingCashflow = []
    try:
        if OperatingCashflow2[0] == '-':
            OperatingCashflow = OperatingCashflow1
        else:
            OperatingCashflow = OperatingCashflow2
    except:
        OperatingCashflow = OperatingCashflow1
    for element in NetIncome:
        EPS.append(round(element/Sharecount[y],3))
        y = y + 1
    x = 0
    EPSGrowth = []
    ROE = []
    ROA = []
    DTE = []
    while x < (len(EPS)-1):
        percent = ((EPS[x]-EPS[x+1])/EPS[x+1])*100
        EPSGrowth.append((round(percent,3) , '%'))
        x = x + 1
    EPSGrowth.append('-')
    z = 0
    EBITDAGrowth = []
    a = len(Liability) - 1
    while a >= 0:
        if a == len(NetIncome):
            ROE.append(round(NetIncome[a-1]/shareholderequity[a],3))
            ROA.append(round(NetIncome[a-1]/totalassets[a],3))
            DTE.append(round(Liability[a]/shareholderequity[a],3))
        else:
            ROE.append(round(NetIncome[a]/shareholderequity[a],3))
            ROA.append(round(NetIncome[a]/totalassets[a],3))
            DTE.append(round(Liability[a]/shareholderequity[a],3))
        a = a - 1
    try:
        len(EBITDA)
    except:
        EBITDA = []
        while z < len(EPS):
            EBITDA.append('N/A')
            z = z + 1
    big = len(Liability)
    dex = []
    while big > 0:
        dex.append(date.today().year-big+1)
        big = big - 1
    df = pd.DataFrame({CompanyCode: dex, 'EPS': list(reversed(EPS)), 'EPS Growth': list(reversed(EPSGrowth)), 'Net Income': list(reversed(NetIncome)),'ROE': ROE, 'ROA': ROA,'Debt to Equity':DTE, 'Shareholder Equity': list(reversed(shareholderequity)),
                       'Shares':list(reversed(Sharecount)),'Operating Cashflow': list(reversed(OperatingCashflow)), 'Free Cashflow': list(reversed(FreeCashflow)), 'EBITDA': list(reversed(EBITDA))},index=range(date.today().year-len(Liability)+1,date.today().year+1))
    return df

#This is the part of the code that takes buy points and displays them
def FibonacciGrapher(CompanyCode, dates, homie, selldate, scost,selldate1, scost1,BBUY, BBUYDate): 
    CompanyCode = CompanyCode
    x = 0
    y = 0
    w = 0
    differenceidentify = 0
    count = 125
    High = []
    Low = []
    Fib68 = []
    FibValue68 = []
    Fib50 = []
    FibValue50 = []
    Fib38 = []
    FibValue38 = []
    HighValue = []
    LowValue = []
    Extension = []
    ExtensionValue = []
    HighValue2 = []
    LowValue2 = []
    LowValue3 = []
    HighValue3 = []
    stock = pdr.get_data_yahoo(CompanyCode,start=datetime.datetime(2018,1,1), end=date.today())
    days = stock['Close'].count()
    close_prices = stock['Close']
    df1 = pd.DataFrame(stock, columns=['Close'])
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
        Fib68.append((High[x]-Low[x])*0.618 + Low[x])
        Fib50.append((High[x]-Low[x])*0.50 + Low[x])
        Fib38.append((High[x]-Low[x])*0.382 + Low[x])
        Extension.append(round((((float(High[x]) - float(Low[x]))*1.618)+float(Low[x])),2))
        x = x + 1
        if ((x+1)*100 == days):
            x = x+2
        differenceidentify = 0
    df1 = df1.dropna()
    while y < (days-count):
        HighValue.append(None)
        LowValue.append(None)
        ExtensionValue.append(None)
        FibValue68.append(None)
        FibValue50.append(None)
        FibValue38.append(None)
        y = y + 1
    while w < count:
        HighValue.append(round(float(High[0]), 2))
        LowValue.append(round(float(Low[0]), 2))
        FibValue68.append(round(float(Fib68[0]), 2))
        FibValue50.append(round(float(Fib50[0]), 2))
        FibValue38.append(round(float(Fib38[0]), 2))
        ExtensionValue.append(round(float(Extension[0]), 2))
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
    df1['Extension'] = ExtensionValue
    df1['Fib68'] = FibValue68
    df1['Fib50'] = FibValue50
    df1['Fib38'] = FibValue38
    df1['HFib2'] = HighValue2
    df1['LFib2'] = LowValue2
    df1['MMA'] = df1.rolling(window=50).mean()['Close']
    df1['SMA'] = df1.rolling(window=20).mean()['Close']
    df1['LMA'] = df1.rolling(window=100).mean()['Close']
    fig = go.Figure()
    king = 0
    if "." in CompanyCode:
        king = ('Australian Market - '+ CompanyCode)
    else:
        king = ('US Market - '+ CompanyCode)
    fig = px.line(df1, x=df1.index, y="Close", title=king, width=1000, height = 700)
    df2 = pd.DataFrame(data = {'Dates':dates,'BuyPrice':homie})
    fig.add_trace(go.Scatter(x=df2['Dates'],y=df2['BuyPrice'], mode = 'markers',marker=dict(size=12, color="lightgreen"),showlegend=False))
    df2 = pd.DataFrame(data = {'Dates':BBUYDate,'BuyPrice':BBUY})
    fig.add_trace(go.Scatter(x=df2['Dates'],y=df2['BuyPrice'], mode = 'markers',marker=dict(size=12, color="green"),showlegend=False))
    df3 = pd.DataFrame(data = {'Dates':selldate,'SellPrice':scost})
    fig.add_trace(go.Scatter(x=df3['Dates'],y=df3['SellPrice'], mode = 'markers',marker=dict(size=12, color="Red"),showlegend=False))
    df4 = pd.DataFrame(data = {'Dates':selldate1,'SellPrice1':scost1})
    fig.add_trace(go.Scatter(x=df4['Dates'],y=df4['SellPrice1'], mode = 'markers',marker=dict(size=12, color="Yellow"),showlegend=False))
    fig.update_xaxes(dtick="M2",tickformat="%d\n%b\n%Y")
    fig.add_trace(go.Scatter(x=df1.index,y=df1['LMA'], mode = 'lines',marker=dict(size=1, color="orange"),showlegend=False))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['HFib'], mode = 'lines',marker=dict(size=1, color="purple"),showlegend=False))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['HFib2'], mode = 'lines',marker=dict(size=1, color="purple"),showlegend=False))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['High3'], mode = 'lines',marker=dict(size=1, color="purple"),showlegend=False))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['LFib'], mode = 'lines',marker=dict(size=1, color="black"),showlegend=False))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['LFib2'], mode = 'lines',marker=dict(size=1, color="black"),showlegend=False))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['Low3'], mode = 'lines',marker=dict(size=1, color="black"),showlegend=False))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['Fib68'], mode = 'lines',marker=dict(size=1, color="red"),showlegend=False))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['Fib50'], mode = 'lines',marker=dict(size=1, color="green"),showlegend=False))
    fig.add_trace(go.Scatter(x=df1.index,y=df1['Fib38'], mode = 'lines',marker=dict(size=1, color="orange"),showlegend=False))
    return fig


#This is the part of the app for producing the main info and buy/sell points
def TradingAlgo(selected_dropdown_value, junky, signalinput):
    costbases = 0
    signalinput = float(signalinput)
    factor = junky
    sharecount = 0
    counter = 0
    pscost = 0
    currentprices = 0
    currentcostbases = 0
    TYPValue = 0
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
    timer = 0
    timetrack = []
    MFI = [50,50,50,50,50,50,50,50,50,50,50,50,50,50]
    SellRange = []
    seller = 0
    BuyRange = []
    buyer = 0
    CompanyCode = selected_dropdown_value
    outputlist = []
    if "." in CompanyCode:
        market = pdr.get_data_yahoo('^AXJO',start=datetime.datetime(2018,1,1), end=date.today())
        outputlist.append(('Australian Market - '+ CompanyCode))
    else:
        market = pdr.get_data_yahoo('^GSPC',start=datetime.datetime(2018,1,1), end=date.today())
        outputlist.append(('US Market - '+ CompanyCode))
    stock = pdr.get_data_yahoo(CompanyCode,start=datetime.datetime(2018,1,1), end=date.today())
    days = stock['Close'].count()
    cuck = market['Close'].count()  
    while timer < days:
        timetrack.append((timer+1))
        timer = timer + 1
    while buyer < days:
        BuyRange.append(25)
        buyer = buyer + 1
    while seller < days:
        SellRange.append(80)
        seller = seller + 1
    df2 = pd.DataFrame(stock)
    df1 = pd.DataFrame(market)
    df2.index = pd.to_datetime(df2.index)
    df2['LMA'] = df2.rolling(window=100).mean()['Close']
    df2['SMA'] = df2.rolling(window=30).mean()['Close']
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
    df2['SELL'] = SellRange
    df2['BUYER'] = BuyRange
    df2['Timer'] = timetrack
    df2['26 EMA'] = df2.ewm(span = 26, min_periods = 26).mean()['Close']
    df2['12 EMA'] = df2.ewm(span = 12, min_periods = 12).mean()['Close']
    df2['MACD'] = df2['12 EMA'] - df2['26 EMA']
    df2['Signal Line'] = df2.ewm(span = 9, min_periods = 9).mean()['MACD']
    xtra = 0
    trade_return = []
    maxValnear = []
    bigposition = []
    market_return = []
    largebuy = 0
    largebuycounter = 0
    small = 0
    smallcounter = 0
    dates = []
    homie = []
    bbb = 0
    lastmax = []
    lastmin = []
    bbbcounter = 0
    mfucker = []
    signal = 0
    tendies = []
    MFItracker = []
    internalcounter = 3
    BBUY = []
    BBUYDate = []
    while counter < (days):
        TYPValue = round(float(df2['Close'][counter]),2) + TYPValue
        if round(float(df2['MFI'][counter]),2) < 30:
            if df2['MACD'][counter] < 0:
                if (df2['MACD'][counter]) < (df2['Signal Line'][counter]):
                    if abs((df2['MACD'][counter] - df2['Signal Line'][counter])) < abs((df2['MACD'][counter-1] - df2['Signal Line'][counter-1])):
                        if abs(df2['MACD'][counter]) > abs(df2['MACD'][counter-1]):
                            Valnear = df2['Close'][(counter):(counter+90)]
                            maxValnear.append(max(Valnear))
                            maxposition = np.where(Valnear == Valnear.max())
                            maxposition = maxposition[0][0]
                            bigposition.append(int(maxposition))
                            if counter < 90:
                                lastmax.append(max(df2['Close'][(0):(counter)]))
                                lastmin.append(min(df2['Close'][(0):(counter)]))
                            else:
                                lastmax.append(max(df2['Close'][(counter - 90):(counter-5)]))
                                lastmin.append(min(df2['Close'][(counter - 90):(counter - 5)]))
                            if days-counter > 90:
                                mfucker.append((((lastmax[len(lastmax)-1]-lastmin[len(lastmax)-1])*0.618+round(float(df2['Close'][counter]),2))-maxValnear[sharecount])/maxValnear[sharecount])                                
                                signal = round((((lastmax[len(lastmax)-1]-lastmin[len(lastmax)-1])*0.618+round(float(df2['Close'][counter]),2))-round(float(df2['Close'][(counter)]),2))/round(float(df2['Close'][(counter)]),2),5)
                            if days - counter <= 90:
                                Valnear = df2['Close'][(counter):(days-1)]
                                if days - counter < 2:
                                    maxValnear.append(df2['Close'][counter])
                                    mfucker.append((((lastmax[len(lastmax)-1]-lastmin[len(lastmax)-1])*0.618+round(float(df2['Close'][counter]),2))-maxValnear[sharecount])/maxValnear[sharecount])
                                    signal = round((((lastmax[len(lastmax)-1]-lastmin[len(lastmax)-1])*0.618+round(float(df2['Close'][counter]),2))-round(float(df2['Close'][(counter)]),2))/round(float(df2['Close'][(counter)]),2),5)
                                else:
                                    maxValnear.append(max(Valnear))
                                    mfucker.append((((lastmax[len(lastmax)-1]-lastmin[len(lastmax)-1])*0.618+round(float(df2['Close'][counter]),2))-maxValnear[sharecount])/maxValnear[sharecount])
                                    signal = round((((lastmax[len(lastmax)-1]-lastmin[len(lastmax)-1])*0.618+round(float(df2['Close'][counter]),2))-round(float(df2['Close'][(counter)]),2))/round(float(df2['Close'][(counter)]),2),5)
                            if signal > signalinput:
                                tendies.append(round((maxValnear[sharecount-1] - round(float(df2['Close'][(counter)]),3))/round(float(df2['Close'][(counter)]),3),3))
                                dates.append(df2.index.date[counter])
                                homie.append(round(float(df2['Close'][counter]),3))
                                MFItracker.append(df2['MFI'][counter])
                                if (counter + 1) == days:
                                    costbases = costbases + round(float(df2['Close'][(counter)]),3)
                                else:
                                    costbases = costbases + round(float(df2['Close'][(counter+1)]),3)
                                bbb = bbb + round(float(df2['SMA'][counter]),2)
                                bbbcounter = bbbcounter + 1
                                sharecount = sharecount + 1
                                trade_return.append(((maxValnear[sharecount-1] - round(float(df2['Close'][(counter)]),3))/round(float(df2['Close'][(counter)]),3))*100)
                                market_return.append(((df1['Close'][counter+int(maxposition)] - round(float(df1['Close'][(counter)]),3))/round(float(df1['Close'][(counter)]),3))*100)
                                if df2['Close'][counter] < df2['LMA'][counter]*0.95:
                                    largebuy = (maxValnear[sharecount-1] - round(float(df2['Close'][(counter)]),3))/round(float(df2['Close'][(counter)]),3)*100 + largebuy
                                    largebuycounter = largebuycounter + 1
                                else:
                                    small = (maxValnear[sharecount-1] - round(float(df2['Close'][(counter)]),3))/round(float(df2['Close'][(counter)]),3)*100 + small
                                    smallcounter = smallcounter + 1
                                signal = 0
                                lengthy = 15
                                if (days - counter) < 15:
                                    lengthy = counter - days - 3
                                while internalcounter < lengthy:
                                    if df2['MFI'][(counter+internalcounter)] > MFItracker[len(MFItracker)-1]:
                                        if (df2['Close'][counter+internalcounter]-df2['Close'][counter])/df2['Close'][counter] < -0.04:
                                            tendies.append(round((maxValnear[sharecount-1] - round(float(df2['Close'][(counter+internalcounter)]),3))/round(float(df2['Close'][(counter+internalcounter)]),3),3))
                                            dates.append(df2.index.date[counter+internalcounter])
                                            homie.append(round(float(df2['Close'][counter+internalcounter]),3))
                                            MFItracker.append(df2['MFI'][counter+internalcounter])
                                            BBUY.append(round(float(df2['Close'][counter+internalcounter]),3))
                                            BBUYDate.append(df2.index.date[counter+internalcounter])
                                            costbases = costbases + round(float(df2['Close'][(counter+internalcounter)]),3)
                                            sharecount = sharecount + 1
                                            Valnear = df2['Close'][(counter+internalcounter):(counter+90+internalcounter)]
                                            if days - counter - internalcounter <= 90:
                                                Valnear = df2['Close'][(counter+internalcounter):(days-1)]
                                            maxValnear.append(max(Valnear))
                                            trade_return.append(((maxValnear[sharecount-1] - round(float(df2['Close'][(counter)]),3))/round(float(df2['Close'][(counter)]),3))*100)
                                            if df2['Close'][(counter+internalcounter)] < df2['LMA'][(counter+internalcounter)]*0.95:
                                                largebuy = (maxValnear[sharecount-1] - round(float(df2['Close'][(counter+internalcounter)]),3))/round(float(df2['Close'][(counter+internalcounter)]),3)*100 + largebuy
                                                largebuycounter = largebuycounter + 1
                                            else:
                                                small = (maxValnear[sharecount-1] - round(float(df2['Close'][(counter+internalcounter)]),3))/round(float(df2['Close'][(counter+internalcounter)]),3)*100 + small
                                                smallcounter = smallcounter + 1
                                    internalcounter = internalcounter + 1
                                internalcounter = 0
        counter = counter + 1
    TYPValue = TYPValue / (days)
    outputlist.append(("The last buy date is: ", dates[len(dates)-1]))
    outputlist.append(("The last price bought for: $", homie[len(homie)-1]))
    outputlist.append(("The expected sell point is: $", round((lastmax[len(lastmax)-1]-lastmin[len(lastmax)-1])*0.618+homie[len(homie)-1],2)))
    outputlist.append(("The expected return on the buy: ", round(((((lastmax[len(lastmax)-1]-lastmin[len(lastmax)-1])*0.618+homie[len(homie)-1]-homie[len(homie)-1])/homie[len(homie)-1])*100),3),"%"))
    outputlist.append(("The average closeness to the predicted sell point was: ", round((sum(mfucker)/len(mfucker))*100,3),"%"))
    outputlist.append(("The average return on the each buy is: ", round((sum(tendies)/len(tendies))*100,3),"%"))
    sellcounter = 0
    selldate = []
    sprice = 0
    scost = []
    discount = 0
    scount = 0
    while sellcounter < (days - 1):
        if round(float(df2['MFI'][sellcounter]),2) > 73:
            if df2['Signal Line'][sellcounter] > 0:
                if (df2['MACD'][sellcounter-1]) > (df2['Signal Line'][sellcounter-1]):
                    if abs((df2['MACD'][sellcounter-1] - df2['Signal Line'][sellcounter-1])) < abs((df2['MACD'][sellcounter-2] - df2['Signal Line'][sellcounter-2])):
                        selldate.append(df2.index.date[sellcounter])
                        scost.append(round(float(df2['Close'][sellcounter]),3))
                        sprice = sprice + round(float(df2['Close'][(sellcounter + 1)]),3)
                        scount = scount + 1
        sellcounter = sellcounter + 1
    sellcounter = 0
    selldate1 = []
    sprice1 = 0
    scost1 = []
    discount1 = 0
    scount1 = 0
    while sellcounter < (days - 1):
        if round(float(df2['MFI'][sellcounter]),2) > 73 or round(float(df2['MFI'][sellcounter-1]),2) > 73:
            if df2['Signal Line'][sellcounter] > 0:
                if (df2['MACD'][sellcounter-1]) > (df2['Signal Line'][sellcounter-1]):
                    if df2['MACD'][sellcounter] < df2['Signal Line'][sellcounter]:
                        selldate1.append(df2.index.date[sellcounter])
                        scost1.append(round(float(df2['Close'][sellcounter]),3))
                        sprice1 = sprice1 + round(float(df2['Close'][(sellcounter + 1)]),3)
                        scount1 = scount1 + 1
        sellcounter = sellcounter + 1
    if sharecount == 0:
        currentprices = round(float(df2['Close'][(days-1)]),2)
        outputlist.append(("No Shares."))
        outputlist.append(("The typical share price during the period was: $", round(TYPValue,2)))
        outputlist.append(("################################"))
        outputlist.append(("The MFI 3 days ago was: ", round(float(df2['MFI'][(days - 3)]), 2)))
        outputlist.append(("The MFI 2 days ago was: ", round(float(df2['MFI'][(days - 2)]), 2)))
        outputlist.append(("The MFI 1 day ago was: ", round(float(df2['MFI'][(days - 1)]), 2)))
        outputlist.append(("The current price is: $",currentprices))
        outputlist.append(("If bought today the sell point would be: $", round((max(df2['Close'][(days - 90):(days-5)])-min(df2['Close'][(days - 90):(days-5)]))*0.618+currentprices),2))
        outputlist.append(("The realised return would be: ", round((((max(df2['Close'][(days - 90):(days-5)])-min(df2['Close'][(days - 90):(days-5)]))*0.618+currentprices-currentprices)/currentprices)*100,3),"%"))
        outputlist.append(("################################"))
        outputlist.append(("The MACD 3 days ago was: ",df2['MACD'][days-3]))
        outputlist.append(("The MACD 2 days ago was: ",df2['MACD'][days-2]))
        outputlist.append(("The MACD 1 day ago was: ",df2['MACD'][days-1]))
        outputlist.append(("The difference 3 days ago was: ", (df2['MACD'][days-3] - df2['Signal Line'][days-3])))
        outputlist.append(("The difference 2 days ago was: ", (df2['MACD'][days-2] - df2['Signal Line'][days-2])))
        outputlist.append(("The difference 1 day ago was: ", (df2['MACD'][days-1] - df2['Signal Line'][days-1])))
        outputlist.append(("The current 100 MA value is: $", df2['LMA'][days-1]))
        outputlist.append(("The current value over the 100 MA is: ", (df2['Close'][days-1]-df2['LMA'][days-1])/df2['LMA'][days-1]))
        if round(float(df2['MFI'][days-1]),2) < 30:
            if df2['MACD'][days-1] < 0:
                if (df2['MACD'][days-1]) < (df2['Signal Line'][days-1]):
                    if abs((df2['MACD'][days-1] - df2['Signal Line'][days-1])) < abs((df2['MACD'][days-2] - df2['Signal Line'][days-2])):
                        if abs(df2['MACD'][days-1]) > abs(df2['MACD'][days-2]):
                            outputlist.append("--- BIG BUY ---")
        if factor == 'bitch':  
            bloop = FibonacciGrapher(CompanyCode, dates, homie, selldate, scost, selldate1, scost1,BBUY, BBUYDate)
            return bloop
        else:
            return outputlist
    
    currentprices = round(float(df2['Close'][(days-1)]),2)
    pscost = costbases / sharecount
    if largebuycounter > 0:
        outputlist.append(("The return when bought with +5% discount was: ", round(largebuy / largebuycounter,2),"  (",largebuycounter,")"))
    if smallcounter > 0:
        outputlist.append(("The close return for every other purchase: ", round(small / smallcounter,2),"  (",smallcounter,")"))
    outputlist.append(("Total number of shares bought: ", sharecount))
    outputlist.append(("Total cost base of shares: $", round(costbases,2)))
    outputlist.append(("Per share cost base is: $", round(pscost,2)))
    outputlist.append(("The typical price for shares bought in a similar time are: $", round(bbb/bbbcounter, 2)))
    outputlist.append(("################################"))
    outputlist.append(("The MFI 3 days ago was: ", round(float(df2['MFI'][(days - 3)]), 2)))
    outputlist.append(("The MFI 2 days ago was: ", round(float(df2['MFI'][(days - 2)]), 2)))
    outputlist.append(("The MFI 1 day ago was: ", round(float(df2['MFI'][(days - 1)]), 2)))
    outputlist.append(("The current price is: $",currentprices))
    outputlist.append(("If bought today the sell point would be: $", round((max(df2['Close'][(days - 90):(days-5)])-min(df2['Close'][(days - 90):(days-5)]))*0.618+currentprices,2)))
    outputlist.append(("The realised return would be: ", round((((max(df2['Close'][(days - 90):(days-5)])-min(df2['Close'][(days - 90):(days-5)]))*0.618+currentprices-currentprices)/currentprices)*100,3),"%"))
    outputlist.append(("################################"))
    outputlist.append(("The MACD 3 days ago was: ",round(df2['MACD'][days-3],3)))
    outputlist.append(("The MACD 2 days ago was: ",round(df2['MACD'][days-2],3)))
    outputlist.append(("The MACD 1 day ago was: ",round(df2['MACD'][days-1],3)))
    outputlist.append(("The difference 3 days ago was: ", round((df2['MACD'][days-3] - df2['Signal Line'][days-3]),3)))
    outputlist.append(("The difference 2 days ago was: ", round((df2['MACD'][days-2] - df2['Signal Line'][days-2]),3)))
    outputlist.append(("The difference 1 day ago was: ", round((df2['MACD'][days-1] - df2['Signal Line'][days-1]),3)))
    outputlist.append(("The current 100 MA value is: $", round(df2['LMA'][days-1],3)))
    outputlist.append(("The current value over the 100 MA is: ", round(((df2['Close'][days-1]-df2['LMA'][days-1])/df2['LMA'][days-1])*100,3),'%'))
    if round(float(df2['MFI'][days-1]),2) < 30:
            if df2['MACD'][days-1] < 0:
                if (df2['MACD'][days-1]) < (df2['Signal Line'][days-1]):
                    if abs((df2['MACD'][days-1] - df2['Signal Line'][days-1])) < abs((df2['MACD'][days-2] - df2['Signal Line'][days-2])):
                        if abs(df2['MACD'][days-1]) > abs(df2['MACD'][days-2]):
                            outputlist.append("--- BIG BUY ---")
    if factor == 'bitch': 
        bloop = FibonacciGrapher(CompanyCode, dates, homie, selldate, scost, selldate1, scost1,BBUY, BBUYDate)
        return bloop
    else:
        return outputlist
    
def MACD_BuySignal_graphed(selected_dropdown_value):
    CompanyCode = selected_dropdown_value
    stock = pdr.get_data_yahoo(CompanyCode,start=datetime.datetime(2019,9,1), end=date.today())
    days = stock['Close'].count()
    timer = 0
    timetrack = []
    while timer < (days - 33):
        timetrack.append(0)
        timer = timer + 1
    df2 = pd.DataFrame(stock, columns = ['Close'])
    df2['26 EMA'] = df2.ewm(span = 26, min_periods = 26).mean()['Close']
    df2['12 EMA'] = df2.ewm(span = 12, min_periods = 12).mean()['Close']
    df2['MACD'] = df2['12 EMA'] - df2['26 EMA']
    df2['Signal Line'] = df2.ewm(span = 9, min_periods = 9).mean()['MACD']
    df2 = df2.dropna()
    df2['Zero Line'] = timetrack
    fig = go.Figure()
    if "." in CompanyCode:
        king = ('MACD Graph - '+ CompanyCode)
    else:
        king = ('MACD Graph - '+ CompanyCode)
    fig.add_trace(go.Scatter(x=df2.index,y=df2['MACD'], mode = 'lines',marker=dict(size=1, color="red"),showlegend=False))
    fig.update_xaxes(dtick="M2",tickformat="%d\n%b\n%Y")
    fig.add_trace(go.Scatter(x=df2.index,y=df2['Signal Line'], mode = 'lines',marker=dict(size=1, color="green"),showlegend=False))
    fig.add_trace(go.Scatter(x=df2.index,y=df2['Zero Line'], mode = 'lines',marker=dict(size=1, color="blue"),showlegend=False))
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
    SellRange = []
    seller = 0
    BuyRange = []
    buyer = 0
    CompanyCode = selected_dropdown_value
    stock = pdr.get_data_yahoo(CompanyCode,start=datetime.datetime(2019,9,1), end=date.today())
    days = stock['Close'].count()
    while buyer < days:
        BuyRange.append(30)
        buyer = buyer + 1
    while seller < days:
        SellRange.append(75)
        seller = seller + 1
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
    df2['SELL'] = SellRange
    df2['BUYER'] = BuyRange
    fig = go.Figure()
    if "." in CompanyCode:
        king = ('Money Flow Index - '+ CompanyCode)
    else:
        king = ('Money Flow Index - '+ CompanyCode)
    fig.add_trace(go.Scatter(x=df2.index,y=df2['MFI'], mode = 'lines',marker=dict(size=1, color="blue"),showlegend=False))
    fig.update_xaxes(dtick="M2",tickformat="%d\n%b\n%Y")
    fig.add_trace(go.Scatter(x=df2.index,y=df2['BUYER'], mode = 'lines',marker=dict(size=1, color="green"),showlegend=False))
    fig.add_trace(go.Scatter(x=df2.index,y=df2['SELL'], mode = 'lines',marker=dict(size=1, color="red"),showlegend=False))
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
    

#this creates the app -- imports the stylesheet
app = dash.Dash(__name__)
server = app.server

#This sets the apps basic colours
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

#This is the total layout of the app.
app.layout = html.Div([
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
        html.Table(id = 'my-returns')
        
        ],style={'width': '20%', 'float': 'left','display': 'inline-block','border':'solid', 'padding-left':'5%','padding-bottom':'2%'}),
    html.Div([
        html.H4('Fundamentals'),
        html.Table(id = 'my-fundamentals')
        
        ],style={'width': '70%', 'float': 'right','display': 'inline-block','border':'solid', 'padding-right':'2%','padding-bottom':'2%'})

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
    return [html.Tr(html.Th('Output List'))] + [html.Tr(html.Td(output)) for output in outputlist]

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
    print(table)
    # Header
    return html.Table([html.Tr([html.Th(col) for col in table.columns])] + [html.Tr([html.Td(table.iloc[i][col]) for col in table.columns]) for i in range(0,len(table.EPS))],style={'border-spacing': '13px'})

if __name__ == '__main__':
    app.run_server(debug=True)
