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

#This is the part of the code that takes buy points and displays them
def FibonacciGrapher(CompanyCode, dates, homie, selldate, scost,selldate1, scost1): 
    CompanyCode = CompanyCode
    x = 0
    y = 0
    buyloop = 0
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
    while buyloop < len(homie):
        df2 = pd.DataFrame(data = {'Dates':dates,'BuyPrice':homie})
        fig.add_trace(go.Scatter(x=df2['Dates'],y=df2['BuyPrice'], mode = 'markers',marker=dict(size=12, color="Green"),showlegend=False))
        buyloop = buyloop + 1
    sellloop = 0
    while sellloop <  len(scost):
        df3 = pd.DataFrame(data = {'Dates':selldate,'SellPrice':scost})
        fig.add_trace(go.Scatter(x=df3['Dates'],y=df3['SellPrice'], mode = 'markers',marker=dict(size=12, color="Red"),showlegend=False))
        sellloop = sellloop + 1
    sellloop = 0
    while sellloop <  len(scost1):
        df4 = pd.DataFrame(data = {'Dates':selldate1,'SellPrice1':scost1})
        fig.add_trace(go.Scatter(x=df4['Dates'],y=df4['SellPrice1'], mode = 'markers',marker=dict(size=12, color="Yellow"),showlegend=False))
        sellloop = sellloop + 1
    fig.update_xaxes(dtick="M1",tickformat="%b\n%Y")
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
    while counter < (days):
        TYPValue = round(float(df2['Close'][counter]),2) + TYPValue
        if round(float(df2['MFI'][counter]),2) < 30:
            if df2['MACD'][counter] < 0:
                if (df2['MACD'][counter-1]) < (df2['Signal Line'][counter-1]):
                    if abs((df2['MACD'][counter-1] - df2['Signal Line'][counter-1])) < abs((df2['MACD'][counter-2] - df2['Signal Line'][counter-2])):
                        if abs(df2['MACD'][counter-1]) > abs(df2['MACD'][counter-2]):
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
                                tendies.append(round((maxValnear[sharecount-1] - round(float(df2['Close'][(counter)]),2))/round(float(df2['Close'][(counter)]),2),3))
                                dates.append(df2.index.date[counter])
                                homie.append(round(float(df2['Close'][counter]),2))
                                if (counter + 1) == days:
                                    costbases = costbases + round(float(df2['Close'][(counter)]),2)
                                else:
                                    costbases = costbases + round(float(df2['Close'][(counter+1)]),2)
                                bbb = bbb + round(float(df2['SMA'][counter]),2)
                                bbbcounter = bbbcounter + 1
                                sharecount = sharecount + 1
                                trade_return.append(((maxValnear[sharecount-1] - round(float(df2['Close'][(counter)]),2))/round(float(df2['Close'][(counter)]),2))*100)
                                market_return.append(((df1['Close'][counter+int(maxposition)] - round(float(df1['Close'][(counter)]),2))/round(float(df1['Close'][(counter)]),2))*100)
                                if df2['Close'][counter] < df2['LMA'][counter]*0.95:
                                    largebuy = (maxValnear[sharecount-1] - round(float(df2['Close'][(counter)]),2))/round(float(df2['Close'][(counter)]),2)*100 + largebuy
                                    largebuycounter = largebuycounter + 1
                                else:
                                    small = (maxValnear[sharecount-1] - round(float(df2['Close'][(counter)]),2))/round(float(df2['Close'][(counter)]),2)*100 + small
                                    smallcounter = smallcounter + 1
                            signal = 0
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
                        scost.append(round(float(df2['Close'][sellcounter]),2))
                        sprice = sprice + round(float(df2['Close'][(sellcounter + 1)]),2)
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
                        scost1.append(round(float(df2['Close'][sellcounter]),2))
                        sprice1 = sprice1 + round(float(df2['Close'][(sellcounter + 1)]),2)
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
            bloop = FibonacciGrapher(CompanyCode, dates, homie, selldate, scost, selldate1, scost1)
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
        bloop = FibonacciGrapher(CompanyCode, dates, homie, selldate, scost, selldate1, scost1)
        return bloop
    else:
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
        
        ],style={'width': '30%', 'float': 'right','display': 'inline-block'})
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



if __name__ == '__main__':
    app.run_server(debug=True)
