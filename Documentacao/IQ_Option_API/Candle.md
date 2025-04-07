Candle
===================================

get candles
---------------------------------------------

only get close clndle, not realtime

`Iq.get_candles(ACTIVES,interval,count,endtime)             #ACTIVES:sample input "EURUSD" OR "EURGBP".... youcan             #interval:duration of candles             #count:how many candles you want to get from now to past             #endtime:get candles from past to "endtime"`

### sample

`from iqoptionapi.stable_api import IQ_Option import time Iq=IQ_Option("email","password") Iq.connect()#connect to iqoption end_from_time=time.time() ANS=[] for i in range(70):     data=Iq.get_candles("EURUSD", 60, 1000, end_from_time)     ANS =data+ANS     end_from_time=int(data[0]["from"])-1 print(ANS)`

get realtime candles
---------------------------------------------------------------

### indicator sample

`from talib.abstract import * from iqoptionapi.stable_api import IQ_Option import time import numpy as np print("login...") Iq=IQ_Option("email","password") Iq.connect()#connect to iqoption goal="EURUSD" size=10#size=[1,5,10,15,30,60,120,300,600,900,1800,3600,7200,14400,28800,43200,86400,604800,2592000,"all"] timeperiod=10 maxdict=20 print("start stream...") Iq.start_candles_stream(goal,size,maxdict) print("Start EMA Sample") while True:     candles=Iq.get_realtime_candles(goal,size)      inputs = {         'open': np.array([]),         'high': np.array([]),         'low': np.array([]),         'close': np.array([]),         'volume': np.array([])     }     for timestamp in candles:          inputs["open"]=np.append(inputs["open"],candles[timestamp]["open"] )         inputs["high"]=np.append(inputs["open"],candles[timestamp]["max"] )         inputs["low"]=np.append(inputs["open"],candles[timestamp]["min"] )         inputs["close"]=np.append(inputs["open"],candles[timestamp]["close"] )         inputs["volume"]=np.append(inputs["open"],candles[timestamp]["volume"] )      print("Show EMA")     print(EMA(inputs, timeperiod=timeperiod))     print("\n")     time.sleep(1) Iq.stop_candles_stream(goal,size)`

### Sample

`from iqoptionapi.stable_api import IQ_Option import logging import time #logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s') print("login...") Iq=IQ_Option("email","password") Iq.connect()#connect to iqoption goal="EURUSD" size="all"#size=[1,5,10,15,30,60,120,300,600,900,1800,3600,7200,14400,28800,43200,86400,604800,2592000,"all"] maxdict=10 print("start stream...") Iq.start_candles_stream(goal,size,maxdict) #DO something print("Do something...") time.sleep(10)  print("print candles") cc=Iq.get_realtime_candles(goal,size) for k in cc:     print(goal,"size",k,cc[k]) print("stop candle") Iq.stop_candles_stream(goal,size)`

### size

![](../image/time_interval.png)

### start\_candles\_stream()

`goal="EURUSD" size="all"#size=[1,5,10,15,30,60,120,300,600,900,1800,3600,7200,14400,28800,43200,86400,604800,2592000,"all"] maxdict=10 print("start stream...") Iq.start_candles_stream(goal,size,maxdict)`

### get\_realtime\_candles()

get\_realtime\_candles() after call start\_candles\_stream()

`Iq.get_realtime_candles(goal,size)`

### stop\_candles\_stream()

if you not using get\_realtime\_candles() anymore please close the stream

`Iq.stop_candles_stream(goal,size)`