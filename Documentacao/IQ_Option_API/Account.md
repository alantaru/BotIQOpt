
Account
=====================================

get\_balance()
------------------------------------------------

`Iq.get_balance()`

get\_balance\_v2()
-------------------------------------------------------

more accuracy

`Iq.get_balance_v2()`

get\_currency()
--------------------------------------------------

you will check what currency you use

`Iq.get_currency()`

reset\_practice\_balance()
-----------------------------------------------------------------------

reset practice balance to $10000

`from iqoptionapi.stable_api import IQ_Option Iq=IQ_Option("email","password") Iq.connect()#connect to iqoption print(Iq.reset_practice_balance())`

Change real/practice Account
------------------------------------------------------------------------------

MODE="PRACTICE"/"REAL"/"TOURNAMENT"  
PRACTICE - it is demo account  
REAL - It is our money in risk  
TOURNAMENT - Tournaments account  

`balance_type="PRACTICE" Iq.change_balance(balance_type)`

get Other People stratagy
-------------------------------------------------------------------------

### sample

`from iqoptionapi.stable_api import IQ_Option import logging import time  #logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s') Iq=IQ_Option("email","password") Iq.connect()#connect to iqoption while_run_time=10  #For digital option  name="live-deal-digital-option" #"live-deal-binary-option-placed"/"live-deal-digital-option" active="EURUSD" _type="PT1M"#"PT1M"/"PT5M"/"PT15M" buffersize=10# print("_____________subscribe_live_deal_______________") Iq.subscribe_live_deal(name,active,_type,buffersize)  start_t=time.time() while True:     #data size is below buffersize     #data[0] is the last data     data=(Iq.get_live_deal(name,active,_type))     print("__For_digital_option__ data size:"+str(len(data)))     print(data)     print("\n\n")     time.sleep(1)     if time.time()-start_t>while_run_time:         break print("_____________unscribe_live_deal_______________") Iq.unscribe_live_deal(name,active,_type)  #For binary option  name="live-deal-binary-option-placed" active="EURUSD" _type="turbo"#"turbo"/"binary" buffersize=10# print("_____________subscribe_live_deal_______________") Iq.subscribe_live_deal(name,active,_type,buffersize)  start_t=time.time() while True:     #data size is below buffersize     #data[0] is the last data     data=(Iq.get_live_deal(name,active,_type))     print("__For_binary_option__ data size:"+str(len(data)))     print(data)     print("\n\n")     time.sleep(1)     if time.time()-start_t>while_run_time:         break print("_____________unscribe_live_deal_______________") Iq.unscribe_live_deal(name,active,_type)`

### subscribe\_live\_deal

`Iq.subscribe_live_deal(name,active,_type,buffersize)`

### unscribe\_live\_deal

`Iq.unscribe_live_deal(name,active,_type)`

### get\_live\_deal

`Iq.get_live_deal(name,active,_type)`

### pop\_live\_deal

pop the data from list

`Iq.pop_live_deal(name,active,_type)`

get Other people detail
---------------------------------------------------------------------

### sample

`from iqoptionapi.stable_api import IQ_Option import logging import time  #logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s') Iq=IQ_Option("email","password") Iq.connect()#connect to iqoption while_run_time=10  #For binary option name="live-deal-binary-option-placed" active="EURUSD" _type="turbo"#"turbo"/"binary" buffersize=10# print("_____________subscribe_live_deal_______________") print("\n\n") Iq.subscribe_live_deal(name,active,_type,buffersize)  last_trade_data=Iq.get_live_deal(name,active,_type)[0]  user_id=last_trade_data["user_id"] counutry_id=last_trade_data["country_id"] print("_______get_user_profile_client__________") print(Iq.get_user_profile_client(user_id)) pro_data=Iq.get_user_profile_client(user_id) print("\n\n")  print("___________request_leaderboard_userinfo_deals_client______") print(Iq.request_leaderboard_userinfo_deals_client(user_id,counutry_id)) user_data=Iq.request_leaderboard_userinfo_deals_client(user_id,counutry_id) worldwide=user_data["result"]["entries_by_country"]["0"]["position"] profit=user_data["result"]["entries_by_country"]["0"]["score"] print("\n") print("user_name:"+pro_data["user_name"]) print("This week worldwide:"+str(worldwide)) print("This week's gross profit:"+str(profit)) print("\n\n")  print("___________get_users_availability____________") print(Iq.get_users_availability(user_id)) print("\n\n") print("_____________unscribe_live_deal_______________") Iq.unscribe_live_deal(name,active,_type)`

### get\_user\_profile\_client()

this api can get user name and image

`Iq.get_user_profile_client(user_id)`

### request\_leaderboard\_userinfo\_deals\_client()

this api can get user detail

`Iq.request_leaderboard_userinfo_deals_client(user_id,counutry_id)`

### get\_users\_availability()

`Iq.get_users_availability(user_id)`

[

Previous 2 factor Auth



](../../2%20factor/ "2 factor Auth")[

Next For all

](../../all/all/ "For all")

[](https://github.com/iqoptionapi/iqoptionapi "github.com")

{"clipboard.copy": "Copy to clipboard", "clipboard.copied": "Copied to clipboard", "search.config.lang": "en", "search.config.pipeline": "trimmer, stopWordFilter", "search.config.separator": "\[\\\\s\\\\-\]+", "search.result.placeholder": "Type to start searching", "search.result.none": "No matching documents", "search.result.one": "1 matching document", "search.result.other": "# matching documents"} app = initialize({ base: "../../..", features: \[\], search: Object.assign({ worker: "../../../assets/javascripts/worker/search.37585f48.min.js" }, typeof search !== "undefined" && search) })