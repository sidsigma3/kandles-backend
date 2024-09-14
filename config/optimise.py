import random
import numpy as np
from deap import base, creator, tools, algorithms
import datetime as dt
from json import JSONEncoder
from io import BytesIO
import pandas as pd
import pandas_ta as ta
import os
import json
import sys
from pandas._libs.tslibs.timestamps import Timestamp


buf = BytesIO()
input_data = sys.stdin.read()


# Parse the JSON data
data = json.loads(input_data)
constraints = data.get('constraints', {})
variable_inputs = data.get('variableInputs', [])
goal = data.get('goal')
strategy = data.get('strategy')

stopLoss=strategy['slPct']
target=strategy['targetPct']
symbols = strategy['backSymbol']
start_date = strategy['startDate']
end_date = strategy['endDate']
initial_capital = strategy['backCapital']
quantity=strategy['backQuantity']

graph_type = strategy['graphType']


position_size_type = strategy['positionSizeType']
max_quantity = strategy.get('maxQuantity', None)
max_size_amount = strategy.get('sizeAmount',None)

moveSl = strategy['moveSlPct']
moveInstrument = strategy['moveInstrumentPct']
time_period = strategy['timePeriod']
trade_type = strategy['marketType']

conditions = strategy['strategyDetails']['conditions']
operator = strategy['strategyDetails']['logicalOperators']

exit_conditions = strategy['strategyDetailsExit']['conditions']
exit_operator = strategy['strategyDetailsExit']['logicalOperators']

conditions2 = strategy['strategyDetails2']['conditions']
operator2 = strategy['strategyDetails2']['logicalOperators']

exit_conditions2 = strategy['strategyDetailsExit2']['conditions']
exit_operator2 = strategy['strategyDetailsExit2']['logicalOperators']

max_long_entry = strategy['maxLong']
max_short_entry = strategy['maxShort']

selected_days = strategy['selectedDaysForIndi']


constraint_condition = False

# constraints = {}
# goal ={
#     "optimize": "Maximize",
#     "parameter": ""
# }

# variable_inputs =  [
#     {
#         "indicatorOne": {
#             "indiInputs": {
#                 "period": {
#                     "from": "1",
#                     "to": "20"
#                 }
#             }
#         },
#         "indicatorTwo": {
#             "indiInputs": {
#                 "period": {
#                     "from": "10",
#                     "to": "30"
#                 }
#             }
#         }
#     }
# ]

# variable_inputs = [
#     {
#         "indicatorOne": {
#             "indiInputs": {
#                 "fastPeriod": {
#                     "from": "1",
#                     "to": "30"
#                 },
#                 "slowPeriod": {
#                     "from": "1",
#                     "to": "33"
#                 },
#                 "signalPeriod": {
#                     "from": "1",
#                     "to": "33"
#                 }
#             }
#         },
#         "indicatorTwo": {
#             "indiInputs": {
#                 "fastPeriod": {
#                     "from": "1",
#                     "to": "33"
#                 },
#                 "slowPeriod": {
#                     "from": "1",
#                     "to": "33"
#                 },
#                 "signalPeriod": {
#                     "from": "1",
#                     "to": "33"
#                 }
#             }
#         }
#     }
# ]

# stopLoss = 0.01
# target = 0.02

# start_date= '01-01-2020'
# end_date = '01-01-2022'
# initial_capital = 100000000
# quantity = 50

# symbols = [
#   "HDFCBANK",
#   "SBIN"
# ]

# conditions = [
#     {
#         "indicatorOne": {
#             "value": "rsi",
#             "displayValue": "RSI",
#             "indiInputs": {
#                 "period": 14
#             }
#         },
#         "comparator": "crosses-above",
#         "indicatorTwo": {
#             "value": "rsi",
#             "displayValue": "RSI",
#             "indiInputs": {
#                 "period": 23
#             }
#         }
#     }
# ]


# conditions = [
#     {
#         "indicatorOne": {
#             "value": "macd",
#             "displayValue": "MACD",
#             "indiInputs": {
#                 "fastPeriod": 12,
#                 "slowPeriod": 26,
#                 "signalPeriod": 9
#             }
#         },
#         "comparator": "crosses-above",
#         "indicatorTwo": {
#             "value": "macd",
#             "displayValue": "MACD",
#             "indiInputs": {
#                 "fastPeriod": 23,
#                 "slowPeriod": 35,
#                 "signalPeriod": 19
#             }
#         }
#     }
# ]

# conditions=[]
# operator = []
# conditions2=[]
# operator2=[]

# conditions2 = [
#     {
#         "indicatorOne": {
#             "value": "rsi",
#             "displayValue": "RSI",
#             "indiInputs": {
#                 "period": 14
#             }
#         },
#         "comparator": "crosses-below",
#         "indicatorTwo": {
#             "value": "rsi",
#             "displayValue": "RSI",
#             "indiInputs": {
#                 "period": 23
#             }
#         }
#     }
# ]


# conditions2 = [
#     {
#         "indicatorOne": {
#             "value": "macd",
#             "displayValue": "MACD",
#             "indiInputs": {
#                 "fastPeriod": 12,
#                 "slowPeriod": 26,
#                 "signalPeriod": 9
#             }
#         },
#         "comparator": "crosses-below",
#         "indicatorTwo": {
#             "value": "macd",
#             "displayValue": "MACD",
#             "indiInputs": {
#                 "fastPeriod": 22,
#                 "slowPeriod": 34,
#                 "signalPeriod": 17
#             }
#         }
#     }
# ]

# operator2 = []

# # exit_conditions = [
# #         {
# #             "indicatorOne": {
# #                 "value": "ema",
# #                 "displayValue": "EMA",
# #                 "indiInputs": {
# #                     "period": 14
# #                 }
# #             },
# #             "comparator": "crosses-below",
# #             "indicatorTwo": {
# #                 "value": "sma",
# #                 "displayValue": "SMA",
# #                 "indiInputs": {
# #                     "period": 14
# #                 }
# #             }
# #         }
# #     ]

# exit_operator= []
# exit_conditions =[]

# exit_operator2= []
# exit_conditions2 =[]

# strategy_type = 'sell'
# trailing_sl = 0.1

# graph_type = 'candle'

# position_size_type = None
# max_quantity = None
# max_size_amount = None
# moveSl = None
# moveInstrument = None
# time_period = 'hourly'
# trade_type = 'cnc'
# max_long_entry = 3
# max_short_entry = 3

# selected_days = {
#     "Mon": True,
#     "Tue": True,
#     "Wed": True,
#     "Thu": True,
#     "Fri": True
# }



def compute_heikin_ashi(data):
    heikin_ashi = pd.DataFrame(index=data.index)
    

    open_col = 'open'
    high_col = 'high'
    low_col = 'low'
    close_col = 'close'

  
    heikin_ashi[close_col] = (data[open_col] + data[high_col] + data[low_col] + data[close_col]) / 4
    heikin_ashi[open_col] = (heikin_ashi[close_col].shift(1) + heikin_ashi[close_col].shift(1)) / 2
    heikin_ashi[high_col] = data[[open_col, close_col, high_col]].max(axis=1)
    heikin_ashi[low_col] = data[[open_col, close_col, low_col]].min(axis=1)

    
    heikin_ashi[open_col].iloc[0] = data[open_col].iloc[0]
    heikin_ashi.fillna(method='backfill', inplace=True)  # Handle NaN values for HA_Open in the first row

    return heikin_ashi



default_output_selection = {
    'macd': 'line',  # Use the MACD signal line
    'macd-s':'signal' ,   
    'adx': 'main',     # Use the main ADX line
    'bollinger': 'middle',  # Use the middle Bollinger Band
    'stoch': 'stoch_d',     # Use the %D line of Stochastic
    'supertrend': 'trend',  # Use the main SuperTrend line
    'ichimoku': 'kijun_sen', # Use the Kijun-sen line of Ichimoku
    'ppo': 'ppo_line'  ,     # Use the main line of PPO
    
}


def get_output_column_name(indicator_name, params):
    if indicator_name in default_output_selection:
        output_key = default_output_selection[indicator_name]
        if indicator_name in ['macd', 'macd-s']:
            fast = params.get('fastPeriod', 12)
            slow = params.get('slowPeriod', 26)
            signal = params.get('signalPeriod', 9)

            # Ensure fast is the minimum and slow is the maximum
            fast, slow = min(fast, slow), max(fast, slow)

            outputs = {
                'line': f"MACD_{fast}_{slow}_{signal}",
                'hist': f"MACDh_{fast}_{slow}_{signal}",
                'signal': f"MACDs_{fast}_{slow}_{signal}"
            }

        elif indicator_name == 'bollinger':
            period = params.get('period', 20)
            std_dev = params.get('stdDev', 2)
            outputs = {
                'upper': f"BBU_5_2.0",
                'middle': f"BBM_5_2.0",
                'lower': f"BBL_5_2.0"
            }

        elif indicator_name == 'stoch':
            k_period = params.get('kPeriod', 14)
            d_period = params.get('dPeriod', 3)
            outputs = {
                'stoch_k': f"STOCHk_14_3_3",
                'stoch_d': f"STOCHd_14_3_3"
            }

        elif indicator_name == 'adx':
            period = params.get('period', 14)
            outputs = {
                'main': f"ADX_{period}",
                'plus_di': f"DMP_{period}",
                'minus_di': f"DMN_{period}"
            }

        elif indicator_name == 'supertrend':
            period = params.get('period', 10)  # Default to 10 if not provided
            multiplier = params.get('multiplier', 3)  # Default to 3 if not provided
            outputs = {
                'trend': f"SUPERT_{period}_{multiplier}.0",
                'detail': f"SUPERTd_{period}_{multiplier}.0",
                'low': f"SUPERTl_{period}_{multiplier}.0",
                'signal': f"SUPERTs_{period}_{multiplier}.0"
            }
        return outputs[output_key]
    else:
        return f"{indicator_name.upper()}"  # Default to a generic name if not specified

    


indicator_data_requirements = {
    'adx': ['high', 'low', 'close'],
    'rsi': ['close'],
    'macd': ['close'],
    'macd-s':['close'],
    'stoch': ['high', 'low', 'close'],
    'vwap': ['high', 'low', 'close', 'volume'],
    'bollinger': ['close'],
    'parabolic_sar': ['high', 'low', 'close'],
    'ichimoku': ['high','low','close'],
    'psar':['high','low'],
    'atr':['high','low','close'],
    'cci':['high','low','close'],
    'supertrend':['high','low','close']
   
}



def map_params(indicator_name, user_params):
   
    param_mapping = {
        'rsi': {'period': 'length'},
        'macd': {'fastPeriod': 'fast', 'slowPeriod': 'slow', 'signalPeriod': 'signal'},
        'macd-s': {'fastPeriod': 'fast', 'slowPeriod': 'slow', 'signalPeriod': 'signal'},
        'stochastic': {'kPeriod': 'k', 'dPeriod': 'd', 'slowing': 'slow'},
        'bollinger': {'period': 'length', 'stdDev': 'std'},
        'parabolic_sar': {'step': 'accel', 'max': 'max_af'},
        'ema': {'period': 'length'},
        'sma': {'period': 'length'},
        'wma': {'period': 'length'},
        'tma': {'period': 'length'},  # Triangular Moving Average
        'cci': {'period': 'length'},  # Commodity Channel Index
        'atr': {'period': 'length'},  # Average True Range
        'adx': {'period': 'length'},  # Average Directional Index
        'obv': {},  # On-Balance Volume does not require additional parameters
        'vwap': {},  # VWAP typically does not take parameters, but might be configurable in some implementations
        'ichimoku': {'conversionLinePeriod': 'tenkan', 'baseLinePeriod': 'kijun', 'laggingSpan2Period': 'senkou_b', 'displacement': 'shift'},
        'stochrsi': {'period': 'length'},  # Stochastic RSI
        'williams': {'period': 'length'},  # Williams %R
        'keltner_channel': {'length': 'length', 'mult': 'scalar', 'offset': 'offset'},
        'ultimate_oscillator': {'short': 'timeperiod1', 'medium': 'timeperiod2', 'long': 'timeperiod3', 'ws': 'weight1', 'wm': 'weight2', 'wl': 'weight3'},
        'trix': {'length': 'length'},
        'apo': {'fast': 'fastperiod', 'slow': 'slowperiod'},
        'ppo': {'fast': 'fastperiod', 'slow': 'slowperiod', 'signal': 'signalperiod'},
        'mom': {'period': 'length'},
        'dmi': {'period': 'length'},  # DMI or ADX
        'bbwidth': {'period': 'length'},
        'supertrend': {'period': 'length', 'multiplier': 'multiplier'}, 
        'pivot_point': {'period': 'length'},  
        'rsi_ma': {'period': 'length', 'ma_period': 'ma_length'}, 
        'plus_di': {'period': 'length'}, 
        'minus_di': {'period': 'length'} 
    }

    # Get the specific mapping for the given indicator
    current_mapping = param_mapping.get(indicator_name, {})
    
    # Translate user parameter names to pandas_ta parameter names
    translated_params = {current_mapping.get(k, k): v for k, v in user_params.items()}

    return translated_params


def calculate_indicator(dataframe, indicator_details):
  
    indicator_name = indicator_details['value']
    
    temp_name=''

    if indicator_name == 'macd-s':
        temp_name = 'macd'

    user_inputs = indicator_details.get('indiInputs', {})
    params = indicator_details.get('indiInputs', {})
    
    # Map user inputs to the library-specific parameter names
    func_params = map_params(indicator_name, user_inputs)
    required_columns = indicator_data_requirements.get(indicator_name, ['close'])
    column_map = {
        'high': 'high',
        'low': 'low',
        'open': 'open',
        'close': 'close',
        'volume': 'volume'
    }

    data_columns = {col: dataframe[column_map[col]] for col in required_columns if col in column_map}

    if indicator_name == 'close':
        price_series = dataframe['close'].shift(-(indicator_details['indiInputs']['offset']))  
        return price_series
       

    elif indicator_name == 'open':
        price_series = dataframe['open'].shift(-(indicator_details['indiInputs']['offset']))  
        return price_series
       
    
    elif indicator_name == 'high':
        price_series = dataframe['high'].shift(-(indicator_details['indiInputs']['offset'])) 
        return price_series
        
    
    elif indicator_name =='low':
        price_series = dataframe['low'].shift(-(indicator_details['indiInputs']['offset']))  
        return price_series
       

    elif indicator_name == 'volume':
        # Special handling for volume, which uses a different column name
        volume_series = dataframe['volume'].shift(-(indicator_details['indiInputs']['offset']))  # Apply offset
        return volume_series   
    
       
    elif indicator_name == 'number':
        # Return a constant series if the indicator is a fixed number
        constant_value = float(indicator_details['indiInputs']['number'])
        return pd.Series([constant_value] * len(dataframe), index=dataframe.index)
    
    

    try:
        # func_params = map_params(indicator_name, indicator_details['indiInputs'])
        if temp_name == 'macd':
            indicator_function = getattr(ta, temp_name.lower())
        else:
            indicator_function = getattr(ta, indicator_name.lower())
        
        result = indicator_function(**data_columns, **func_params)
        
    
        if result.empty:
            raise ValueError(f"Indicator calculation failed: {indicator_name} returned no data.")
        # Select the specific output column if necessary (for example, for MACD, select signal line)
        output_column = get_output_column_name(indicator_name, params)

        
        
        if isinstance(result, pd.DataFrame) and output_column in result.columns:
           
            return result[output_column]
        
        return result
    
        # if len(data_columns) == 1:
        #     result = indicator_function(list(data_columns.values())[0], **func_params)
        # else:
        #     result = indicator_function(**data_columns, **func_params)

        # print(result)

        # if result is None or result.empty:
        #     raise ValueError(f"Indicator calculation failed: {indicator_name} returned no data.")
        # return result

        


    except AttributeError:
        print(f"Error: Indicator '{indicator_name}' is not supported by pandas_ta or is incorrectly named.")
        return pd.Series([None] * len(dataframe), index=dataframe.index)
    except Exception as e:
        print(f"Error calculating {indicator_name}: {e}")
        return pd.Series([None] * len(dataframe), index=dataframe.index)




def evaluate_condition(data, entry_condition, exit_condition ):
    # Calculate indicators for entry conditions
    entry_indicator_one_value = calculate_indicator(data, entry_condition['indicatorOne'])
    entry_indicator_two_value = calculate_indicator(data, entry_condition['indicatorTwo'])
    
    # Calculate indicators for exit conditions

   
   
    # Initialize signals to False for the entire series
    buy_signal = pd.Series([False] * len(data), index=data.index)
    sell_signal = pd.Series([False] * len(data), index=data.index)

    # Evaluate entry condition for buy signal
    if entry_condition['comparator'] == 'crosses-below':
        buy_signal = (entry_indicator_one_value < entry_indicator_two_value) & (entry_indicator_one_value.shift(1) >= entry_indicator_two_value.shift(1))
        sell_signal = (entry_indicator_one_value > entry_indicator_two_value) & (entry_indicator_one_value.shift(1) <= entry_indicator_two_value.shift(1))
    
    elif entry_condition['comparator'] == 'crosses-above':
        buy_signal = (entry_indicator_one_value > entry_indicator_two_value) & (entry_indicator_one_value.shift(1) <= entry_indicator_two_value.shift(1))
        sell_signal = (entry_indicator_one_value < entry_indicator_two_value) & (entry_indicator_one_value.shift(1) >= entry_indicator_two_value.shift(1))

    elif entry_condition['comparator'] == 'equal-to':
        buy_signal = entry_indicator_one_value == entry_indicator_two_value
        sell_signal = entry_indicator_one_value != entry_indicator_two_value
    
    elif entry_condition['comparator'] == 'higher-than':
        buy_signal = entry_indicator_one_value > entry_indicator_two_value
        sell_signal = entry_indicator_one_value <= entry_indicator_two_value
    
    elif entry_condition['comparator'] == 'lower-than':
        buy_signal = entry_indicator_one_value < entry_indicator_two_value
        sell_signal = entry_indicator_one_value >= entry_indicator_two_value


    if exit_condition:
        exit_indicator_one_value = calculate_indicator(data, exit_condition['indicatorOne'])
        exit_indicator_two_value = calculate_indicator(data, exit_condition['indicatorTwo'])

    # Evaluate exit condition for sell signal
        if exit_condition['comparator'] == 'crosses-below':
            sell_signal = (exit_indicator_one_value < exit_indicator_two_value) & (exit_indicator_one_value.shift(1) >= exit_indicator_two_value.shift(1))
        elif exit_condition['comparator'] == 'crosses-above':
            sell_signal = (exit_indicator_one_value > exit_indicator_two_value) & (exit_indicator_one_value.shift(1) <= exit_indicator_two_value.shift(1))
        elif exit_condition['comparator'] == 'equal-to':
            sell_signal = exit_indicator_one_value == exit_indicator_two_value
        elif exit_condition['comparator'] == 'higher-than':
            sell_signal = exit_indicator_one_value > exit_indicator_two_value
        elif exit_condition['comparator'] == 'lower-than':
            sell_signal = exit_indicator_one_value < exit_indicator_two_value

    return buy_signal, sell_signal


def combine_signals(data, entry_conditions, entry_logical_operators, exit_conditions=None, exit_logical_operators=None):
    
    if len(entry_conditions) == 1:
        buy_signal, sell_signal = evaluate_condition(data, entry_conditions[0], exit_conditions[0] if exit_conditions else None)
        return buy_signal, sell_signal
    
    elif len(entry_conditions) > 1:
        final_buy_signal, final_sell_signal = evaluate_condition(data, entry_conditions[0], exit_conditions[0] if exit_conditions else None)
        for i in range(1, len(entry_conditions)):
            next_buy_signal, next_sell_signal = evaluate_condition(data, entry_conditions[i], exit_conditions[i] if exit_conditions else None)
            if entry_logical_operators and i-1 < len(entry_logical_operators):
                if entry_logical_operators[i-1].upper() == 'AND':
                    final_buy_signal &= next_buy_signal
                    final_sell_signal &= next_sell_signal
                elif entry_logical_operators[i-1].upper() == 'OR':
                    final_buy_signal |= next_buy_signal
                    final_sell_signal |= next_sell_signal

        if exit_conditions:
            for j in range(len(exit_conditions)):
                exit_buy_signal, exit_sell_signal = evaluate_condition(data, entry_conditions[j], exit_conditions[j])
                if exit_logical_operators and j < len(exit_logical_operators):
                    if exit_logical_operators[j].upper() == 'AND':
                        final_buy_signal &= exit_buy_signal
                        final_sell_signal |= exit_sell_signal
                    elif exit_logical_operators[j].upper() == 'OR':
                        final_buy_signal |= exit_buy_signal
                        final_sell_signal |= exit_sell_signal

        return final_buy_signal, final_sell_signal
    else:
        return pd.Series([False] * len(data)), pd.Series([False] * len(data))





file_path_vix = os.path.join(r'D:\stock_historical_data', f'historical_data_{time_period}', 'INDIA VIX.csv')

df_vix = pd.read_csv(file_path_vix)

df_vix['date'] = pd.to_datetime(df_vix['date'], utc=False)
df_vix.set_index('date', inplace=True)  # Set the 'date' column as the index

def classify_vix(high,low):
    return (high,low)

# Apply classification to VIX data
# df_vix['vix_classification'] = classify_vix(df_vix['high'],df_vix['low'])
df_vix['vix_classification'] = df_vix.apply(lambda row: classify_vix(row['high'], row['low']), axis=1)

# df_vix['vix_classification'] = df_vix['close'].apply(classify_vix)

vix_classification_dict = df_vix['vix_classification'].to_dict()




data_directory = r'D:\stock_historical_data\historical_data_daily' 

# function to get sentiment of day using open and close price

def determine_sentiment(open_price, close_price):
    if close_price > open_price * 1.01:
        return 'bullish'
    elif close_price < open_price * 0.99:
        return 'bearish'
    else:
        return 'sideways'

market_sentiment_data = {}

for instrument in symbols:
    file_path = os.path.join(data_directory, f'{instrument}.csv')
    
    # Load the DataFrame for the current instrument
    data_frame = pd.read_csv(file_path)
    data_frame['date'] = pd.to_datetime(data_frame['date'], utc=False)
    data_frame.set_index('date', inplace=True)  # Set the 'date' column as the index
    
    # Iterate over each row in the DataFrame
    for index, row in data_frame.iterrows():
        open_price = row['open']
        close_price = row['close']
        date = index.date()  # Extract the date from the index
        
        # Determine the market sentiment for the current row
        sentiment = determine_sentiment(open_price, close_price)
        
        # Store the sentiment data in the market_sentiment_data dictionary
        if instrument not in market_sentiment_data:
            market_sentiment_data[instrument] = {}
        market_sentiment_data[instrument][date] = sentiment






def is_within_trading_hours(current_date,start_time='09:15', end_time='15:15'):
    if time_period == 'daily':
        return True
    
    current_time = current_date.time()
    start_time = dt.datetime.strptime(start_time, '%H:%M').time()
    end_time = dt.datetime.strptime(end_time, '%H:%M').time()
    return start_time <= current_time < end_time


def execute_and_analyze_strategy(symbol,data, stop_loss_pct, price_move_pct, trailing_stop_loss_pct, target_pct, initial_funds, quantity, trade_type,position_size_type=None, max_size_amount=None, max_quantity=None, exit_conditions=None,exit_conditions_short=None):
    is_in_position = False
    is_in_position_short = False
    entry_price = 0
    entry_price_short = 0
    funds = initial_funds
    funds_short = initial_funds
    funds_total = initial_funds
    trades = []
    trades_short = []
    gains = []
    losses = []
    gains_short = []
    losses_short = []
    total_pnl = 0
    total_pnl_short = 0
    total_brokerage = 0
    total_brokerage_short = 0
    win_streak = 0
    loss_streak = 0
    current_streak = 0
    win_streak_short = 0
    loss_streak_short = 0
    current_streak_short = 0
    wins = 0
    losses_count = 0
    wins_short = 0
    losses_count_short = 0
    last_outcome = None
    last_outcome_short = None
    peak_funds = initial_funds
    max_drawdown = 0
    max_drawdown_days = 0
    drawdown_start = None
    peak_funds_short = initial_funds
    max_drawdown_short = 0
    max_drawdown_days_short = 0
    drawdown_start_short = None
    peak_funds_total = initial_funds
    max_drawdown_total = 0
    max_drawdown_days_total = 0
    drawdown_start_total = None
    trade_dates = [data.index[0]]
    cumulative_pnl = [0]
    trade_dates_short = [data.index[0]]
    cumulative_pnl_short = [0]
    trailing_stop_loss_price = 0
    trailing_stop_loss_price_short = 0
    invested_fund = 0
    invested_fund_short = 0
    tsl_count = 0
    tsl_count_short = 0
    target_count = 0
    target_count_short = 0
    stoploss_count = 0
    stoploss_count_short = 0
    entry_price_reference = 0
    entry_price_reference_short = 0
    sell_signal_count = 0
    sell_signal_count_short = 0
    market_close = 0
    market_close_short = 0
    quantity_short = quantity
    trade_number = 0
    long_trade_number = 0
    short_trade_number = 0

    buy_signals = 0
    sell_signals = 0

    daily_trade_count = 0
    daily_short_trade_count = 0
    current_day = data.index[0].date()

    def is_day_selected(current_date, selected_days):
        day_name = current_date.strftime('%a')
        return selected_days.get(day_name, False)

# loop over every candle data
    for i in range(len(data)):
        current_date = data.index[i]
        high_price = data['high'].iloc[i]
        low_price = data['low'].iloc[i]
        close_price = data['close'].iloc[i]
        open_price = data['open'].iloc[i]
        brokerage_per_trade = min(close_price * quantity * 0.0003, 20)
        day_name = current_date.strftime('%a')
        
        if not is_day_selected(current_date, selected_days):
            continue


        if data['Buy_Signal'].iloc[i] == True:
            buy_signals+=1

        if data['Sell_Signal'].iloc[i]==True:
            sell_signals+=1

        
        #market sentiment data retrive
        day_type = market_sentiment_data[symbol].get(current_date.date(), 'sideways')

        vix_classification = vix_classification_dict.get(current_date, 'unknown')
        
        #last candle of the day flag
        if i < len(data) - 1 and current_date.date() != data.index[i+1].date():
            is_last_candle_of_day = True

        else:
            is_last_candle_of_day = False

        if current_date.date() != current_day:
            daily_trade_count = 0
            daily_short_trade_count = 0
            current_day = current_date.date()
        
        if is_in_position:
            exit_price = None
            action = None
        
            if exit_conditions:
                exit_signal = data['Sell_Signal'].iloc[i]
                if exit_signal:
                    exit_price = close_price
                    action = 'exit_signal'
                    sell_signal_count += 1

            if not exit_price and stop_loss_pct!= 0:
               #long condition
                    target_reached_price = entry_price * (1 + target_pct)
                    
                    if trailing_stop_loss_pct is not None and trailing_stop_loss_pct != 0:
                        # Check if the close price exceeds the highest profit achieved
                        if high_price >= entry_price_reference * (1 + price_move_pct):
                            next_reference_price = entry_price_reference * (1 + price_move_pct)
                            # Move the trailing stop loss in the profit direction
                            while high_price >= next_reference_price:
                                entry_price_reference = next_reference_price
                                trailing_stop_loss_price *= (1 + trailing_stop_loss_pct)
                                next_reference_price = entry_price_reference * (1 + price_move_pct)
                         
                        if low_price <= trailing_stop_loss_price:
                            action = 'tsl'
                            tsl_count += 1
                            exit_price = trailing_stop_loss_price
                    if high_price >= target_reached_price:
                        action = 'target'
                        target_count += 1
                        exit_price = target_reached_price
             

            if trade_type == 'mis' and is_last_candle_of_day:
                action = 'Market Close'
                exit_price = close_price
                market_close += 1

            if not exit_price and stop_loss_pct!= 0:
                stop_loss_price = entry_price * (1 - stop_loss_pct)
                if low_price <= stop_loss_price:
                    action = 'stoploss'
                    stoploss_count += 1
                    exit_price = stop_loss_price
    

            if exit_price:
                pnl = (exit_price - entry_price) * quantity - brokerage_per_trade 
                funds += (exit_price * quantity - brokerage_per_trade)
                funds_total += (exit_price * quantity - brokerage_per_trade) 
                total_pnl += pnl
                total_brokerage += brokerage_per_trade

                invested_price = entry_price * quantity + brokerage_per_trade
                invested_fund += invested_price
                trades.append({
                    'symbol': symbol, 
                    'action': 'Sell' ,
                    'price': exit_price,
                    'pnl': pnl,
                    'quantity': quantity,
                    'brokerage': brokerage_per_trade,
                    'date': current_date,
                    'invested': invested_price,
                    'stopLossprice': stop_loss_price,
                    'targetPrice': target_reached_price,
                    'trailingSl': trailing_stop_loss_price,
                    'trigger': action,
                    'day_type': day_type,
                    'signalType':'Long Exit',
                    'tradeNumber':long_trade_number,
                    'vix':vix_classification,
                    'day':day_name
                    
                })
                is_in_position = False
                trailing_stop_loss_price = 0
                trade_dates.append(current_date)
                cumulative_pnl.append(cumulative_pnl[-1] + pnl)

                # Track win or loss
                if pnl > 0:
                    wins += 1
                    gains.append(pnl)
                    if last_outcome == 'win':
                        current_streak += 1
                    else:
                        current_streak = 1
                    win_streak = max(win_streak, current_streak)
                    last_outcome = 'win'
                else:
                    losses_count += 1
                    losses.append(-pnl)
                    if last_outcome == 'loss':
                        current_streak += 1
                    else:
                        current_streak = 1
                    loss_streak = max(loss_streak, current_streak)
                    last_outcome = 'loss'

                if funds > peak_funds:
                    peak_funds = funds
                    drawdown_start = None
                else:
                    if drawdown_start is None:
                        drawdown_start = current_date
                    current_drawdown = (peak_funds - funds) / peak_funds
                    current_drawdown_percentage = current_drawdown * 100  
                 
                    max_drawdown = max(max_drawdown, current_drawdown_percentage)
                
                    if drawdown_start is not None:
                        drawdown_days = (current_date - drawdown_start).days
                        max_drawdown_days = max(max_drawdown_days, drawdown_days)

                if funds_total > peak_funds_total:
                    peak_funds_total = funds
                    drawdown_start_total = None
                else:
                    if drawdown_start_total is None:
                        drawdown_start_total = current_date
                    current_drawdown_total = (peak_funds_total - funds) / peak_funds_total
                    current_drawdown_percentage_total = current_drawdown_total * 100  
                 
                    max_drawdown_total = max(max_drawdown_total, current_drawdown_percentage_total)
                
                    if drawdown_start_total is not None:
                        drawdown_days_total = (current_date - drawdown_start_total).days
                        max_drawdown_days_total = max(max_drawdown_days_total, drawdown_days_total)
                                
                
                # if funds > peak_funds:
                #     peak_funds = funds
                #     drawdown_start = None
                # else:
                #     if drawdown_start is None:
                #         drawdown_start = current_date
                #     current_drawdown = (funds - peak_funds)/peak_funds
                  
                #     if current_drawdown < max_drawdown:
                #         max_drawdown = current_drawdown
                #         max_drawdown_days = (current_date - drawdown_start).days  


                # if funds > peak_funds:
                #     peak_funds = funds
                #     peak_date = current_date
                #     trough_value = funds
                #     trough_date = current_date
                # elif funds < trough_value:
                #     trough_value = funds
                #     trough_date = current_date
                #     current_drawdown = (trough_value - peak_funds) / peak_funds
                #     if current_drawdown < max_drawdown:
                #         max_drawdown = current_drawdown
                #         max_drawdown_days = (trough_date - peak_date).days
    

        if is_in_position_short:
            action_short = None
            exit_price_short = None

            if exit_conditions_short:
                exit_signal_short = data['exit_signals_short'].iloc[i]
                if exit_signal_short:
                    exit_price_short = close_price
                    action_short = 'exit_signal'
                    sell_signal_count_short += 1
    

            if not exit_price_short and stop_loss_pct!= 0:
                #shrt condition
                    target_reached_price_short = entry_price_short * (1 - target_pct)
                    
                    if trailing_stop_loss_pct is not None and trailing_stop_loss_pct != 0:
                        # Check if the close price exceeds the highest profit achieved
                        if low_price <= entry_price_reference_short * (1 - price_move_pct):
                            next_reference_price_short = entry_price_reference_short * (1 - price_move_pct)
                            # Move the trailing stop loss in the profit direction
                            while low_price <= next_reference_price_short:
                                entry_price_reference_short = next_reference_price_short
                                trailing_stop_loss_price_short *= (1 - trailing_stop_loss_pct)
                                next_reference_price_short = entry_price_reference_short * (1 - price_move_pct)
                         
                        if high_price >= trailing_stop_loss_price_short:
                            action_short = 'tsl'
                            tsl_count_short += 1
                            exit_price_short = trailing_stop_loss_price_short
                    if low_price <= target_reached_price_short:
                        action_short = 'target'
                        target_count_short += 1
                        exit_price_short = target_reached_price_short

            if trade_type == 'mis' and is_last_candle_of_day:
                market_close_short+=1
                action_short = 'Market Close'
                exit_price_short = close_price


            if not exit_price_short and stop_loss_pct!= 0:
                stop_loss_price_short = entry_price_short * (1 + stop_loss_pct)
                if high_price >= stop_loss_price_short:
                    action_short = 'stoploss'
                    stoploss_count_short += 1
                    exit_price_short = stop_loss_price_short

                                  
            if exit_price_short:
                pnl_short =  (entry_price_short - exit_price_short) * quantity_short - brokerage_per_trade_short
                funds_short += (entry_price_short * quantity_short - exit_price_short * quantity_short - brokerage_per_trade_short)
                funds_total += (entry_price_short * quantity_short - exit_price_short * quantity_short - brokerage_per_trade_short)
                total_pnl_short += pnl_short
                total_brokerage_short += brokerage_per_trade_short
                invested_fund_short += invested_price_short
        
                trades_short.append({
                    'symbol': symbol,  
                    'action': 'Buy',
                    'price': exit_price_short,
                    'pnl': pnl_short,
                    'quantity': quantity_short,
                    'brokerage': brokerage_per_trade_short,
                    'date': current_date,
                    'invested': invested_price_short,
                    'stopLossprice': stop_loss_price_short,
                    'targetPrice': target_reached_price_short,
                    'trailingSl': trailing_stop_loss_price_short,
                    'trigger': action_short,
                    'day_type': day_type,
                    'signalType':'Short Exit',
                    'tradeNumber':short_trade_number,
                    'vix':vix_classification,
                    'day':day_name
                })
                is_in_position_short = False
                trailing_stop_loss_price_short = 0
                trade_dates.append(current_date)
             
                cumulative_pnl.append(cumulative_pnl[-1] + pnl_short)

                # Track win or loss
                if pnl_short > 0:
                    wins_short += 1
                    gains_short.append(pnl_short)
                    if last_outcome_short == 'win':
                        current_streak_short += 1
                    else:   
                        current_streak_short = 1
                    win_streak_short = max(win_streak_short, current_streak_short)
                    last_outcome_short = 'win'
                else:
                    losses_count_short += 1
                    losses_short.append(-pnl_short)
                    if last_outcome_short == 'loss':
                        current_streak_short += 1
                    else:
                        current_streak_short = 1
                    loss_streak_short = max(loss_streak_short, current_streak_short)
                    last_outcome_short = 'loss'



                if funds_short > peak_funds_short:
                    peak_funds_short = funds_short
                    drawdown_start_short = None
                else:
                    if drawdown_start_short is None:
                        drawdown_start_short = current_date
                    current_drawdown_short = (peak_funds_short - funds_short) / peak_funds_short
                    current_drawdown_percentage_short = current_drawdown_short * 100  
                 
                    max_drawdown_short = max(max_drawdown_short, current_drawdown_percentage_short)
                
                    if drawdown_start_short is not None:
                        drawdown_days_short = (current_date - drawdown_start_short).days
                        max_drawdown_days_short = max(max_drawdown_days_short, drawdown_days_short)


                if funds_total > peak_funds_total:
                    peak_funds_total = funds_total
                    drawdown_start_total = None
                else:
                    if drawdown_start_total is None:
                        drawdown_start_total = current_date
                    current_drawdown_total = (peak_funds_total - funds_total) / peak_funds_total
                    current_drawdown_percentage_total = current_drawdown_total * 100  
                 
                    max_drawdown_total = max(max_drawdown_total, current_drawdown_percentage_total)
                
                    if drawdown_start_total is not None:
                        drawdown_days_total = (current_date - drawdown_start_total).days
                        max_drawdown_days_total = max(max_drawdown_days_total, drawdown_days_total)

                # if funds > peak_funds:
                #     peak_funds = funds
                #     drawdown_start = None
                # else:
                #     if drawdown_start is None:
                #         drawdown_start = current_date
                #     current_drawdown = peak_funds - funds
                #     max_drawdown = max(max_drawdown, current_drawdown)
                #     if drawdown_start is not None:
                #         drawdown_days = (current_date - drawdown_start).days
                #         max_drawdown_days = max(max_drawdown_days, drawdown_days)
                    
                # if funds > peak_funds:
                #     peak_funds = funds
                #     drawdown_start = None
                # else:
                #     if drawdown_start is None:
                #         drawdown_start = current_date
                #     current_drawdown = (peak_funds - funds)/peak_funds
                #     if current_drawdown > max_drawdown:
                #         max_drawdown = current_drawdown
                #         max_drawdown_days = (current_date - drawdown_start).days  

                # if funds > peak_funds:
                #     peak_funds = funds
                #     peak_date = current_date
                #     trough_value = funds
                #     trough_date = current_date
                # elif funds < trough_value:
                #     trough_value = funds
                #     trough_date = current_date
                #     current_drawdown = (trough_value - peak_funds) / peak_funds
                #     if current_drawdown < max_drawdown:
                #         max_drawdown = current_drawdown
                #         max_drawdown_days = (trough_date - peak_date).days
    


                    
        entry_signal = data['Buy_Signal'].iloc[i] 
        entry_signal_short = data['entry_signals_short'].iloc[i]

        if entry_signal and not is_in_position and is_within_trading_hours(current_date) and conditions and daily_trade_count < max_long_entry:
            entry_price = close_price 
            trade_number+=1
            long_trade_number = trade_number

            if position_size_type:
                if position_size_type == 'capital':
                    quantity = min(max_size_amount // entry_price, max_quantity)
                elif position_size_type == 'risk':
                    stop_loss_price = entry_price * (1 - stop_loss_pct)
                    risk_per_share = entry_price - stop_loss_price
                    quantity = min(max_size_amount // risk_per_share, max_quantity)

            target_reached_price = (entry_price * target_pct) + entry_price
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            trailing_stop_loss_price = stop_loss_price if trailing_stop_loss_pct is not None else 0
            brokerage_per_trade = min(entry_price * quantity * 0.0003, 20)
            funds -= (entry_price * quantity + brokerage_per_trade) 
            funds_total -= (entry_price * quantity + brokerage_per_trade) 
            is_in_position = True
            entry_price_reference = entry_price
            invested_price = entry_price * quantity + brokerage_per_trade
           
            trades.append({
                'symbol': symbol,  
                'action': 'Buy' ,
                'price': entry_price,
                'quantity': quantity,
                'trailingSlPrice': trailing_stop_loss_price,
                'brokerage': brokerage_per_trade,
                'date': current_date,
                'targetPrice': target_reached_price,
                'stopLossprice': stop_loss_price,
                'trailingSl': trailing_stop_loss_price,
                'trigger': 'Entry Signal',
                'day_type': day_type,
                'signalType':'Long Entry',
                'tradeNumber':long_trade_number,
                'vix':vix_classification,
                'day':day_name
            })
            trade_dates.append(current_date)
            cumulative_pnl.append(cumulative_pnl[-1])
            total_brokerage += brokerage_per_trade
            daily_trade_count+=1

        if entry_signal_short and not is_in_position_short and is_within_trading_hours(current_date) and conditions2 and daily_short_trade_count < max_short_entry:
            entry_price_short = close_price 

            # if is_in_position:
            #     trade_number +=1
            #     short_trade_number = trade_number
            
            trade_number +=1
            short_trade_number = trade_number

            # Adjust quantity based on position sizing type if provided
            if position_size_type:
                if position_size_type == 'capital':
                    quantity_short = min(max_size_amount // entry_price_short, max_quantity)
                elif position_size_type == 'risk':
                    stop_loss_price = entry_price_short * (1 - stop_loss_pct)
                    risk_per_share = entry_price_short - stop_loss_price_short
                    quantity_short = min(max_size_amount // risk_per_share, max_quantity)

            target_reached_price_short = entry_price_short - (entry_price_short * target_pct) 
            stop_loss_price_short = entry_price_short * (1 + stop_loss_pct)
            trailing_stop_loss_price_short = stop_loss_price_short if trailing_stop_loss_pct is not None else 0
            brokerage_per_trade_short = min(entry_price_short * quantity_short * 0.0003, 20)
            funds_short -= (entry_price_short * quantity_short - brokerage_per_trade_short)
            funds_total -= (entry_price_short * quantity_short - brokerage_per_trade_short)
            is_in_position_short = True
            entry_price_reference_short = entry_price_short
            invested_price_short = entry_price_short * quantity_short + brokerage_per_trade_short
           
            trades_short.append({
                'symbol': symbol,  
                'action': 'Sell',
                'price': entry_price_short,
                'quantity': quantity_short,
                'trailingSlPrice': trailing_stop_loss_price_short,
                'brokerage': brokerage_per_trade_short,
                'date': current_date,
                'targetPrice': target_reached_price_short,
                'stopLossprice': stop_loss_price_short,
                'trailingSl': trailing_stop_loss_price_short,
                'trigger': 'Entry Signal',
                'day_type': day_type,
                'signalType':'Short Entry',
                'tradeNumber':short_trade_number,
                'vix':vix_classification,
                'day':day_name
            })
            trade_dates.append(current_date)
            cumulative_pnl.append(cumulative_pnl[-1])
            total_brokerage_short += brokerage_per_trade_short 
            daily_short_trade_count +=1   

    # Final metrics calculation
    total_trades = wins + losses_count
    avg_gain = sum(gains) / wins if wins else 0
    avg_loss = sum(losses) / losses_count if losses_count else 0
    win_rate = (wins / total_trades * 100) if total_trades else 0
    loss_rate = (losses_count / total_trades * 100) if total_trades else 0
    reward_to_risk = avg_gain / avg_loss if avg_loss != 0 else 0
    expectancy = (reward_to_risk * win_rate / 100) - (loss_rate / 100) if total_trades else 0

    total_trades_short = wins_short + losses_count_short
    avg_gain_short = sum(gains_short) / wins_short if wins_short else 0
    avg_loss_short = sum(losses_short) / losses_count_short if losses_count_short else 0
    win_rate_short = (wins_short / total_trades_short * 100) if total_trades_short else 0
    loss_rate_short = (losses_count_short / total_trades_short * 100) if total_trades_short else 0
    reward_to_risk_short = avg_gain_short / avg_loss_short if avg_loss_short != 0 else 0
    expectancy_short = (reward_to_risk_short * win_rate_short / 100) - (loss_rate_short / 100) if total_trades_short else 0

    if avg_loss + avg_loss_short != 0:
        reward_to_risk_avg = (avg_gain + avg_gain_short) / (avg_loss + avg_loss_short)
    else:
        reward_to_risk_avg = 0  # or any appropriate value you want to set in this case

    if total_trades + total_trades_short != 0:
        win_rate_avg = (wins + wins_short) / (total_trades + total_trades_short) * 100
        loss_rate_avg = (losses_count + losses_count_short) / (total_trades + total_trades_short) * 100
        expectancy_total = (reward_to_risk_avg * (win_rate_avg / 100)) - (loss_rate_avg / 100)
    else:
        win_rate_avg = 0
        loss_rate_avg = 0
        expectancy_total = 0
    
    if total_trades == 0:
        invested_fund_total = invested_fund_short/total_trades_short if total_trades_short else 0
    elif total_trades_short == 0 :
        invested_fund_total = invested_fund/total_trades if total_trades else 0

    else:
        invested_fund_total = (invested_fund+invested_fund_short) / (total_trades+ total_trades_short)

    return {
        'trades': trades,
        'tradesShort':trades_short,
        'Total Signals': total_trades,
        'Number of Wins': wins,
        'Number of Losses': losses_count,
        'Winning Streak': win_streak,
        'Losing Streak': loss_streak,
        'gains':gains,
        'gainsShort':gains_short,
        'losses':losses,
        'lossesShort':losses_short,
        'Max Gains': max(gains, default=0),
        'Max Loss': max(losses, default=0),
        'Avg Gain per Winning Trade': avg_gain,
        'Avg Loss per Losing Trade': avg_loss,
        'Total Signals Short': total_trades_short,
        'Number of Wins Short': wins_short,
        'Number of Losses Short': losses_count_short,
        'Winning Streak Short': win_streak_short,
        'Losing Streak Short': loss_streak_short,
        'Max Gains Short': max(gains_short, default=0),
        'Max Loss Short': max(losses_short, default=0),
        'Avg Gain per Winning Trade Short': avg_gain_short,
        'Avg Loss per Losing Trade Short': avg_loss_short,
        'Max Drawdown': max_drawdown,
        'Max Drawdown Days': max_drawdown_days,
        'Max Drawdown Short': max_drawdown_short,
        'Max Drawdown Days Short': max_drawdown_days_short,
        'Max Drawdown Total': max_drawdown_total,
        'Max Drawdown Days Total': max_drawdown_days_total,
        'Win Rate (%)': win_rate,
        'Loss Rate (%)': loss_rate,
        'Expectancy': expectancy,
        'Profit Factor': reward_to_risk,
        'Total PnL': total_pnl,
        'Win Rate Short(%)': win_rate_short,
        'Loss Rate Short(%)': loss_rate_short,
        'Expectancy Short': expectancy_short,
        'Expectancy total':expectancy_total,
        'Profit Factor Short': reward_to_risk_short,
        'Profit Factor Total' : reward_to_risk_avg,
        'Total PnL Short': total_pnl_short,
        'Total Brokerage': total_brokerage,
        'Total Brokerage Short': total_brokerage_short,
        'Net PnL After Brokerage': total_pnl - total_brokerage,
        'Net PnL After Brokerage Short': total_pnl_short - total_brokerage_short,
        'Remaining Funds': funds,
        'Trade Dates': trade_dates,
        'targetCount': target_count,
        'tslCount': tsl_count,
        'slCount': stoploss_count,
        'targetCountShort': target_count_short,
        'tslCountShort': tsl_count_short,
        'slCountShort': stoploss_count_short,
        'investedFund': invested_fund/total_trades if total_trades is not 0 else 0,
        'investedFundShort': invested_fund_short/total_trades_short if total_trades_short is not 0 else 0,
        'investedTotal':invested_fund_total,
        'Cumulative PnL': cumulative_pnl,
        'sellSignalCount':sell_signal_count,
        'marketCloseCount':market_close,
        'marketCloseCountShort':market_close_short,
        
    }

def apply_constraints(strategy_metrics, constraints, goal):
    """
    Apply constraints to the strategy metrics and adjust the goal metric based on those constraints.

    Parameters:
        strategy_metrics (dict): The metrics obtained from strategy analysis.
        constraints (dict): The constraints to apply.
        goal (dict): The optimization goal (e.g., maximize or minimize a certain metric).

    Returns:
        tuple: Adjusted metric value and a boolean indicating if constraints are met.
    """
    metric_key = goal['parameter']
    metric_value = strategy_metrics.get(metric_key, 0)  # Default to 0 if metric not found

    constraints_met = True

    # Check constraints
    for constraint_key, constraint in constraints.items():
        try:
            percentage = float(constraint.get('weight', 0))  # Convert weight to float
        except ValueError:
            print(f"Invalid weight value: {constraint.get('weight', '0')}")
            percentage = 0.0

        direction = constraint.get('direction', 'Maximum')

        if constraint_key in strategy_metrics:
            value = strategy_metrics[constraint_key]

            if direction == 'Maximum':
                if value > percentage:  # Constraint is not met
                    constraints_met = False
                    break
            elif direction == 'Minimum':
                if value < percentage:  # Constraint is not met
                    constraints_met = False
                    break
            else:
                raise ValueError(f"Unknown constraint direction: {direction}")
        else:
            print(f"Constraint key '{constraint_key}' not found in strategy metrics")

    # Apply the goal if constraints are met
    if constraints_met:
        if goal['optimize'].lower() == 'maximize':
            return (metric_value, constraints_met)  # Return both values
        elif goal['optimize'].lower() == 'minimize':
            return (-metric_value, constraints_met)  # Return both values
        else:
            raise ValueError(f"Unknown optimization goal: {goal['optimize']}")
    else:
        # Apply penalty if constraints are not met
        if goal['optimize'].lower() == 'maximize':
            return (float('-inf'), constraints_met)  # Example penalty; ensures this result is less favorable
        elif goal['optimize'].lower() == 'minimize':
            return (float('inf'), constraints_met)   # Example penalty; ensures this result is less favorable
        else:
            raise ValueError(f"Unknown optimization goal: {goal['optimize']}")



    

start_date = pd.to_datetime(start_date, format='%d-%m-%Y')
end_date = pd.to_datetime(end_date, format='%d-%m-%Y')

# Define the strategy evaluation function (Your execute_and_analyze_strategy_2 function)
def evaluate_strategy(individual):
    combined_metrics = {}
    param_index = 0

    # Prepare conditions with optimized parameters
    for cond in conditions:
        for indicator_key in ['indicatorOne', 'indicatorTwo']:
            indicator = cond.get(indicator_key, {})
            if 'indiInputs' in indicator:
                for param_name, param_range in indicator['indiInputs'].items():
                    if param_index < len(individual):
                        indicator['indiInputs'][param_name] = round(individual[param_index])
                        param_index += 1
                indicator['value'] = indicator.get('value')
                indicator['displayValue'] = indicator.get('displayValue')

    for symbol in symbols:
        file_path = os.path.join(r'D:\stock_historical_data\historical_data_' + time_period, f'{symbol}.csv')
        sbi_data = pd.read_csv(file_path)
        sbi_data['date'] = pd.to_datetime(sbi_data['date'], utc=False)
        data_timezone = sbi_data['date'].dt.tz

        global start_date, end_date
        start_date = start_date.tz_localize(data_timezone)
        end_date = end_date.tz_localize(data_timezone)

        sbi_data = sbi_data[(sbi_data['date'] >= start_date) & (sbi_data['date'] <= end_date)]
        start_date = start_date.tz_localize(None)
        end_date = end_date.tz_localize(None)

        sbi_data.set_index('date', inplace=True)

        # Generate signals based on conditions
        if graph_type == 'heikin-ashi':
            data = compute_heikin_ashi(sbi_data)
            buy_signals, sell_signals = combine_signals(data, conditions, operator, exit_conditions, exit_operator)
            entry_signals_short, exit_signals_short = combine_signals(data, conditions2, operator2, exit_conditions2, exit_operator2)
        else:
            buy_signals, sell_signals = combine_signals(sbi_data, conditions, operator, exit_conditions, exit_operator)
            entry_signals_short, exit_signals_short = combine_signals(sbi_data, conditions2, operator2, exit_conditions2, exit_operator2)
        
        sbi_data['Buy_Signal'] = buy_signals
        sbi_data['Sell_Signal'] = sell_signals
        sbi_data['entry_signals_short'] = entry_signals_short
        sbi_data['exit_signals_short'] = exit_signals_short

        # Execute strategy analysis for each symbol
        strategy_metrics = execute_and_analyze_strategy(
            symbol, sbi_data, stopLoss, moveInstrument, moveSl, target,
            initial_capital, quantity, trade_type, position_size_type,
            max_size_amount, max_quantity, exit_conditions, exit_conditions2
        )

        # Aggregate metrics
        for key, value in strategy_metrics.items():
            if isinstance(value, (int, float)):
                if key not in combined_metrics:
                    combined_metrics[key] = []
                combined_metrics[key].append(value)

    # Calculate averages or other aggregate measures
    overall_metrics = {
        key: sum(values) / len(values)
        for key, values in combined_metrics.items()
    }

    # Apply constraints and goal to the overall metrics
    adjusted_metric, constraints_met = apply_constraints(overall_metrics, constraints, goal)
    constraint_condition = constraints_met

    return (adjusted_metric,)  # Return as a tuple for DEAP compatibility




        
    # total_score = 0
    # for metrics in results:
    #     # Apply constraints to the metrics
    #     metrics_score = apply_constraints(metrics, constraints, goal)
        
    #     # Update total score based on the goal
    #     if goal['optimize'].lower() == 'maximize':
    #         total_score += metrics_score
    #     elif goal['optimize'].lower() == 'minimize':
    #         total_score -= metrics_score
    #     else:
    #         raise ValueError(f"Unknown optimization goal: {goal['optimize']}")
    
    # return (total_score,)


    # Define constraints (e.g., win rate should be above 50%)
    avg_win_rate = np.mean([metrics['Win Rate (%)'] for metrics in results])
    avg_max_drawdown = np.mean([metrics['Max Drawdown'] for metrics in results])
    
    # if avg_win_rate >= 50:     
    #     return total_pnl,
    # else:
    #     return -np.inf,  # Penalize solutions that do not meet the constraints


# Genetic Algorithm setup
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMax)

# toolbox = base.Toolbox()
# toolbox.register("attr_int", random.randint, 1, 100)  # Example range for RSI period
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# toolbox.register("evaluate", evaluate_strategy)
# toolbox.register("mate", tools.cxBlend, alpha=0.5)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
# toolbox.register("select", tools.selTournament, tournsize=3)



# RSI_PERIOD_MIN = int(variable_inputs[0]['indiInputs']['period']['from'])
# RSI_PERIOD_MAX = int(variable_inputs[0]['indiInputs']['period']['to'])

# Custom initialization function
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Custom initialization function
def init_param(param_range):
    return random.randint(int(param_range['from']), int(param_range['to']))

def create_individual(indicator_params):
    individual = []
    for condition in indicator_params:
        for indicator_key in ['indicatorOne', 'indicatorTwo']:
            indicator = condition.get(indicator_key, {})
            if 'indiInputs' in indicator:
                for param_name, param_range in indicator['indiInputs'].items():
                    individual.append(init_param(param_range))
    return individual

def create_toolbox(indicator_params):
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: create_individual(indicator_params))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_strategy)
    toolbox.register("mate", tools.cxTwoPoint)  # Use cxTwoPoint instead of cxBlend
    toolbox.register("mutate", mutate_params, indicator_params=indicator_params, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

def mutate_params(individual, indicator_params, indpb):
    param_index = 0
    for condition in indicator_params:
        for indicator_key in ['indicatorOne', 'indicatorTwo']:
            indicator = condition.get(indicator_key, {})
            if 'indiInputs' in indicator:
                for param_name, param_range in indicator['indiInputs'].items():
                    if random.random() < indpb:
                        individual[param_index] = init_param(param_range)
                    else:
                        # Ensure the mutated parameter stays within its range
                        individual[param_index] = max(int(param_range['from']), 
                                                      min(individual[param_index], int(param_range['to'])))
                    param_index += 1
    return individual,

def optimize_strategy():
    toolbox = create_toolbox(variable_inputs)
    random.seed(42) 
    population = toolbox.population(n=10)
    ngen = 10
    cxpb = 0.5
    mutpb = 0.2

    for gen in range(ngen):
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
        fits = toolbox.map(toolbox.evaluate, offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))

    best_ind = tools.selBest(population, k=1)[0]
    return best_ind

best_parameters = optimize_strategy()


def convert_timestamp(obj):
    if isinstance(obj, Timestamp):
        return obj.isoformat()  # Convert to ISO 8601 string format
    raise TypeError("Type not serializable")



def backtest_with_best_parameters(best_parameters):
    results = []
    param_index = 0
    
   

    for cond in conditions:
        for indicator_key in ['indicatorOne', 'indicatorTwo']:
            indicator = cond.get(indicator_key, {})
            
            if 'indiInputs' in indicator:
                for param_name in indicator['indiInputs']:
                    # Round the parameter value to the nearest integer
                    indicator['indiInputs'][param_name] = round(best_parameters[param_index],0)
                    param_index += 1

   

    for symbol in symbols:
        file_path = os.path.join(r'D:\stock_historical_data\historical_data_' + time_period, f'{symbol}.csv')
        sbi_data = pd.read_csv(file_path)
        sbi_data['date'] = pd.to_datetime(sbi_data['date'], utc=False)  
        data_timezone = sbi_data['date'].dt.tz
    
        global start_date, end_date
        start_date = start_date.tz_localize(data_timezone)
        end_date = end_date.tz_localize(data_timezone)

        sbi_data = sbi_data[(sbi_data['date'] >= start_date) & (sbi_data['date'] <= end_date)]

        start_date = start_date.tz_localize(None)
        end_date = end_date.tz_localize(None)

        close_prices = sbi_data[['date', 'close']].reset_index().to_json(orient='records')
        sbi_data.set_index('date', inplace=True)

        if graph_type == 'heikin-ashi':
            data = compute_heikin_ashi(sbi_data)
            buy_signals, sell_signals = combine_signals(data, conditions, operator, exit_conditions, exit_operator)
            entry_signals_short, exit_signals_short = combine_signals(data, conditions2, operator2, exit_conditions2, exit_operator2)
            sbi_data['Buy_Signal'] = buy_signals
            sbi_data['Sell_Signal'] = sell_signals
            sbi_data['entry_signals_short'] = entry_signals_short
            sbi_data['exit_signals_short'] = exit_signals_short
        else:
            buy_signals, sell_signals = combine_signals(sbi_data, conditions, operator, exit_conditions, exit_operator)
            entry_signals_short, exit_signals_short = combine_signals(sbi_data, conditions2, operator2, exit_conditions2, exit_operator2)
            sbi_data['Buy_Signal'] = buy_signals
            sbi_data['Sell_Signal'] = sell_signals
            sbi_data['entry_signals_short'] = entry_signals_short
            sbi_data['exit_signals_short'] = exit_signals_short

        strategy_metrics = execute_and_analyze_strategy(
            symbol, sbi_data, stopLoss, moveInstrument, moveSl, target,
            initial_capital, quantity, trade_type, position_size_type,
            max_size_amount, max_quantity, exit_conditions, exit_conditions2
        )
        results.append(strategy_metrics)
    
    return results

# Run backtest
results = backtest_with_best_parameters(best_parameters)
combined_data = {
    'best_parameters': best_parameters,
    'results': results,
    'constraint':constraint_condition
}

print(json.dumps(combined_data, default=convert_timestamp))



