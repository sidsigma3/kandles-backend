import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from nsepython import *
import base64
from io import BytesIO
from json import JSONEncoder
from pandas import Timestamp
import os
import datetime as dt

buf = BytesIO()
input_data = json.loads(sys.argv[1])


#inputs from frontend

stopLoss=input_data.get('slPct')
target=input_data.get('targetPct')
symbol = input_data.get('backSymbol')
start_date = input_data.get('startDate')
end_date = input_data.get('endDate')
initial_capital = input_data.get('backCapital')
quantity=input_data.get('backQuantity')
strategy_type = input_data.get('entryType')
graph_type = input_data.get('graphType')
trailing_sl=input_data.get('trailPct')

position_size_type = input_data.get('positionSizeType')
max_quantity = input_data.get('maxQuantity')
max_size_amount = input_data.get('sizeAmount')

moveSl = input_data.get('moveSlPct')
moveInstrument = input_data.get('moveInstrumentPct')
time_period = input_data.get('timePeriod')
trade_type = input_data.get('marketType')

conditions = input_data.get('strategyDetails')['conditions']
operator = input_data.get('strategyDetails')['logicalOperators']

exit_conditions = input_data.get('strategyDetailsExit')['conditions']
exit_operator = input_data.get('strategyDetailsExit')['logicalOperators']

conditions2 = input_data.get('strategyDetails2')['conditions']
operator2 = input_data.get('strategyDetails2')['logicalOperators']

exit_conditions2 = input_data.get('strategyDetailsExit2')['conditions']
exit_operator2 = input_data.get('strategyDetailsExit2')['logicalOperators']

max_long_entry = input_data.get('maxLong', 1)
max_short_entry = input_data.get('maxShort', 1)

selected_days = input_data.get('selectedDaysForIndi')

# stopLoss = 0.01
# target = 0.02

# start_date= '01-01-2021'
# end_date = '01-01-2022'
# initial_capital = 100000000
# quantity = 50

# symbol = [
#   "HDFCBANK",
#   "SBIN"
# ]
# conditions = [
#     {
#         "indicatorOne": {
#             "value": "macd",
#             "displayValue": "MACD",
#             "indiInputs": {
#                 "fastPeriod": 18,
#                 "slowPeriod": 7,
#                 "signalPeriod": 11
#             }
#         },
#         "comparator": "crosses-above",
#         "indicatorTwo": {
#             "value": "macd",
#             "displayValue": "MACD",
#             "indiInputs": {
#                 "fastPeriod": 33,
#                 "slowPeriod": 32,
#                 "signalPeriod": 30
#             }
#         }
#     },
#     {
#         "indicatorOne": {
#             "value": "rsi",
#             "displayValue": "RSI",
#             "indiInputs": {
#                 "period": 4
#             }
#         },
#         "comparator": "higher-than",
#         "indicatorTwo": {
#             "value": "number",
#             "displayValue": "Number",
#             "indiInputs": {
#                 "number": 43
#             }
#         }
#     },
#     {
#         "indicatorOne": {
#             "value": "rsi",
#             "displayValue": "RSI",
#             "indiInputs": {
#                 "period": 8
#             }
#         },
#         "comparator": "crosses-above",
#         "indicatorTwo": {
#             "value": "rsi",
#             "displayValue": "RSI",
#             "indiInputs": {
#                 "period": 25
#             }
#         }
#     }
# ]

# # conditions=[]
# operator = [
#     "AND",
#     "AND"
# ]
# conditions2=[]
# operator2=[]


# conditions2 = [{
#   "indicatorOne": {
#     "value": "rsi",
#     "displayValue": "RSI",
#     "indiInputs": {
#       "period": 14
#     }
#   },
#   "comparator": "crosses-below",
#   "indicatorTwo": {
#     "value": "rsi",
#     "displayValue": "RSI",
#     "indiInputs": {
#       "period": 23
#     }
#   }
# }]

# operator2 = []

# exit_conditions = [
#         {
#             "indicatorOne": {
#                 "value": "ema",
#                 "displayValue": "EMA",
#                 "indiInputs": {
#                     "period": 14
#                 }
#             },
#             "comparator": "crosses-below",
#             "indicatorTwo": {
#                 "value": "sma",
#                 "displayValue": "SMA",
#                 "indiInputs": {
#                     "period": 14
#                 }
#             }
#         }
#     ]

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



#column name mapping for indicators having multiple output indicators

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



#mapping for indicators required parameters
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



#mappinng of name of parameters from frontend and valid parmeters name as per ta lib
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

# Assuming sbi_data is your DataFrame with stock data

# Calculate indicators as per the conditions
# indicator_one_period = int(input_data.get('indicatorDetails')['indicatorOne']['period'])
# indicator_two_period = int(input_data.get('indicatorDetails')['indicatorTwo']['period'])

# input_conditions = input_data.get('indicatorDetails')['comparator']

# indicator_one_details = input_data.get('indicatorDetails')['indicatorOne']
# indicator_two_details = input_data.get('indicatorDetails')['indicatorTwo']



trades = []  # Initialize a list to collect trades
total_pnl = 0  # Initialize total P&L

# symbol = "SBIN"
series = "EQ"
# start_date = "08-06-2021"
# end_date ="14-06-2022"




def suppress_print(func, *args, **kwargs):
    original_stdout = sys.stdout  # Save the original stdout
    sys.stdout = open(os.devnull, 'w')  # Redirect stdout to devnull
    result = func(*args, **kwargs)  # Call the function
    sys.stdout.close()
    sys.stdout = original_stdout  # Restore original stdout
   
    return result



#heikin-ashi graph conversion
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




# monthly pnl calculation

def compute_monthly_pnl(trades,invested):
    # Create DataFrame from trade dates and cumulative P&L
    # data = pd.DataFrame({
    #     'Trade Date': pd.to_datetime(trade_dates),
    #     'Cumulative PnL': cumulative_pnl
    # })
    # data.set_index('Trade Date', inplace=True)

    # # Compute monthly P&L by taking the difference of cumulative PnL
    # monthly_pnl = data['Cumulative PnL'].resample('M').last().diff().fillna(0)

    # # Reshape data into a pivot table format with years as rows and months as columns
    # monthly_pnl_df = monthly_pnl.reset_index()
    # monthly_pnl_df['Year'] = monthly_pnl_df['Trade Date'].dt.year
    # monthly_pnl_df['Month'] = monthly_pnl_df['Trade Date'].dt.strftime('%B')

    # # Pivot and reindex the table to have months in chronological order
    # pivot_table = monthly_pnl_df.pivot_table(values='Cumulative PnL', index='Year', columns='Month', aggfunc='sum', fill_value=0)
    # months_ordered = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    # pivot_table = pivot_table.reindex(columns=months_ordered)  # This ensures months are in the right order

    # # Add a total row at the end
    # pivot_table.loc['Total'] = pivot_table.sum()

    # return pivot_table
    trades_df = pd.DataFrame(trades)

    # Filter out trades without PnL
    trades_with_pnl = trades_df[~trades_df['pnl'].isnull()]
    
    # Convert trade dates to datetime
    trade_dates = pd.to_datetime(trades_with_pnl['date'])
    trades_with_pnl['date'] = trade_dates

    # Create a DataFrame for PnL data
    pnl_data = trades_with_pnl[['date', 'pnl', 'invested']].rename(columns={'date': 'Trade Date', 'pnl': 'PnL', 'invested': 'Invested'})

    # Resample monthly PnL
    monthly_pnl = pnl_data.set_index('Trade Date').resample('M').sum().round(0)

   

    # Create pivot table for monthly PnL
    pivot_table = monthly_pnl.reset_index().pivot_table(values='PnL', index=monthly_pnl.index.year, columns=monthly_pnl.index.month_name(), aggfunc='sum', fill_value=0)

    

    # Calculate total PnL per year
    pivot_table.loc['Total'] = pivot_table.sum().round(0)
    pivot_table['Total Yearly PnL'] = pivot_table.sum(axis=1).round(0)

    # Reorder months
    months_ordered = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    pivot_table = pivot_table.reindex(columns=months_ordered + ['Total Yearly PnL'])
    
    # Calculate yearly ROI
    yearly_pnl = pnl_data.set_index('Trade Date').resample('Y').sum().round(0)

    
    
    # Calculate yearly win percentage
    yearly_trade_counts = trades_with_pnl.set_index('date').resample('Y').size()

   

    yearly_roi = round((yearly_pnl['PnL'] /(invested)) * 100,2)
    yearly_win_counts = trades_with_pnl[trades_with_pnl['pnl'] > 0].set_index('date').resample('Y').size()
    yearly_win_percentage = round((yearly_win_counts / yearly_trade_counts) * 100 ,2)




    

    # Add yearly ROI and Win Percentage to the pivot table
    for year in yearly_roi.index.year:
        pivot_table.loc[year, 'Yearly ROI (%)'] = yearly_roi.loc[f'{year}-12-31']
        pivot_table.loc[year, 'Yearly Win %'] = yearly_win_percentage.loc[f'{year}-12-31']

    # Add totals for ROI and Win Percentage
    pivot_table.loc['Total', 'Yearly ROI (%)'] = yearly_roi.sum()
    pivot_table.loc['Total', 'Yearly Win %'] = yearly_win_percentage.mean()

    return pivot_table



def compute_combined_pnl(long_trades_df, short_trades_df):
    # Compute individual monthly PnL for long and short trades
    # long_monthly_pnl = compute_monthly_pnl(long_trades_df,)
    # short_monthly_pnl = compute_monthly_pnl(short_trades_df,)

  
    # Combine the data
    combined_pnl = long_trades_df.add(short_trades_df, fill_value=0)
    
    combined_pnl['Yearly Win %'] = round((long_trades_df['Yearly Win %'] + short_trades_df['Yearly Win %']) / 2,2)
    
    
    # Recalculate total yearly PnL
    # combined_pnl['Total Yearly PnL'] = combined_pnl.drop(columns=['Yearly ROI (%)', 'Yearly Win %']).sum(axis=1).round(0)
    # combined_pnl.loc['Total'] = combined_pnl.drop(index='Total').sum().round(0)

    # Recalculate yearly ROI and Win Percentage
    yearly_roi = (combined_pnl['Yearly ROI (%)'].drop('Total') * (combined_pnl['Total Yearly PnL'].drop('Total') / combined_pnl['Total Yearly PnL'].drop('Total').sum())).sum()
    # yearly_win_percentage = (combined_pnl['Yearly Win %'].drop('Total') * (combined_pnl['Total Yearly PnL'].drop('Total') / combined_pnl['Total Yearly PnL'].drop('Total').sum())).sum()

    # Add totals for ROI and Win Percentage
    # combined_pnl.loc['Total', 'Yearly ROI (%)'] = yearly_roi
    # combined_pnl.loc['Total', 'Yearly Win %'] = round(yearly_win_percentage,2)
    


    return combined_pnl



#indicator calculation function

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



def add_signals(dataframe, indicator_one, indicator_two, comparator):
    """Add buy and sell signals to the dataframe based on the comparator logic."""
    if comparator == "crosses-below":
        dataframe['Buy_Signal'] = (indicator_one < indicator_two) & (indicator_one.shift(1) >= indicator_two.shift(1))
        dataframe['Sell_Signal'] = (indicator_one > indicator_two) & (indicator_one.shift(1) <= indicator_two.shift(1))
    # Add additional logic for other comparators...
        
def is_within_trading_hours(current_date,start_time='09:15', end_time='15:15'):
    if time_period == 'daily':
        return True
    
    current_time = current_date.time()
    start_time = dt.datetime.strptime(start_time, '%H:%M').time()
    end_time = dt.datetime.strptime(end_time, '%H:%M').time()
    return start_time <= current_time < end_time







# funtion for evalution of indicators values to get true and false signal using comaprator 

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

# Function to combine signals from multiple conditions
# def combine_signals(data, conditions, logical_operators ,exit_condition, exit_operator ):
#     if len(conditions) == 1:
#         # If there is only one condition, just evaluate it and return its signals
#         buy_signal, sell_signal = evaluate_condition(data, conditions[0] , exit_condition[0])
#         return buy_signal, sell_signal
    
#     elif len(conditions) > 1:
#         # Combine signals when there are multiple conditions
#         final_buy_signal, final_sell_signal = evaluate_condition(data, conditions[0] ,exit_condition[0])
#         for i in range(1, len(conditions)):
#             next_buy_signal, next_sell_signal = evaluate_condition(data, conditions[i] , exit_condition)
#             if logical_operators and i-1 < len(logical_operators):
#                 if logical_operators[i-1].upper() == 'AND':
#                     final_buy_signal &= next_buy_signal
#                     final_sell_signal &= next_sell_signal
#                 elif logical_operators[i-1].upper() == 'OR':
#                     final_buy_signal |= next_buy_signal
#                     final_sell_signal |= next_sell_signal
#         return final_buy_signal, final_sell_signal
#     else:
#         # If no conditions are provided
#         return pd.Series([False] * len(data)), pd.Series([False] * len(data))



# Function to combine signals from multiple conditions

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







# sbi_data = suppress_print(equity_history, symbol, series, start_date, end_date)


# sbi_data['mTIMESTAMP'] = pd.to_datetime(sbi_data['mTIMESTAMP'], format='%d-%b-%Y')

# sbi_data.sort_values(by='mTIMESTAMP', inplace=True)
# sbi_data.reset_index(drop=True, inplace=True)

# sbi_data['RSI'] = ta.momentum.rsi(sbi_data['CH_CLOSING_PRICE'], window=14)


# sbi_data['Indicator_One'] = calculate_indicator(sbi_data['CH_CLOSING_PRICE'], indicator_one_details)
# sbi_data['Indicator_Two'] = calculate_indicator(sbi_data['CH_CLOSING_PRICE'], indicator_two_details)


# buy_signals, sell_signals = combine_signals(sbi_data, conditions, operator)

# sbi_data['Buy_Signal'] = buy_signals
# sbi_data['Sell_Signal'] = sell_signals







# comparator = input_conditions

# if comparator == "crosses-below":
#     sbi_data['Buy_Signal'] = (sbi_data['Indicator_One'] < sbi_data['Indicator_Two']) & (sbi_data['Indicator_One'].shift(1) >= sbi_data['Indicator_Two'].shift(1))
#     sbi_data['Sell_Signal'] = (sbi_data['Indicator_One'] > sbi_data['Indicator_Two']) & (sbi_data['Indicator_One'].shift(1) <= sbi_data['Indicator_Two'].shift(1))

# elif comparator == "crosses-above":
#     sbi_data['Buy_Signal'] = (sbi_data['Indicator_One'] > sbi_data['Indicator_Two']) & (sbi_data['Indicator_One'].shift(1) <= sbi_data['Indicator_Two'].shift(1))
#     sbi_data['Sell_Signal'] = (sbi_data['Indicator_One'] < sbi_data['Indicator_Two']) & (sbi_data['Indicator_One'].shift(1) >= sbi_data['Indicator_Two'].shift(1))

# elif comparator == "equal-to":
#     tolerance = 0.01  # Define your tolerance level
#     sbi_data['Buy_Signal'] = abs(sbi_data['Indicator_One'] - sbi_data['Indicator_Two']) <= tolerance
#     sbi_data['Sell_Signal'] = abs(sbi_data['Indicator_One'] - sbi_data['Indicator_Two']) > tolerance

# elif comparator == "lower-than":
#     sbi_data['Buy_Signal'] = sbi_data['Indicator_One'] < sbi_data['Indicator_Two']
#     sbi_data['Sell_Signal'] = sbi_data['Indicator_One'] >= sbi_data['Indicator_Two']

# elif comparator == "higher-than":
#     sbi_data['Buy_Signal'] = sbi_data['Indicator_One'] > sbi_data['Indicator_Two']
#     sbi_data['Sell_Signal'] = sbi_data['Indicator_One'] <= sbi_data['Indicator_Two']

# Generate buy signals (RSI crosses above RSI_BUY from below)
# sbi_data['Buy'] = (sbi_data['RSI'] > RSI_BUY) & (sbi_data['RSI'].shift(1) <= RSI_BUY)

# Generate sell signals (RSI crosses below RSI_SELL from above)
# sbi_data['Sell'] = (sbi_data['RSI'] < RSI_SELL) & (sbi_data['RSI'].shift(1) >= RSI_SELL)





# Example usage
# strategy_metrics = execute_and_analyze_strategy(sbi_data, stopLoss, target,initial_capital,quantity)




# def execute_and_analyze_strategy_2(data, strategy_type, stop_loss_pct, trailing_stop_loss_pct, target_pct, initial_funds, quantity, position_size_type=None, max_size_amount=None, max_quantity=None, exit_conditions=None):
#     is_in_position = False
#     entry_price = 0
#     funds = initial_funds
#     trades = []
#     gains = []
#     losses = []
#     total_pnl = 0
#     total_brokerage = 0
#     win_streak = 0
#     loss_streak = 0
#     current_streak = 0
#     wins = 0
#     losses_count = 0
#     last_outcome = None
#     peak_funds = initial_funds
#     max_drawdown = 0
#     max_drawdown_days = 0
#     drawdown_start = None
#     trade_dates = [data.index[0]]
#     cumulative_pnl = [0]
#     trailing_stop_loss_price = 0
#     invested_fund = 0
#     tsl_count = 0
#     target_count = 0
#     stoploss_count = 0

#     for i in range(len(data)):
#         current_date = data.index[i]
#         high_price = data['CH_TRADE_HIGH_PRICE'].iloc[i]
#         low_price = data['CH_TRADE_LOW_PRICE'].iloc[i]
#         close_price = data['CH_CLOSING_PRICE'].iloc[i]
#         open_price = data['CH_OPENING_PRICE'].iloc[i]
#         brokerage_per_trade = min(close_price * quantity * 0.0003, 20)

#         # Determine the day type
#         if close_price > open_price * 1.01:
#             day_type = 'bullish'
#         elif close_price < open_price * 0.09:
#             day_type = 'bearish'
#         else:
#             day_type = 'sideways'

#         if is_in_position:
#             exit_price = None
#             action = None
#             trail_price = None

#             if exit_conditions:
#                 exit_signal = data['Sell_Signal'].iloc[i]
#                 if exit_signal:
#                     exit_price = close_price
#                     action = 'sell_signal'

#             if not exit_price:
#                 if trailing_stop_loss_pct != 0:
#                     if strategy_type == 'buy':
#                         trailing_stop_loss_price = max(trailing_stop_loss_price, high_price * (1 - trailing_stop_loss_pct))
#                         trail_price = trailing_stop_loss_price
#                         target_reached_price = entry_price * (1 + target_pct)
#                         if low_price <= trailing_stop_loss_price:
#                             action = 'tsl'
#                             tsl_count += 1
#                             exit_price = trailing_stop_loss_price
#                         elif high_price >= target_reached_price:
#                             action = 'target'
#                             target_count += 1
#                             exit_price = target_reached_price
#                     else:  # sell strategy
#                         trail_price = trailing_stop_loss_price
#                         trailing_stop_loss_price = min(trailing_stop_loss_price, low_price * (1 + trailing_stop_loss_pct))
#                         target_reached_price = entry_price * (1 - target_pct)
#                         if high_price >= trailing_stop_loss_price:
#                             action = 'tsl'
#                             tsl_count += 1
#                             exit_price = trailing_stop_loss_price
#                         elif low_price <= target_reached_price:
#                             action = 'target'
#                             target_count += 1
#                             exit_price = target_reached_price
#                 else:
#                     if strategy_type == 'buy':
#                         stop_loss_price = entry_price * (1 - stop_loss_pct)
#                         target_reached_price = entry_price * (1 + target_pct)
#                         if low_price <= stop_loss_price:
#                             action = 'stoploss'
#                             stoploss_count += 1
#                             exit_price = stop_loss_price
#                         elif high_price >= target_reached_price:
#                             action = 'target'
#                             target_count += 1
#                             exit_price = target_reached_price
#                     else:  # sell strategy
#                         stop_loss_price = entry_price * (1 + stop_loss_pct)
#                         target_reached_price = entry_price * (1 - target_pct)
#                         if high_price >= stop_loss_price:
#                             action = 'stoploss'
#                             stoploss_count += 1
#                             exit_price = stop_loss_price
#                         elif low_price <= target_reached_price:
#                             action = 'target'
#                             target_count += 1
#                             exit_price = target_reached_price

#             if exit_price:
#                 pnl = (exit_price - entry_price) * quantity - brokerage_per_trade if strategy_type == 'buy' else (entry_price - exit_price) * quantity - brokerage_per_trade
#                 funds += (exit_price * quantity - brokerage_per_trade) if strategy_type == 'buy' else (entry_price * quantity - exit_price * quantity - brokerage_per_trade)
#                 total_pnl += pnl
#                 total_brokerage += brokerage_per_trade
#                 target_price = entry_price
#                 trailing_stop_loss_price = 0
#                 targetPrice = (entry_price * target_pct) + entry_price
#                 stopLossPrice = entry_price - (entry_price * stop_loss_pct)

#                 invested_price = entry_price * quantity

#                 trades.append({
#                     'symbol': symbol,
#                     'action': 'Sell' if strategy_type == 'buy' else 'Buy',
#                     'price': exit_price,
#                     'pnl': pnl,
#                     'quantity': quantity,
#                     'brokerage': brokerage_per_trade,
#                     'targetPrice': targetPrice,
#                     'stopLossprice': stopLossPrice,
#                     'trailingSl': trail_price,
#                     'date': current_date,
#                     'invested': invested_price,
#                     'trigger': action,
#                     'day_type': day_type
#                 })
#                 is_in_position = False
#                 trade_dates.append(current_date)
#                 cumulative_pnl.append(cumulative_pnl[-1] + pnl)

#                 # Track win or loss
#                 if pnl > 0:
#                     wins += 1
#                     gains.append(pnl)
#                     if last_outcome == 'win':
#                         current_streak += 1
#                     else:
#                         current_streak = 1
#                     win_streak = max(win_streak, current_streak)
#                     last_outcome = 'win'
#                 else:
#                     losses_count += 1
#                     losses.append(-pnl)
#                     if last_outcome == 'loss':
#                         current_streak += 1
#                     else:
#                         current_streak = 1
#                     loss_streak = max(loss_streak, current_streak)
#                     last_outcome = 'loss'

#                 if funds > peak_funds:
#                     peak_funds = funds
#                     drawdown_start = None
#                 else:
#                     if drawdown_start is None:
#                         drawdown_start = current_date
#                     current_drawdown = peak_funds - funds
#                     max_drawdown = max(max_drawdown, current_drawdown)
#                     if drawdown_start is not None:
#                         drawdown_days = (current_date - drawdown_start).days
#                         max_drawdown_days = max(max_drawdown_days, drawdown_days)

#         entry_signal = data['Buy_Signal'].iloc[i] if strategy_type == 'buy' else data['Sell_Signal'].iloc[i]

#         if entry_signal and not is_in_position:
#             entry_price = close_price

#             # Adjust quantity based on position sizing type if provided
#             if position_size_type:
#                 if position_size_type == 'capital':
#                     quantity = min(max_size_amount // entry_price, max_quantity)
#                 elif position_size_type == 'risk':
#                     stop_loss_price = entry_price * (1 - stop_loss_pct)
#                     risk_per_share = entry_price - stop_loss_price
#                     quantity = min(max_size_amount // risk_per_share, max_quantity)

#             target_price = (entry_price * target_pct) + entry_price
#             stop_loss_price = entry_price - (entry_price * stop_loss_pct)
#             if trailing_stop_loss_pct != 0:
#                 trailing_sl_price = stop_loss_price
#             else:
#                 trailing_sl_price = 0
#             brokerage_per_trade = min(entry_price * quantity * 0.0003, 20)
#             funds -= (entry_price * quantity + brokerage_per_trade) if strategy_type == 'buy' else (entry_price * quantity - brokerage_per_trade)
#             is_in_position = True
#             trailing_stop_loss_price = entry_price * (1 - trailing_stop_loss_pct) if strategy_type == 'buy' else entry_price * (1 + trailing_stop_loss_pct)
#             invested_price = entry_price * quantity
#             invested_fund += invested_price
#             trades.append({
#                 'symbol': symbol,
#                 'action': 'Buy' if strategy_type == 'buy' else 'Sell',
#                 'price': entry_price,
#                 'quantity': quantity,
#                 'trailingSlPrice': trailing_stop_loss_price,
#                 'brokerage': brokerage_per_trade,
#                 'date': current_date,
#                 'targetPrice': target_price,
#                 'stopLossprice': stop_loss_price,
#                 'trailingSl': trailing_sl_price,
#                 'invested': invested_price,
#                 'trigger': 'Entry',
#                 'day_type': day_type
#             })
#             trade_dates.append(current_date)
#             cumulative_pnl.append(cumulative_pnl[-1])
#             total_brokerage += brokerage_per_trade

#     # Final metrics calculation
#     total_trades = wins + losses_count
#     avg_gain = sum(gains) / wins if wins else 0
#     avg_loss = sum(losses) / losses_count if losses_count else 0
#     win_rate = (wins / total_trades * 100) if total_trades else 0
#     loss_rate = (losses_count / total_trades * 100) if total_trades else 0
#     reward_to_risk = avg_gain / avg_loss if avg_loss != 0 else 0
#     expectancy = (reward_to_risk * win_rate / 100) - (loss_rate / 100) if total_trades else 0

#     return {
#         'trades': trades,
#         'Total Signals': total_trades,
#         'Number of Wins': wins,
#         'Number of Losses': losses_count,
#         'Winning Streak': win_streak,
#         'Losing Streak': loss_streak,
#         'Max Gains': max(gains, default=0),
#         'Max Loss': max(losses, default=0),
#         'Avg Gain per Winning Trade': avg_gain,
#         'Avg Loss per Losing Trade': avg_loss,
#         'Max Drawdown': max_drawdown,
#         'Max Drawdown Days': max_drawdown_days,
#         'Win Rate (%)': win_rate,
#         'Loss Rate (%)': loss_rate,
#         'Expectancy': expectancy,
#         'Profit Factor': reward_to_risk,
#         'Total PnL': total_pnl,
#         'Total Brokerage': total_brokerage,
#         'Net PnL After Brokerage': total_pnl - total_brokerage,
#         'Remaining Funds': funds,
#         'Trade Dates': trade_dates,
#         'targetCount': target_count,
#         'tslCount': tsl_count,
#         'slCount': stoploss_count,
#         'investedFund': invested_fund,
#         'Cumulative PnL': cumulative_pnl
#     }

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

# Iterate over each symbol
for instrument in symbol:
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




#function for backtesting 

def execute_and_analyze_strategy(data, strategy_type, stop_loss_pct, price_move_pct, trailing_stop_loss_pct, target_pct, initial_funds, quantity, trade_type,position_size_type=None, max_size_amount=None, max_quantity=None, exit_conditions=None,exit_conditions_short=None):
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
    peak_funds = 0
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

    reward_to_risk_avg = (avg_gain + avg_gain_short) / (avg_loss + avg_loss_short) 
    win_rate_avg = (wins + wins_short) / (total_trades + total_trades_short) * 100
    loss_rate_avg = (losses_count + losses_count_short) / (total_trades + total_trades_short) * 100
    
    expectancy_total = ((reward_to_risk_avg * (win_rate_avg / 100))) -(loss_rate_avg/100)  if total_trades_short==0 or total_trades==0 else 0


    
    if invested_fund == 0:
        invested_fund_total = invested_fund_short/total_trades_short 
    elif invested_fund_short == 0 :
        invested_fund_total = invested_fund/total_trades 

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


# def execute_and_analyze_strategy_2(data, strategy_type, stop_loss_pct, price_move_pct, trailing_stop_loss_pct, target_pct, initial_funds, quantity, trade_type,position_size_type=None, max_size_amount=None, max_quantity=None, exit_conditions=None):
#     is_in_position = False
#     entry_price = 0
#     funds = initial_funds
#     trades = []
#     gains = []
#     losses = []
#     total_pnl = 0
#     total_brokerage = 0
#     win_streak = 0
#     loss_streak = 0
#     current_streak = 0
#     wins = 0
#     losses_count = 0
#     last_outcome = None
#     peak_funds = initial_funds
#     max_drawdown = 0
#     max_drawdown_days = 0
#     drawdown_start = None
#     trade_dates = [data.index[0]]
#     cumulative_pnl = [0]
#     trailing_stop_loss_price = 0
#     invested_fund = 0
#     tsl_count = 0
#     target_count = 0
#     stoploss_count = 0
#     entry_price_reference = 0
#     sell_signal_count = 0
#     market_close = 0


#     for i in range(len(data)):
#         current_date = data.index[i]
#         high_price = data['high'].iloc[i]
#         low_price = data['low'].iloc[i]
#         close_price = data['close'].iloc[i]
#         open_price = data['open'].iloc[i]
#         brokerage_per_trade = min(close_price * quantity * 0.0003, 20)

#         # Determine the day type
#         # if close_price > open_price * 1.01:
#         #     day_type = 'bullish'
#         # elif close_price < open_price * 0.99:
#         #     day_type = 'bearish'
#         # else:
#         #     day_type = 'sideways'
        
#         day_type = market_sentiment_data[symbol].get(current_date.date(), 'sideways')

#         if i < len(data) - 1 and current_date.date() != data.index[i+1].date():
#             is_last_candle_of_day = True

#         else:
#             is_last_candle_of_day = False

#         if is_in_position:
#             exit_price = None
#             action = None

#             if exit_conditions:
#                 if strategy_type == 'buy':
#                     exit_signal = data['Sell_Signal'].iloc[i]
#                     if exit_signal:
#                         exit_price = close_price
#                         action = 'sell_signal'
#                         sell_signal_count += 1
#                 else:
#                     exit_signal = data['Buy_Signal'].iloc[i]
#                     if exit_signal:
#                         exit_price = close_price
#                         action = 'buy_signal'
#                         sell_signal_count += 1

#             if not exit_price:
#                 if strategy_type == 'buy':
#                     target_reached_price = entry_price * (1 + target_pct)
                    
#                     if trailing_stop_loss_pct is not None:
#                         # Check if the close price exceeds the highest profit achieved
#                         if high_price >= entry_price_reference * (1 + price_move_pct):
#                             next_reference_price = entry_price_reference * (1 + price_move_pct)
#                             # Move the trailing stop loss in the profit direction
#                             while high_price >= next_reference_price:
#                                 entry_price_reference = next_reference_price
#                                 trailing_stop_loss_price *= (1 + trailing_stop_loss_pct)
#                                 next_reference_price = entry_price_reference * (1 + price_move_pct)
                         
#                         if low_price <= trailing_stop_loss_price:
#                             action = 'tsl'
#                             tsl_count += 1
#                             exit_price = trailing_stop_loss_price
#                     if high_price >= target_reached_price:
#                         action = 'target'
#                         target_count += 1
#                         exit_price = target_reached_price
#                 else:  # sell strategy
#                     target_reached_price = entry_price * (1 - target_pct)
                    
#                     if trailing_stop_loss_pct is not None:
#                         # Check if the close price exceeds the highest profit achieved
#                         if low_price <= entry_price_reference * (1 - price_move_pct):
#                             next_reference_price = entry_price_reference * (1 - price_move_pct)
#                             # Move the trailing stop loss in the profit direction
#                             while low_price <= next_reference_price:
#                                 entry_price_reference = next_reference_price
#                                 trailing_stop_loss_price *= (1 - trailing_stop_loss_pct)
#                                 next_reference_price = entry_price_reference * (1 - price_move_pct)
                         
#                         if high_price >= trailing_stop_loss_price:
#                             action = 'tsl'
#                             tsl_count += 1
#                             exit_price = trailing_stop_loss_price
#                     if low_price <= target_reached_price:
#                         action = 'target'
#                         target_count += 1
#                         exit_price = target_reached_price

#             if trade_type == 'mis' and is_last_candle_of_day:
#                 action = 'Market Close'
#                 exit_price = close_price
#                 market_close += 1

#             if not exit_price:
#                 if strategy_type == 'buy':
#                     stop_loss_price = entry_price * (1 - stop_loss_pct)
#                     if low_price <= stop_loss_price:
#                         action = 'stoploss'
#                         stoploss_count += 1
#                         exit_price = stop_loss_price
#                 else:  # sell strategy
#                     stop_loss_price = entry_price * (1 + stop_loss_pct)
#                     if high_price >= stop_loss_price:
#                         action = 'stoploss'
#                         stoploss_count += 1
#                         exit_price = stop_loss_price

           
            

#             if exit_price:
#                 pnl = (exit_price - entry_price) * quantity - brokerage_per_trade if strategy_type == 'buy' else (entry_price - exit_price) * quantity - brokerage_per_trade
#                 funds += (exit_price * quantity - brokerage_per_trade) if strategy_type == 'buy' else (entry_price * quantity - exit_price * quantity - brokerage_per_trade)
#                 total_pnl += pnl
#                 total_brokerage += brokerage_per_trade

#                 invested_price = entry_price * quantity

#                 trades.append({
#                     'symbol': symbol,  # Assuming the symbol can be derived from the DataFrame index name
#                     'action': 'Sell' if strategy_type == 'buy' else 'Buy',
#                     'price': exit_price,
#                     'pnl': pnl,
#                     'quantity': quantity,
#                     'brokerage': brokerage_per_trade,
#                     'date': current_date,
#                     'invested': invested_price,
#                     'stopLossprice': stop_loss_price,
#                     'targetPrice': target_reached_price,
#                     'trailingSl': trailing_stop_loss_price,
#                     'trigger': action,
#                     'day_type': day_type
#                 })
#                 is_in_position = False
#                 trailing_stop_loss_price = 0
#                 trade_dates.append(current_date)
#                 cumulative_pnl.append(cumulative_pnl[-1] + pnl)

#                 # Track win or loss
#                 if pnl > 0:
#                     wins += 1
#                     gains.append(pnl)
#                     if last_outcome == 'win':
#                         current_streak += 1
#                     else:
#                         current_streak = 1
#                     win_streak = max(win_streak, current_streak)
#                     last_outcome = 'win'
#                 else:
#                     losses_count += 1
#                     losses.append(-pnl)
#                     if last_outcome == 'loss':
#                         current_streak += 1
#                     else:
#                         current_streak = 1
#                     loss_streak = max(loss_streak, current_streak)
#                     last_outcome = 'loss'

#                 if funds > peak_funds:
#                     peak_funds = funds
#                     drawdown_start = None
#                 else:
#                     if drawdown_start is None:
#                         drawdown_start = current_date
#                     current_drawdown = peak_funds - funds
#                     max_drawdown = max(max_drawdown, current_drawdown)
#                     if drawdown_start is not None:
#                         drawdown_days = (current_date - drawdown_start).days
#                         max_drawdown_days = max(max_drawdown_days, drawdown_days)


#         entry_signal = data['Buy_Signal'].iloc[i] if strategy_type == 'buy' else data['Sell_Signal'].iloc[i]

#         if entry_signal and not is_in_position and is_within_trading_hours(current_date):
#             entry_price = close_price 

#             # Adjust quantity based on position sizing type if provided
#             if position_size_type:
#                 if position_size_type == 'capital':
#                     quantity = min(max_size_amount // entry_price, max_quantity)
#                 elif position_size_type == 'risk':
#                     stop_loss_price = entry_price * (1 - stop_loss_pct)
#                     risk_per_share = entry_price - stop_loss_price
#                     quantity = min(max_size_amount // risk_per_share, max_quantity)

#             target_reached_price = (entry_price * target_pct) + entry_price
#             stop_loss_price = entry_price * (1 - stop_loss_pct)
#             trailing_stop_loss_price = stop_loss_price if trailing_stop_loss_pct is not None else 0
#             brokerage_per_trade = min(entry_price * quantity * 0.0003, 20)
#             funds -= (entry_price * quantity + brokerage_per_trade) if strategy_type == 'buy' else (entry_price * quantity - brokerage_per_trade)
#             is_in_position = True
#             entry_price_reference = entry_price
#             invested_price = entry_price * quantity
#             invested_fund += invested_price
#             trades.append({
#                 'symbol': symbol,  # Assuming the symbol can be derived from the DataFrame index name
#                 'action': 'Buy' if strategy_type == 'buy' else 'Sell',
#                 'price': entry_price,
#                 'quantity': quantity,
#                 'trailingSlPrice': trailing_stop_loss_price,
#                 'brokerage': brokerage_per_trade,
#                 'date': current_date,
#                 'targetPrice': target_reached_price,
#                 'stopLossprice': stop_loss_price,
#                 'trailingSl': trailing_stop_loss_price,
#                 'trigger': 'Entry Signal',
#                 'day_type': day_type
#             })
#             trade_dates.append(current_date)
#             cumulative_pnl.append(cumulative_pnl[-1])
#             total_brokerage += brokerage_per_trade

#     # Final metrics calculation
#     total_trades = wins + losses_count
#     avg_gain = sum(gains) / wins if wins else 0
#     avg_loss = sum(losses) / losses_count if losses_count else 0
#     win_rate = (wins / total_trades * 100) if total_trades else 0
#     loss_rate = (losses_count / total_trades * 100) if total_trades else 0
#     reward_to_risk = avg_gain / avg_loss if avg_loss != 0 else 0
#     expectancy = (reward_to_risk * win_rate / 100) - (loss_rate / 100) if total_trades else 0

#     return {
#         'trades': trades,
#         'Total Signals': total_trades,
#         'Number of Wins': wins,
#         'Number of Losses': losses_count,
#         'Winning Streak': win_streak,
#         'Losing Streak': loss_streak,
#         'Max Gains': max(gains, default=0),
#         'Max Loss': max(losses, default=0),
#         'Avg Gain per Winning Trade': avg_gain,
#         'Avg Loss per Losing Trade': avg_loss,
#         'Max Drawdown': max_drawdown,
#         'Max Drawdown Days': max_drawdown_days,
#         'Win Rate (%)': win_rate,
#         'Loss Rate (%)': loss_rate,
#         'Expectancy': expectancy,
#         'Profit Factor': reward_to_risk,
#         'Total PnL': total_pnl,
#         'Total Brokerage': total_brokerage,
#         'Net PnL After Brokerage': total_pnl - total_brokerage,
#         'Remaining Funds': funds,
#         'Trade Dates': trade_dates,
#         'targetCount': target_count,
#         'tslCount': tsl_count,
#         'slCount': stoploss_count,
#         'investedFund': invested_fund,
#         'Cumulative PnL': cumulative_pnl,
#         'sellSignalCount':sell_signal_count,
#         'marketCloseCount':market_close,
#     }


# def execute_and_analyze_strategy_2(data, strategy_type, stop_loss_pct, price_move_pct, trailing_stop_loss_pct, target_pct, initial_funds, quantity, position_size_type=None, max_size_amount=None, max_quantity=None, exit_conditions=None):
#     is_in_position = False
#     entry_price = 0
#     funds = initial_funds
#     trades = []
#     gains = []
#     losses = []
#     total_pnl = 0
#     total_brokerage = 0
#     win_streak = 0
#     loss_streak = 0
#     current_streak = 0
#     wins = 0
#     losses_count = 0
#     last_outcome = None
#     peak_funds = initial_funds
#     max_drawdown = 0
#     max_drawdown_days = 0
#     drawdown_start = None
#     trade_dates = [data.index[0]]
#     cumulative_pnl = [0]
#     trailing_stop_loss_price = 0
#     invested_fund = 0
#     tsl_count = 0
#     target_count = 0
#     stoploss_count = 0
#     entry_price_reference = 0
#     sell_signal_count = 0

#     for i in range(len(data)):
#         current_date = data.index[i]
#         high_price = data['high'].iloc[i]
#         low_price = data['low'].iloc[i]
#         close_price = data['close'].iloc[i]
#         open_price = data['open'].iloc[i]
#         brokerage_per_trade = min(close_price * quantity * 0.0003, 20)

#         # Determine the day type
#         if close_price > open_price * 1.01:
#             day_type = 'bullish'
#         elif close_price < open_price * 0.99:
#             day_type = 'bearish'
#         else:
#             day_type = 'sideways'

#         if is_in_position:
#             exit_price = None
#             action = None

#             if exit_conditions:
#                 exit_signal = data['Sell_Signal'].iloc[i]
#                 if exit_signal:
#                     exit_price = close_price
#                     action = 'sell_signal'
#                     sell_signal_count+=1

#             if not exit_price:
#                 if strategy_type == 'buy':
#                     target_reached_price = entry_price * (1 + target_pct)
                    
#                     if trailing_stop_loss_pct is not None:
#                         # Check if the close price exceeds the highest profit achieved
#                         if high_price >= entry_price_reference * (1 + price_move_pct):
#                             next_reference_price = entry_price_reference * (1 + price_move_pct)
#                             # Move the trailing stop loss in the profit direction
#                             while high_price >= next_reference_price:
#                                 entry_price_reference = next_reference_price
#                                 trailing_stop_loss_price *= (1 + trailing_stop_loss_pct)
#                                 next_reference_price = entry_price_reference * (1 + price_move_pct)
                         
#                         if low_price <= trailing_stop_loss_price:
#                             action = 'tsl'
#                             tsl_count += 1
#                             exit_price = trailing_stop_loss_price
#                     if high_price >= target_reached_price:
#                         action = 'target'
#                         target_count += 1
#                         exit_price = target_reached_price
#                 else:  # sell strategy
#                     target_reached_price = entry_price * (1 - target_pct)
                    
#                     if trailing_stop_loss_pct is not None:
#                         # Check if the close price exceeds the highest profit achieved
#                         if low_price <= entry_price_reference * (1 - price_move_pct):
#                             next_reference_price = entry_price_reference * (1 - price_move_pct)
#                             # Move the trailing stop loss in the profit direction
#                             while low_price <= next_reference_price:
#                                 entry_price_reference = next_reference_price
#                                 trailing_stop_loss_price *= (1 - trailing_stop_loss_pct)
#                                 next_reference_price = entry_price_reference * (1 - price_move_pct)
                         
#                         if high_price >= trailing_stop_loss_price:
#                             action = 'tsl'
#                             tsl_count += 1
#                             exit_price = trailing_stop_loss_price
#                     if low_price <= target_reached_price:
#                         action = 'target'
#                         target_count += 1
#                         exit_price = target_reached_price

#             if not exit_price:
#                 if strategy_type == 'buy':
#                     stop_loss_price = entry_price * (1 - stop_loss_pct)
#                     if low_price <= stop_loss_price:
#                         action = 'stoploss'
#                         stoploss_count += 1
#                         exit_price = stop_loss_price
#                 else:  # sell strategy
#                     stop_loss_price = entry_price * (1 + stop_loss_pct)
#                     if high_price >= stop_loss_price:
#                         action = 'stoploss'
#                         stoploss_count += 1
#                         exit_price = stop_loss_price

#             if exit_price:
#                 pnl = (exit_price - entry_price) * quantity - brokerage_per_trade if strategy_type == 'buy' else (entry_price - exit_price) * quantity - brokerage_per_trade
#                 funds += (exit_price * quantity - brokerage_per_trade) if strategy_type == 'buy' else (entry_price * quantity - exit_price * quantity - brokerage_per_trade)
#                 total_pnl += pnl
#                 total_brokerage += brokerage_per_trade

#                 invested_price = entry_price * quantity

#                 trades.append({
#                     'symbol': symbol,  # Assuming the symbol can be derived from the DataFrame index name
#                     'action': 'Sell' if strategy_type == 'buy' else 'Buy',
#                     'price': exit_price,
#                     'pnl': pnl,
#                     'quantity': quantity,
#                     'brokerage': brokerage_per_trade,
#                     'date': current_date,
#                     'invested': invested_price,
#                     'stopLossprice': stop_loss_price,
#                     'targetPrice': target_reached_price,
#                     'trailingSl': trailing_stop_loss_price,
#                     'trigger': action,
#                     'day_type': day_type
#                 })
#                 is_in_position = False
#                 trailing_stop_loss_price = 0
#                 trade_dates.append(current_date)
#                 cumulative_pnl.append(cumulative_pnl[-1] + pnl)

#                 # Track win or loss
#                 if pnl > 0:
#                     wins += 1
#                     gains.append(pnl)
#                     if last_outcome == 'win':
#                         current_streak += 1
#                     else:
#                         current_streak = 1
#                     win_streak = max(win_streak, current_streak)
#                     last_outcome = 'win'
#                 else:
#                     losses_count += 1
#                     losses.append(-pnl)
#                     if last_outcome == 'loss':
#                         current_streak += 1
#                     else:
#                         current_streak = 1
#                     loss_streak = max(loss_streak, current_streak)
#                     last_outcome = 'loss'

#                 if funds > peak_funds:
#                     peak_funds = funds
#                     drawdown_start = None
#                 else:
#                     if drawdown_start is None:
#                         drawdown_start = current_date
#                     current_drawdown = peak_funds - funds
#                     max_drawdown = max(max_drawdown, current_drawdown)
#                     if drawdown_start is not None:
#                         drawdown_days = (current_date - drawdown_start).days
#                         max_drawdown_days = max(max_drawdown_days, drawdown_days)

#         entry_signal = data['Buy_Signal'].iloc[i] if strategy_type == 'buy' else data['Sell_Signal'].iloc[i]

#         if entry_signal and not is_in_position:
#             entry_price = close_price

#             # Adjust quantity based on position sizing type if provided
#             if position_size_type:
#                 if position_size_type == 'capital':
#                     quantity = min(max_size_amount // entry_price, max_quantity)
#                 elif position_size_type == 'risk':
#                     stop_loss_price = entry_price * (1 - stop_loss_pct)
#                     risk_per_share = entry_price - stop_loss_price
#                     quantity = min(max_size_amount // risk_per_share, max_quantity)

#             target_reached_price = (entry_price * target_pct) + entry_price
#             stop_loss_price = entry_price * (1 - stop_loss_pct)
#             trailing_stop_loss_price = stop_loss_price if trailing_stop_loss_pct is not None else 0
#             brokerage_per_trade = min(entry_price * quantity * 0.0003, 20)
#             funds -= (entry_price * quantity + brokerage_per_trade) if strategy_type == 'buy' else (entry_price * quantity - brokerage_per_trade)
#             is_in_position = True
#             entry_price_reference = entry_price
#             invested_price = entry_price * quantity
#             invested_fund += invested_price
#             trades.append({
#                 'symbol': symbol,  # Assuming the symbol can be derived from the DataFrame index name
#                 'action': 'Buy' if strategy_type == 'buy' else 'Sell',
#                 'price': entry_price,
#                 'quantity': quantity,
#                 'trailingSlPrice': trailing_stop_loss_price,
#                 'brokerage': brokerage_per_trade,
#                 'date': current_date,
#                 'targetPrice': target_reached_price,
#                 'stopLossprice': stop_loss_price,
#                 'trailingSl': trailing_stop_loss_price,
#                 'trigger': 'entry_signal',
#                 'day_type': day_type
#             })
#             trade_dates.append(current_date)
#             cumulative_pnl.append(cumulative_pnl[-1])
#             total_brokerage += brokerage_per_trade

#     # Final metrics calculation
#     total_trades = wins + losses_count
#     avg_gain = sum(gains) / wins if wins else 0
#     avg_loss = sum(losses) / losses_count if losses_count else 0
#     win_rate = (wins / total_trades * 100) if total_trades else 0
#     loss_rate = (losses_count / total_trades * 100) if total_trades else 0
#     reward_to_risk = avg_gain / avg_loss if avg_loss != 0 else 0
#     expectancy = (reward_to_risk * win_rate / 100) - (loss_rate / 100) if total_trades else 0

#     return {
#         'trades': trades,
#         'Total Signals': total_trades,
#         'Number of Wins': wins,
#         'Number of Losses': losses_count,
#         'Winning Streak': win_streak,
#         'Losing Streak': loss_streak,
#         'Max Gains': max(gains, default=0),
#         'Max Loss': max(losses, default=0),
#         'Avg Gain per Winning Trade': avg_gain,
#         'Avg Loss per Losing Trade': avg_loss,
#         'Max Drawdown': max_drawdown,
#         'Max Drawdown Days': max_drawdown_days,
#         'Win Rate (%)': win_rate,
#         'Loss Rate (%)': loss_rate,
#         'Expectancy': expectancy,
#         'Profit Factor': reward_to_risk,
#         'Total PnL': total_pnl,
#         'Total Brokerage': total_brokerage,
#         'Net PnL After Brokerage': total_pnl - total_brokerage,
#         'Remaining Funds': funds,
#         'Trade Dates': trade_dates,
#         'targetCount': target_count,
#         'tslCount': tsl_count,
#         'slCount': stoploss_count,
#         'investedFund': invested_fund,
#         'Cumulative PnL': cumulative_pnl,
#         'sellSignalCount':sell_signal_count,
#     }




#encoding date format
class CustomEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Timestamp):
            # Format date in any format you prefer
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        # Let the base class default method raise the TypeError
        return JSONEncoder.default(self, obj)
    




start_date = pd.to_datetime(start_date, format='%d-%m-%Y')
end_date = pd.to_datetime(end_date, format='%d-%m-%Y')


results = []

base_dir = os.path.join(os.path.dirname(__file__), 'config', 'stock_historical_data')

# looping over every symbol for backtesting

for symbol in symbol:
    
    # file_path = os.path.join(r'D:\stock_historical_data\historical_data_' + time_period, f'{symbol}.csv')
    file_path = os.path.join(base_dir, time_period, f'{symbol}.csv')
    sbi_data = pd.read_csv(file_path)
    # sbi_data = suppress_print(equity_history, symbol, series, start_date, end_date)
    sbi_data['date'] = pd.to_datetime(sbi_data['date'], utc=False)  
    data_timezone = sbi_data['date'].dt.tz
    start_date = start_date.tz_localize(data_timezone)
    end_date = end_date.tz_localize(data_timezone)


    # mask = (sbi_data['date'] >= start_date) & (sbi_data['date'] <= end_date)
    # sbi_data = sbi_data.loc[mask]

    # Filter the DataFrame using the between method
    # sbi_data = sbi_data[sbi_data['date'].between(start_date, end_date)]

    sbi_data = sbi_data[(sbi_data['date'] >= start_date) & (sbi_data['date'] <= end_date)]

    start_date = start_date.tz_localize(None)
    end_date = end_date.tz_localize(None)

    close_prices = sbi_data[['date', 'close']].reset_index().to_json(orient='records')
    # sbi_data.sort_values(by='date', inplace=True)
    sbi_data.set_index('date', inplace=True)



    
    if graph_type == 'heikin-ashi':
        data = compute_heikin_ashi(sbi_data)
        buy_signals, sell_signals = combine_signals(data, conditions, operator , exit_conditions , exit_operator)
        entry_signals_short , exit_signals_short = combine_signals(data,conditions2,operator2,exit_conditions2,exit_operator2 )
        sbi_data['Buy_Signal'] = buy_signals
        sbi_data['Sell_Signal'] = sell_signals
        sbi_data['entry_signals_short'] = entry_signals_short
        sbi_data['exit_signals_short'] = exit_signals_short

    else:
        buy_signals, sell_signals = combine_signals(sbi_data, conditions, operator ,exit_conditions , exit_operator)
        entry_signals_short , exit_signals_short = combine_signals(sbi_data,conditions2,operator2,exit_conditions2,exit_operator2 )
        sbi_data['Buy_Signal'] = buy_signals
        sbi_data['Sell_Signal'] = sell_signals
        sbi_data['entry_signals_short'] = entry_signals_short
        sbi_data['exit_signals_short'] = exit_signals_short
    
    

    # Execute strategy analysis
    # strategy_metrics_2 = execute_and_analyze_strategy_2(sbi_data, strategy_type ,stopLoss,trailing_sl, target, initial_capital, quantity,position_size_type,max_size_amount,max_quantity,exit_conditions)
  
    strategy_metrics = execute_and_analyze_strategy(sbi_data, strategy_type ,stopLoss,moveInstrument,moveSl, target, initial_capital, quantity,trade_type,position_size_type,max_size_amount,max_quantity,exit_conditions,exit_conditions2)
    
   
    # strategy_metrics_2 = execute_and_analyze_strategy_2(sbi_data, strategy_type ,stopLoss,moveInstrument,moveSl, target, initial_capital, quantity,trade_type,position_size_type,max_size_amount,max_quantity,exit_conditions)
    
    trade_dates = pd.to_datetime(strategy_metrics['Trade Dates'])
    
   

    trades = strategy_metrics['trades']
    buy_dates = [pd.to_datetime(trade['date']) for trade in trades if trade['action'] == 'Buy']
    sell_dates = [pd.to_datetime(trade['date']) for trade in trades if trade['action'] == 'Sell']
    buy_prices = [trade['price'] for trade in trades if trade['action'] == 'Buy']
    sell_prices = [trade['price'] for trade in trades if trade['action'] == 'Sell']


    trades2 = strategy_metrics['tradesShort']
    buy_dates_short = [pd.to_datetime(trade['date']) for trade in trades2 if trade['action'] == 'Sell']
    sell_dates_short = [pd.to_datetime(trade['date']) for trade in trades2 if trade['action'] == 'Buy']
    buy_prices_short = [trade['price'] for trade in trades2 if trade['action'] == 'Sell']
    sell_prices_short = [trade['price'] for trade in trades2 if trade['action'] == 'Buy']

    labels = {pd.to_datetime(trade['date']): trade['tradeNumber'] for trade in trades}
    labels_short = {pd.to_datetime(trade['date']): trade['tradeNumber'] for trade in trades2}



    if conditions and not conditions2:
        monthly_pnl = compute_monthly_pnl(strategy_metrics['trades'],strategy_metrics['investedFund'])
        monthly_pnl_short = pd.DataFrame(columns=['date', 'pnl', 'invested'])
        monthly_pnl_total = monthly_pnl

    elif conditions2 and not conditions:
        monthly_pnl_short = compute_monthly_pnl(strategy_metrics['tradesShort'],strategy_metrics['investedFundShort'])
        monthly_pnl = pd.DataFrame(columns=['date', 'pnl', 'invested'])
        monthly_pnl_total = monthly_pnl_short
    else:
        
        monthly_pnl = compute_monthly_pnl(strategy_metrics['trades'],strategy_metrics['investedFund'])
        
        monthly_pnl_short = compute_monthly_pnl(strategy_metrics['tradesShort'],strategy_metrics['investedFundShort'])
       
        monthly_pnl_total = compute_combined_pnl(monthly_pnl,monthly_pnl_short)
        

    plt.figure(figsize=(15, 5))
    plt.plot(sbi_data.index, sbi_data['close'], label='Closing Price')

    # Plotting and annotating buy points
    if conditions:
        for idx, price in zip(buy_dates, buy_prices):
            plt.scatter(idx, price, color='blue', label='Entry' if idx == buy_dates[0] else "", marker='^')
            plt.annotate(labels[idx], 
                        (idx, price),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')

        for idx, price in zip(sell_dates, sell_prices):
            plt.scatter(idx, price, color='red', label='Exit' if idx == sell_dates[0] else "", marker='v')
            plt.annotate(labels[idx],  # This text can include any additional info
                        (idx, price),
                        textcoords="offset points",
                        xytext=(0,-15),
                        ha='center')

        
    
        
    if conditions2:         
        for idx, price in zip(buy_dates_short, buy_prices_short):
            plt.scatter(idx, price, color='blue', label='Entry' if idx == buy_dates_short[0] else "", marker='^')
            plt.annotate(labels_short[idx],  # This text can include any additional info
                        (idx, price),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')


        for idx, price in zip(sell_dates_short, sell_prices_short):
            plt.scatter(idx, price, color='red', label='Exit' if idx == sell_dates_short[0] else "", marker='v')
            plt.annotate(labels_short[idx],  # This text can include any additional info
                        (idx, price),
                        textcoords="offset points",
                        xytext=(0,-15),
                        ha='center')

    # Connect buy and sell points with lines
        

    # for i in range(min(len(buy_dates), len(sell_dates))):
    #     plt.plot([buy_dates[i], sell_dates[i]], [buy_prices[i], sell_prices[i]], color='blue', linestyle='-', linewidth=1)

    plt.legend()
    plt.title('Trades')
    plt.xlabel('Date')
    plt.ylabel('Price')
    

    # Save to buffer and convert to base64 for web display or further processing
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    trade_graph = base64.b64encode(buf.getvalue()).decode('utf-8')
    # Funds Over Time Graph
    buf2 = BytesIO()
    plt.figure(figsize=(15, 5))
    plt.plot(trade_dates, strategy_metrics['Cumulative PnL'], label='Cumulative P&L', marker='o') 

    # Annotate points with P&L values and color based on increase or decrease
    prev_pnl = strategy_metrics['Cumulative PnL'][0]  # Initialize previous P&L value
    for date, pnl in zip(trade_dates, strategy_metrics['Cumulative PnL']):
        color = 'green' if pnl > prev_pnl else 'red'  # Check if current P&L is greater than previous P&L
        plt.text(date, pnl, f'{pnl:.0f}', ha='right' if pnl > prev_pnl else 'left', va='bottom', fontsize=8, color=color)
        prev_pnl = pnl  # Update previous P&L value for the next iteration

    plt.title(f'Cumulative P&L Over Time for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Pnl')
   
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig(buf2, format='png')
    plt.close()
    buf2.seek(0)
    funds_graph = base64.b64encode(buf2.getvalue()).decode('utf-8')


    
    results.append({
        'symbol': symbol,
        'result': strategy_metrics,
        'trade_graph': trade_graph,
        'funds_graph': funds_graph,
        'monthly_pnl': monthly_pnl.to_json(orient='split'),
        'monthly_pnl_short':monthly_pnl_short.to_json(orient='split'),
        'monthly_pnl_total':monthly_pnl_total.to_json(orient='split'),
        'closePrices': close_prices
    })


print(json.dumps(results, cls=CustomEncoder))