import math
import yfinance as yf
import finnhub
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
import time
import json

class getRawData():
    def __init__(self, data_dir, stock_symbol, start_date, end_date, api_key):
        self.data_dir = data_dir
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.finnhub_client = finnhub.Client(api_key = api_key)
    
    def bin_mapping(self, ret) -> str:

        up_down = 'U' if ret >= 0 else 'D'
        integer = math.ceil(abs(100 * ret))
        return up_down + (str(integer) if integer <= 5 else '5+')
    
    def get_returns(self) -> pd.DataFrame:

        """
        get_returns
            Download historical stock data using yfinance, resample it to weekly frequency, and calculate weekly returns.
        """
        stock_data = yf.download(self.stock_symbol, start=self.start_date, end=self.end_date)

        weekly_data = stock_data['Adj Close'].resample('W').ffill()
        weekly_returns = weekly_data.pct_change()[1:]
        weekly_start_prices = weekly_data[:-1]
        weekly_end_prices = weekly_data[1:]

        weekly_data = pd.DataFrame({
            'Start Date': weekly_start_prices.index,
            'Start Price': weekly_start_prices.values,
            'End Date': weekly_end_prices.index,
            'End Price': weekly_end_prices.values,
            'Weekly Returns': weekly_returns.values
        })

        weekly_data['Bin Label'] = weekly_data['Weekly Returns'].map(self.bin_mapping)

        return weekly_data

    def get_news(self, data) -> pd.DataFrame:
        """
        get_news
            For each week defined by the 'Start Date' and 'End Date' in the input DataFrame, fetch news articles related to the stock symbol using the Finnhub API. The news articles are stored as a JSON string in a new column called 'News'.

        :param data: A DataFrame containing 'Start Date' and 'End Date' columns that define the weekly periods for which news articles will be fetched.
        :return: A DataFrame with an additional 'News' column containing JSON strings of news articles for each week.
        :rtype: DataFrame
        """

        news_list = []
        
        for end_date, row in data.iterrows():
            start_date = row['Start Date'].strftime('%Y-%m-%d')
            end_date = row['End Date'].strftime('%Y-%m-%d')
            print(self.stock_symbol, ':', start_date, '-', end_date)
            time.sleep(1)
            weekly_news = self.finnhub_client.company_news(self.stock_symbol, _from=start_date, to=end_date)

            weekly_news = [
                {
                    "date": datetime.fromtimestamp(news_item['datetime']).strftime('%Y%m%d%H%M%S'),
                    "headline": news_item['headline'],
                    "summary": news_item['summary'],
                } for news_item in weekly_news
            ]
            weekly_news.sort(key=lambda x: x['date'])
            news_list.append(json.dumps(weekly_news))

        data['News'] = news_list

        return data
    
    def get_basics(self, data, always=False):

        basic_financials = self.finnhub_client.company_basic_financials(self.stock_symbol, 'all')

        final_basics, basic_list, basic_dict = [], [], defaultdict(dict)
    
        for metric, value_list in basic_financials['series']['quarterly'].items():
            for value in value_list:
                basic_dict[value['period']].update({metric: value['v']})

        for k, v in basic_dict.items():
            v.update({'period': k})
            basic_list.append(v)
            
        basic_list.sort(key=lambda x: x['period'])
                
        for i, row in data.iterrows():
            
            start_date = row['End Date'].strftime('%Y-%m-%d')
            last_start_date = self.start_date if i < 2 else data.loc[i-2, 'Start Date'].strftime('%Y-%m-%d')
            
            used_basic = {}
            for basic in basic_list[::-1]:
                if (always and basic['period'] < start_date) or (last_start_date <= basic['period'] < start_date):
                    used_basic = basic
                    break
            final_basics.append(json.dumps(used_basic))
            
        data['Basics'] = final_basics
        
        return data
    
def prepare_data_for_company(stock_symbol, start_date, end_date, api_key, data_dir,
                             always=False,
                             with_basics=True):
    data_processor = getRawData(data_dir, stock_symbol, start_date, end_date, api_key)
    returns_data = data_processor.get_returns()
    news_data = data_processor.get_news(returns_data)
    if with_basics:
        news_data = data_processor.get_basics(news_data, always)
        news_data.to_csv(f"{data_dir}/{stock_symbol}_{start_date}_{end_date}.csv")
    else:
        news_data['Basics'] = [json.dumps({})] * len(news_data)
        news_data.to_csv(f"{data_dir}/{stock_symbol}_{start_date}_{end_date}_nobasics.csv")

    return news_data