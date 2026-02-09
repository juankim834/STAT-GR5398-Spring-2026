from utils import getRawData
from typing import Tuple, List
import random
import pandas as pd
import json

class promptGenerator(getRawData):

    DEFAULT_SYSTEM_PROMPT = (
        "You are a seasoned stock market analyst. Your task is to list the positive developments "
        "and potential concerns for companies based on relevant news and basic financials from the past weeks, "
        "then provide an analysis and prediction for the companies' stock price movement for the upcoming week.\n"
            "Your answer format should be as follows:\n\n"
            "[Positive Developments]:\n1. ...\n\n"
            "[Potential Concerns]:\n1. ...\n\n"
            "[Prediction & Analysis]:\n...\n"
    )


    def __init__(self, data_dir, stock_symbol, start_date, end_date, api_key, system_prompt=None):
        super().__init__(data_dir, stock_symbol, start_date, end_date, api_key)
        self.system_prompt = system_prompt if system_prompt is not None else self.DEFAULT_SYSTEM_PROMPT

    def get_company_news(self):

        profile = self.finnhub_client.company_profile2(symbol=self.stock_symbol)

        company_template = "[Company Introduction]:\n\n{name} is a leading entity in the {finnhubIndustry} sector. Incorporated and publicly traded since {ipo}, the company has established its reputation as one of the key players in the market. As of today, {name} has a market capitalization of {marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding." \
            "\n\n{name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive progress within the industry."
        
        formatted_str = company_template.format(**profile)

        return formatted_str
    
    def get_prompt_by_row(self, row) -> Tuple[str, List[str], str]:

        start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
        term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
        head = "From {} to {}, {}'s stock price {} from {:.2f} to {:.2f}. Company news during this period are listed below:\n\n".format(
            start_date, end_date, self.stock_symbol, term, row['Start Price'], row['End Price']
        )

        news = json.loads(row["News"])
        news = ["[Headline]: {}\n[Summary]: {}\n".format(
            n['headline'], n['summary']) for n in news if n['date'][:8] <= end_date.replace('-', '') and \
            not n['summary'].startswith("Looking for stock market analysis and research with proves results?")]
        basics = json.loads(row["Basics"])
        if basics:
            basics = "Some recent basic financials of {}, reported at {}, are presented below:\n\n[Basic Financials]:\n\n".format(
                self.stock_symbol, basics['period']) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
        else:
            basics = "[Basic Financials]:\n\nNo basic financial reported."
        return head, news, basics
    
    def sample_news(self, news, k=5):

        return [news[i] for i in sorted(random.sample(range(len(news)), k))]

    def map_bin_label(self, bin_lb):

        lb = bin_lb.replace('U', 'up by ')
        lb = lb.replace('D', 'down by ')
        lb = lb.replace('1', '0-1%')
        lb = lb.replace('2', '1-2%')
        lb = lb.replace('3', '2-3%')
        lb = lb.replace('4', '3-4%')
        if lb.endswith('+'):
            lb = lb.replace('5+', 'more than 5%')
            # lb = lb.replace('5+', '5+%')
        else:
            lb = lb.replace('5', '4-5%')
        
        return lb

    def get_all_prompts(self, min_past_weeks=1, max_past_weeks=4, with_basics=True, return_chat_format=True) -> List[str]:

        if with_basics:
            df = pd.read_csv(f'{self.data_dir}/{self.stock_symbol}_{self.start_date}_{self.end_date}.csv')
        else:
            df = pd.read_csv(f'{self.data_dir}/{self.stock_symbol}_{self.start_date}_{self.end_date}_nobasics.csv')
        
        company_prompt = self.get_company_news()

        prev_rows = []
        all_prompts = []
        
        for row_index, row in df.iterrows():

            prompt = ""
            if len(prev_rows) >= min_past_weeks:
                idx = min(random.choice(range(min_past_weeks, max_past_weeks+1)), len(prev_rows))
                for i in range(-idx, 0):
                    # Add Price Movement (Head)
                    prompt += "\n" + prev_rows[i][0]
                    # Add News of previous weeks
                    sampled_news = self.sample_news(
                        prev_rows[i][1],
                        min(5, len(prev_rows[i][1]))
                    )
                    if sampled_news:
                        prompt += "\n".join(sampled_news)
                    else:
                        prompt += "No relative news reported."

            head, news, basics = self.get_prompt_by_row(row)

            prev_rows.append((head, news, basics))
            if len(prev_rows) > max_past_weeks:
                prev_rows.pop(0)

            if not prompt:
                continue

            prediction = self.map_bin_label(row['Bin Label'])
        
            prompt = company_prompt + '\n' + prompt + '\n' + basics
            prompt += f"\n\nBased on all the information before {row['Start Date']}, let's first analyze the positive developments and potential concerns for {self.stock_symbol}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. " \
                f"Then let's assume your prediction for next week ({row['Start Date']} to {row['End Date']}) is {prediction}. Provide a summary analysis to support your prediction. The prediction result need to be inferred from your analysis at the end, and thus not appearing as a foundational factor of your analysis."

            all_prompts.append(prompt.strip())
        
            if return_chat_format:
                # Format A: Chat-based (Available for GPT-3.5-Turbo, GPT-4, etc.)
                chat_messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt} # Your original prompt here
                ]
                all_prompts.append(chat_messages)
            else:
                # Format B: Standard Prompt-based (Available for GPT-3, Claude, etc.)
                full_text = f"<<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n[INST]\n{prompt}\n[/INST]"
                all_prompts.append(full_text)

        return all_prompts
