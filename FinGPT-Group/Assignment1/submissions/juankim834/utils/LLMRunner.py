import csv
import os
import time
import pandas as pd
from openai import OpenAI
from prompt import promptGenerator as prompt_gen


class LLMRunner:
    """
    A runner class for querying LLM models on stock data and saving results to CSV.
    
    Attributes:
        client (OpenAI): The OpenAI client instance for API calls
        data_dir (str): Directory path for storing output CSV files
        start_date: Start date for the analysis period
        end_date: End date for the analysis period
        model_name (str): Name of the model to use (default: "gpt-4")
    """
    
    def __init__(self, client: OpenAI, data_dir: str, start_date, end_date, model_name: str = "gpt-4"):
        """
        Initialize the LLMRunner with client and configuration parameters.
        
        Args:
            client (OpenAI): OpenAI client instance
            data_dir (str): Directory to store CSV output files
            start_date: Start date for data analysis
            end_date: End date for data analysis
            model_name (str, optional): Model name to use. Defaults to "gpt-4"
        """
        self.client = client
        self.data_dir = data_dir
        self.start_date = start_date
        self.end_date = end_date
        self.model_name = model_name
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def append_to_csv(self, csv_file, input_data, output_data):
        """
        Append a row of input and output data to a CSV file.
        
        Args:
            csv_file (str): Path to the CSV file
            input_data: Input prompt data
            output_data: Model output/answer data
        """
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([input_data, output_data])

    def initialize_csv(self, csv_file, headers=["prompt", "answer"]):
        """
        Initialize a new CSV file with headers.
        
        Args:
            csv_file (str): Path to the CSV file to create
            headers (list, optional): Column headers. Defaults to ["prompt", "answer"]
        """
        with open(csv_file, mode='w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    def get_prompt_gen(self, stock_symbol):
        """
        Create a prompt generator instance for a given stock symbol.
        
        Args:
            stock_symbol (str): Stock ticker symbol
            
        Returns:
            promptGenerator: Configured prompt generator instance
        """
        return prompt_gen(stock_symbol, self.data_dir, self.start_date, self.end_date, api_key=self.client.api_key)
    
    def query_gpt4(self, symbol_list, min_past_weeks=1, max_past_weeks=3, with_basics=True):
        """
        Query GPT-4 for stock predictions and save results to CSV files.
        
        This method processes multiple stock symbols, generates prompts for each,
        queries the LLM, and saves responses to individual CSV files. It supports
        resuming from previous runs by checking existing CSV files.
        
        Args:
            symbol_list (list): List of stock ticker symbols to process
            min_past_weeks (int, optional): Minimum past weeks for prompts. Defaults to 1
            max_past_weeks (int, optional): Maximum past weeks for prompts. Defaults to 3
            with_basics (bool, optional): Include basic stock info in prompts. Defaults to True
        """
        for symbol in symbol_list:
            # Determine CSV filename based on configuration
            csv_file = (f'{self.data_dir}/{symbol}_{self.start_date}_{self.end_date}_gpt-4.csv' 
                       if with_basics 
                       else f'{self.data_dir}/{symbol}_{self.start_date}_{self.end_date}_gpt-4_no_basics.csv')
            
            # Initialize or resume from existing CSV
            if not os.path.exists(csv_file):
                self.initialize_csv(csv_file)
                pre_done = 0
            else:
                df = pd.read_csv(csv_file)
                pre_done = df.shape[0]

            # Generate prompts for this symbol
            prompts_generator = self.get_prompt_gen(symbol)
            prompts = prompts_generator.get_all_prompts(min_past_weeks, max_past_weeks, with_basics)

            # Process each prompt
            for i, prompt in enumerate(prompts):
                # Skip already processed prompts
                if i < pre_done:
                    continue

                print(f"{symbol} - {i}")

                # Retry logic for API calls
                completion = None
                cnt = 0
                while cnt < 5:
                    try:
                        completion = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": prompts_generator.DEFAULT_SYSTEM_PROMPT},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        break
                    except Exception as e:
                        cnt += 1
                        print(f"retry cnt {cnt}, error={e}")
                        time.sleep(1)

                # Extract answer or use empty string if all retries failed
                answer = completion.choices[0].message.content if completion else ""
                if not completion:
                    print(f"Warning: Failed to get completion for {symbol} prompt {i} after 5 retries")
                
                self.append_to_csv(csv_file, prompt, answer)