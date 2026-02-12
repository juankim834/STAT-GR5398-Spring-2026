"""
Stock Prediction Dataset Processing Module

This module handles the conversion of GPT-4 generated stock predictions
into LLaMA-compatible format and creates train/test datasets.
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets

class StockPredictionProcessor:
    """
    Processes stock prediction data from GPT-4 CSV files into LLaMA format.
    
    Attributes:
        data_dir (Path): Directory containing the CSV files
        start_date (str): Start date for the data range
        end_date (str): End date for the data range
        system_prompt (str): System prompt template for LLaMA
        b_inst (str): Beginning instruction token
        e_inst (str): Ending instruction token
        b_sys (str): Beginning system token
        e_sys (str): Ending system token
    """
    PERIOD_LABEL_PATTERN = r"Then let's assume your prediction for next week \((.*)\) is ((?:up|down) by .*%)\."
    PROMPT_REPLACEMENT_PATTERN = (
        r"Then let's assume your prediction for next week \((.*)\) is (up|down) by ((?:.*)%)\. "
        r"Provide a summary analysis to support your prediction\. The prediction result need to be "
        r"inferred from your analysis at the end, and thus not appearing as a foundational factor "
        r"of your analysis\."
    )
    ANSWER_PATTERN = r"\[Prediction & Analysis\]:\s*"

    def __init__(self, data_dir: str, start_date: str, end_date: str, system_prompt: str, b_inst: str, e_inst: str, b_sys: str, e_sys: str):
        self.data_dir = Path(data_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.system_prompt = system_prompt
        self.b_inst = b_inst
        self.e_inst = e_inst
        self.b_sys = b_sys
        self.e_sys = e_sys

    def _get_csv_files(self, symbol:str, with_basics:bool = True) -> Path:
        """Construct the CSV file path for a given symbol."""
        suffix = "gpt-4" if with_basics else "nobasics_gpt-4"
        filename = f"{symbol}_{self.start_date}_{self.end_date}_{suffix}.csv"
        return self.data_dir / filename

    def _extract_period_and_label(self, prompt: str) -> Tuple[str, str]:
        """Extract the period and label from the prompt using regex."""
        match = re.search(self.PERIOD_LABEL_PATTERN, prompt)
        if not match:
            raise ValueError(f"Could not extract period and label from prompt: {prompt[:100]}...")
        return match.group(1), match.group(2)
    
    def _transform_prompt(self, prompt: str, symbol: str, period: str) -> str:
        """Transform the prompt to remove the assumed prediction."""
        new_prompt = re.sub(
            self.PROMPT_REPLACEMENT_PATTERN,
            f"Then make your prediction of the {symbol} stock price movement for next week ({period}). "
            f"Provide a summary analysis to support your prediction.",
            prompt
        )
        
        # Add system prompt formatting
        modified_system_prompt = self.system_prompt.replace(
            ':\n...',
            '\nPrediction: ...\nAnalysis: ...'
        )
        
        return (
            f"{self.b_inst}{self.b_sys}{modified_system_prompt}{self.e_sys}"
            f"{new_prompt}{self.e_inst}"
        )
    
    def _transform_answer(self, answer: str, label: str) -> str:
        """Transform the answer to include explicit prediction and analysis sections."""
        return re.sub(
            self.ANSWER_PATTERN,
            f"[Prediction & Analysis]:\nPrediction: {label.capitalize()}\nAnalysis: ",
            answer
        )
    
    def process_symbol(self, symbol: str, with_basics: bool = True) -> Dict[str, List]:
        """
        Process a single stock symbol's data from CSV.
        
        Args:
            symbol: Stock ticker symbol
            with_basics: Whether to use the file with basic features
            
        Returns:
            Dictionary containing lists of prompts, answers, periods, and labels
        """
        csv_path = self._get_csv_files(symbol, with_basics)
        df = pd.read_csv(csv_path)
        
        prompts, answers, periods, labels = [], [], [], []
        
        for i, row in df.iterrows():
            try:
                prompt, answer = row['prompt'], row['answer']
                
                # Extract period and label
                period, label = self._extract_period_and_label(prompt)
                
                # Transform prompt and answer
                transformed_prompt = self._transform_prompt(prompt, symbol, period)
                transformed_answer = self._transform_answer(answer, label)
                
                prompts.append(transformed_prompt)
                answers.append(transformed_answer)
                periods.append(period)
                labels.append(label)
                
            except Exception as e:
                print(f"Error processing {symbol} at index {i}: {e}")
                print(f"Label: {label if 'label' in locals() else 'N/A'}")
                print(f"Answer preview: {answer[:100] if 'answer' in locals() else 'N/A'}...")
                continue
        
        return {
            "prompt": prompts,
            "answer": answers,
            "period": periods,
            "label": labels,
        }

class StockPredictionDatasetBuilder:
    """
    Builds train/test datasets from processed stock prediction data.
    """
    
    def __init__(self, processor: StockPredictionProcessor):
        self.processor = processor
    
    def create_dataset(
        self,
        symbol_list: List[str],
        train_ratio: float = 0.8,
        with_basics: bool = True
    ) -> DatasetDict:
        """
        Create train/test datasets from a list of stock symbols.
        
        Args:
            symbol_list: List of stock ticker symbols to process
            train_ratio: Ratio of data to use for training (default: 0.8)
            with_basics: Whether to use files with basic features (default: True)
            
        Returns:
            DatasetDict containing 'train' and 'test' datasets
        """
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
        
        train_datasets = []
        test_datasets = []
        
        for symbol in symbol_list:
            # Process the symbol data
            data_dict = self.processor.process_symbol(symbol, with_basics)
            
            # Add symbol column
            data_dict["symbol"] = [symbol] * len(data_dict['label'])
            
            # Create dataset and split
            dataset = Dataset.from_dict(data_dict)
            train_size = round(train_ratio * len(dataset))
            
            train_datasets.append(dataset.select(range(train_size)))
            test_datasets.append(dataset.select(range(train_size, len(dataset))))
        
        # Concatenate all datasets
        train_dataset = concatenate_datasets(train_datasets)
        test_dataset = concatenate_datasets(test_datasets)
        
        return DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })


# Convenience function for backward compatibility
def create_stock_prediction_dataset(
    symbol_list: List[str],
    data_dir: str,
    start_date: str,
    end_date: str,
    system_prompt: str,
    b_inst: str,
    e_inst: str,
    b_sys: str,
    e_sys: str,
    train_ratio: float = 0.8,
    with_basics: bool = True
) -> DatasetDict:
    """
    Convenience function to create a stock prediction dataset.
    
    This function wraps the StockPredictionDataProcessor and 
    StockPredictionDatasetBuilder classes for ease of use.
    """
    processor = StockPredictionProcessor(
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
        system_prompt=system_prompt,
        b_inst=b_inst,
        e_inst=e_inst,
        b_sys=b_sys,
        e_sys=e_sys
    )
    
    builder = StockPredictionDatasetBuilder(processor)
    return builder.create_dataset(symbol_list, train_ratio, with_basics)