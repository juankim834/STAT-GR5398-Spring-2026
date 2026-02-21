import re
import time
from datasets import Dataset
from tqdm import tqdm


def filter_by_ticker(test_dataset, ticker_code):
    """
    Filter a dataset to include only rows containing a specific ticker symbol.
    
    This function searches through the 'prompt' field of each row for a ticker symbol
    matching the pattern "ticker [SYMBOL]" and returns a new dataset containing only
    matching rows.
    
    Args:
        test_dataset (Dataset): A HuggingFace Dataset object containing a 'prompt' column
        ticker_code (str): The ticker symbol to filter by (e.g., 'AAPL', 'MSFT')
    
    Returns:
        Dataset: A new HuggingFace Dataset containing only rows with the specified ticker
    
    Example:
        >>> from datasets import Dataset
        >>> data = Dataset.from_dict({
        ...     'prompt': ['Analyze ticker AAPL', 'Analyze ticker MSFT', 'Analyze ticker AAPL'],
        ...     'answer': ['Up', 'Down', 'Up']
        ... })
        >>> filtered = filter_by_ticker(data, 'AAPL')
        >>> len(filtered)
        2
    """
    filtered_data = []

    for row in test_dataset:
        prompt_content = row['prompt']
        ticker_symbol = re.search(r"ticker\s([A-Z]+)", prompt_content)

        if ticker_symbol and ticker_symbol.group(1) == ticker_code:
            filtered_data.append(row)

    filtered_dataset = Dataset.from_dict({
        key: [row[key] for row in filtered_data] 
        for key in test_dataset.column_names
    })

    return filtered_dataset


def get_unique_ticker_symbols(test_dataset):
    """
    Extract all unique ticker symbols from a dataset.
    
    Searches through all 'prompt' fields in the dataset for ticker symbols matching
    the pattern "ticker [SYMBOL]" and returns a list of unique symbols found.
    
    Args:
        test_dataset (Dataset): A HuggingFace Dataset object containing a 'prompt' column
    
    Returns:
        list: A list of unique ticker symbols (strings) found in the dataset
    
    Example:
        >>> from datasets import Dataset
        >>> data = Dataset.from_dict({
        ...     'prompt': ['Analyze ticker AAPL', 'Analyze ticker MSFT', 'Analyze ticker AAPL']
        ... })
        >>> symbols = get_unique_ticker_symbols(data)
        >>> sorted(symbols)
        ['AAPL', 'MSFT']
    """
    ticker_symbols = set()

    for i in range(len(test_dataset)):
        prompt_content = test_dataset[i]['prompt']
        ticker_symbol = re.search(r"ticker\s([A-Z]+)", prompt_content)

        if ticker_symbol:
            ticker_symbols.add(ticker_symbol.group(1))

    return list(ticker_symbols)


def insert_guidance_after_intro(prompt):
    """
    Restructure a prompt by moving guidance section immediately after the introduction.
    
    This function reorganizes prompts by extracting the guidance section (text between
    "Based on all the information before" and "Following these instructions") and
    placing it immediately after the system introduction, before the actual content.
    
    Args:
        prompt (str): The original prompt string to restructure
    
    Returns:
        str: The restructured prompt with guidance moved after the introduction,
             or the original prompt if required markers are not found
    
    Example:
        >>> prompt = '''[INST]<<SYS>>
        ... You are a seasoned stock market analyst. Your task is to list the positive developments...
        ... [Content here]
        ... Based on all the information before, analyze carefully...
        ... Following these instructions, please come up with 2-4 most important positive factors'''
        >>> new_prompt = insert_guidance_after_intro(prompt)
        >>> # The guidance section will now appear right after the system prompt
    """
    intro_marker = (
        "[INST]<<SYS>>\n"
        "You are a seasoned stock market analyst. Your task is to list the positive developments and "
        "potential concerns for companies based on relevant news and basic financials from the past weeks, "
        "then provide an analysis and prediction for the companies' stock price movement for the upcoming week."
    )
    guidance_start_marker = "Based on all the information before"
    guidance_end_marker = "Following these instructions, please come up with 2-4 most important positive factors"

    intro_pos = prompt.find(intro_marker)
    guidance_start_pos = prompt.find(guidance_start_marker)
    guidance_end_pos = prompt.find(guidance_end_marker)

    if intro_pos == -1 or guidance_start_pos == -1 or guidance_end_pos == -1:
        return prompt

    guidance_section = prompt[guidance_start_pos:guidance_end_pos].strip()

    new_prompt = (
        f"{prompt[:intro_pos + len(intro_marker)]}\n\n"
        f"{guidance_section}\n\n"
        f"{prompt[intro_pos + len(intro_marker):guidance_start_pos]}"
        f"{prompt[guidance_end_pos:]}"
    )

    return new_prompt


def apply_to_all_prompts_in_dataset(test_dataset):
    """
    Apply prompt restructuring to all prompts in a dataset.
    
    Uses the insert_guidance_after_intro function to restructure every prompt
    in the dataset's 'prompt' column.
    
    Args:
        test_dataset (Dataset): A HuggingFace Dataset containing a 'prompt' column
    
    Returns:
        Dataset: A new Dataset with all prompts restructured
    
    Example:
        >>> from datasets import Dataset
        >>> data = Dataset.from_dict({'prompt': [original_prompt1, original_prompt2]})
        >>> updated_data = apply_to_all_prompts_in_dataset(data)
        >>> # All prompts in updated_data will have restructured guidance sections
    """
    updated_dataset = test_dataset.map(
        lambda x: {"prompt": insert_guidance_after_intro(x["prompt"])}
    )

    return updated_dataset


def test_demo(model, tokenizer, prompt):
    """
    Generate a response from a model for a given prompt and measure inference time.
    
    Args:
        model: The language model to use for generation
        tokenizer: The tokenizer corresponding to the model
        prompt (str): The input prompt to generate a response for
    
    Returns:
        tuple: (output_text, inference_time)
            - output_text (str): The generated response with special tokens removed
            - inference_time (float): Time taken for generation in seconds
    
    Example:
        >>> output, time_taken = test_demo(model, tokenizer, "Analyze ticker AAPL")
        >>> print(f"Generated in {time_taken:.2f}s: {output[:50]}...")
    """
    inputs = tokenizer(
        prompt, return_tensors='pt',
        padding=False, max_length=8000
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    start_time = time.time()
    res = model.generate(
        **inputs, max_length=4096, do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True
    )
    end_time = time.time()
    output = tokenizer.decode(res[0], skip_special_tokens=True)
    
    return output, end_time - start_time


def test_acc(test_dataset, modelname, llm_model_dict: dict):
    """
    Evaluate and compare base and fine-tuned model performance on a test dataset.
    
    Runs inference on both a base model and its fine-tuned version for all samples
    in the dataset, collecting predictions and timing information.
    
    Args:
        test_dataset (Dataset): HuggingFace Dataset with 'prompt' and 'answer' columns
        modelname (str): Name of the model to test ('llama3', 'deepseek', or custom)
        llm_model_dict (dict): A dictionary containing pre-loaded models and tokenizers
            with keys following the pattern:
            - '{modelname}_base_model'
            - '{modelname}_model'
            - '{modelname}_tokenizer'
    
    Returns:
        tuple: (answers_base, answers_fine_tuned, gts, times_base, times_fine_tuned)
            - answers_base (list): Predictions from the base model
            - answers_fine_tuned (list): Predictions from the fine-tuned model
            - gts (list): Ground truth answers
            - times_base (list): Inference times for base model
            - times_fine_tuned (list): Inference times for fine-tuned model
    
    Note:
        Requires pre-loaded models and tokenizers with specific naming conventions:
        - llama3: llama3_base_model, llama3_model, llama3_tokenizer
        - deepseek: deepseek_base_model, deepseek_model, deepseek_tokenizer
        - Custom models can be added by extending the elif conditions
    
    Example:
        >>> # After loading models:
        >>> # llama3_base_model = AutoModelForCausalLM.from_pretrained(...)
        >>> # llama3_model = AutoModelForCausalLM.from_pretrained(...)
        >>> # llama3_tokenizer = AutoTokenizer.from_pretrained(...)
        >>> # llm_model_dict = {
        ... #     'llama3_base_model': llama3_base_model,
        ... #     'llama3_model': llama3_model,
        ... #     'llama3_tokenizer': llama3_tokenizer,
        ... # }
        >>> 
        >>> base_ans, ft_ans, ground_truth, base_times, ft_times = test_acc(test_dataset, 'llama3', llm_model_dict)
        >>> print(f"Tested {len(base_ans)} samples")
        >>> print(f"Avg base time: {sum(base_times)/len(base_times):.2f}s")
        >>> print(f"Avg fine-tuned time: {sum(ft_times)/len(ft_times):.2f}s")
    """
    answers_base, answers_fine_tuned, gts, times_base, times_fine_tuned = [], [], [], [], []
    
    if modelname == "llama3":
        base_model = llm_model_dict['llama3_base_model']
        model = llm_model_dict['llama3_model']
        tokenizer = llm_model_dict['llama3_tokenizer']
    elif modelname == "deepseek":
        base_model = llm_model_dict['deepseek_base_model']
        model = llm_model_dict['deepseek_model']
        tokenizer = llm_model_dict['deepseek_tokenizer']
    elif modelname == "your_model_name":  # Add other models as needed
        base_model = llm_model_dict['your_model_base_model']
        model = llm_model_dict['your_model_finetuned_model']
        tokenizer = llm_model_dict['your_model_tokenizer']

    for i in tqdm(range(len(test_dataset)), desc="Processing test samples"):
        try:
            prompt = test_dataset[i]['prompt']
            gt = test_dataset[i]['answer']

            output_base, time_base = test_demo(base_model, tokenizer, prompt)
            answer_base = re.sub(r'.*\[/INST\]\s*', '', output_base, flags=re.DOTALL)

            output_fine_tuned, time_fine_tuned = test_demo(model, tokenizer, prompt)
            answer_fine_tuned = re.sub(r'.*\[/INST\]\s*', '', output_fine_tuned, flags=re.DOTALL)

            answers_base.append(answer_base)
            answers_fine_tuned.append(answer_fine_tuned)
            gts.append(gt)
            times_base.append(time_base)
            times_fine_tuned.append(time_fine_tuned)

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            
    return answers_base, answers_fine_tuned, gts, times_base, times_fine_tuned


# ============================================================================
# COMPLETE USAGE EXAMPLE
# ============================================================================

"""
Complete workflow example for stock market prediction model evaluation:

# 1. Load your dataset
from datasets import load_dataset
test_dataset = load_dataset('your_dataset_name')['test']

# 2. Preprocess: Apply prompt restructuring
test_dataset = apply_to_all_prompts_in_dataset(test_dataset)

# 3. Get overview of tickers in dataset
unique_symbols = get_unique_ticker_symbols(test_dataset)
print(f"Dataset contains {len(unique_symbols)} unique ticker symbols: {unique_symbols}")

# 4. Optional: Filter by specific ticker
aapl_dataset = filter_by_ticker(test_dataset, 'AAPL')
print(f"Found {len(aapl_dataset)} samples for AAPL")

# 5. Load your models (example with Llama3)
from transformers import AutoModelForCausalLM, AutoTokenizer

llama3_base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3-8B')
llama3_model = AutoModelForCausalLM.from_pretrained('your-org/llama3-finetuned-stocks')
llama3_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3-8B')

# 6. Run evaluation
base_ans, ft_ans, ground_truth, base_times, ft_times = test_acc(test_dataset, 'llama3')

# 7. Analyze results
import numpy as np
print(f"Average base model inference time: {np.mean(base_times):.2f}s")
print(f"Average fine-tuned model inference time: {np.mean(ft_times):.2f}s")

# Calculate accuracy (if answers are categorical like 'Up'/'Down')
base_correct = sum(1 for pred, gt in zip(base_ans, ground_truth) if pred.strip() == gt.strip())
ft_correct = sum(1 for pred, gt in zip(ft_ans, ground_truth) if pred.strip() == gt.strip())

print(f"Base model accuracy: {base_correct/len(ground_truth)*100:.2f}%")
print(f"Fine-tuned model accuracy: {ft_correct/len(ground_truth)*100:.2f}%")
"""