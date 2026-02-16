import re
import os
import datasets
from sklearn.metrics import accuracy_score, mean_squared_error
from collections import defaultdict
from rouge_score import rouge_scorer


lora_module_dict = {
    'chatglm2': ['query_key_value'],
    'llama3': [
        'q_proj', 'k_proj', 'v_proj',
        'o_proj', 'gate_proj', 'up_proj', 'down_proj',
        # 'embed_tokens', 'lm_head',
    ],
    'deepseek-llama8b': [
        'q_proj', 'k_proj', 'v_proj',
    ]
}


def tokenize(args, tokenizer, feature):
    
    prompt_ids = tokenizer.encode(
        feature['prompt'].strip(), padding=False,
        max_length=args.max_length, truncation=True
    )
    
    target_ids = tokenizer.encode(
        feature['answer'].strip(), padding=False,
        max_length=args.max_length, truncation=True, add_special_tokens=False
    )
    
    input_ids = prompt_ids + target_ids
    exceed_max_length = len(input_ids) >= args.max_length
    
     # Add EOS Token
    if input_ids[-1] != tokenizer.eos_token_id and not exceed_max_length:
        input_ids.append(tokenizer.eos_token_id)
    
    label_ids = [tokenizer.pad_token_id] * len(prompt_ids) + input_ids[len(prompt_ids):]
    
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "exceed_max_length": exceed_max_length
    }


def parse_model_name(name, from_remote=False):
    
    if name == 'chatglm2':
        return 'THUDM/chatglm2-6b' if from_remote else 'base_models/chatglm2-6b'
    elif name == 'llama2':
        return 'meta-llama/Llama-2-7b-chat-hf' # if from_remote else 'base_models/Llama-2-7b-chat-hf'
    elif name == 'deepseek-llama8b':
        return 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' if from_remote else 'base_models/DeepSeek-R1-Distill-Llama-8B'
    else:
        raise ValueError(f"Undefined base model {name}")
        
    
def load_dataset(names, from_remote=False):
    """
    load_dataset
        Load dataset from Hugging Face or local disk cache. If the dataset does not exist in local cache, it will be downloaded from Hugging Face and saved to local cache for future use. If the dataset does not exist in Hugging Face, it will raise an error.
        
    :param names: A comma-separated string of dataset names. Each name can be optionally followed by '*N' to indicate that the dataset should be repeated N times in the returned list. 
    :param from_remote: If True, forces loading the dataset from Hugging Face even if it exists in local cache. If False, it will try to load from local cache first before falling back to Hugging Face.
    :return: A list of datasets corresponding to the provided names, with repetitions as specified.
    """
    
    dataset_names = [d for d in names.split(',')]
    dataset_list = []

    LOCAL_DATA_DIR = './dataset_cache/'
    
    for origin_name in dataset_names:
        rep = 1
        local_exist = False

        if '*' in origin_name:
            rep = int(origin_name.split('*')[1]) if '*' in origin_name else 1
            cleaned_name = origin_name.split('*')[0]
        else:
            cleaned_name = origin_name

        local_dir = f"fingpt-forecaster-{cleaned_name}"
        cache_path = os.path.join(LOCAL_DATA_DIR, local_dir)
        local_exist = os.path.exists(cache_path)
        tmp_dataset = None
        if from_remote or not local_exist:
            try:
                full_name = f"FinGPT/fingpt-forecaster-{cleaned_name}"
                tmp_dataset = datasets.load_dataset(full_name)
                tmp_dataset.save_to_disk(cache_path)
            except Exception as e:
                print(f"Failed to load dataset from Hugging Face at {full_name}: {e}")
        elif local_exist:
            try:
                tmp_dataset = datasets.load_from_disk(cache_path)
            except Exception as e:
                print(f"Failed to load dataset from disk at {cache_path}: {e}")
        if tmp_dataset is None:
            raise ValueError(f"Dataset {origin_name} could not be loaded from either Hugging Face or local disk.")

        if 'test' not in tmp_dataset:
            tmp_dataset = tmp_dataset.train_test_split(0.2, shuffle=True, seed=42)   
        dataset_list.extend([tmp_dataset] * rep)
    
    return dataset_list


def parse_answer(answer):
    
    match_res = re.match(r"^\s*\[Positive Developments\]:\s*(.*)\s*\[Potential Concerns\]:\s*(.*)\s*\[Prediction (&|and) Analysis\]:\s*(.*)\s*$", answer, flags=re.DOTALL)
    if not match_res:
        return None
    
    pros, cons, pna = match_res.group(1), match_res.group(2), match_res.group(4)
        
    match_res = re.match(r'^Prediction:\s*(.*)\s*Analysis:\s*(.*)\s*$', pna, flags=re.DOTALL)
    if not match_res:
        return None
        
    pred, anal = match_res.group(1), match_res.group(2)
        
    if re.search(r'up|increase', pred.lower()):
        pred_bin = 1
    elif re.search(r'down|decrease|decline', pred.lower()):
        pred_bin = -1
    else:
        pred_bin = 0
            
    match_res = re.search(r'(\d)-(\d)%', pred)
    if not match_res:
        match_res = re.search(r'(?:more than )?(\d)+?%', pred)    
        
    pred_margin = pred_bin * (int(match_res.group(1)) + 0.5) if match_res else 0.
        
    return {
        "positive developments": pros,
        "potential concerns": cons,
        "prediction": pred_margin,
        "prediction_binary": pred_bin,
        "analysis": anal
    }
    

def calc_rouge_score(references, answers):
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    scores_per_pair = [scorer.score(ref, ans) for ref, ans in zip(references, answers)]
    
    rouge1 = sum(score['rouge1'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    rouge2 = sum(score['rouge2'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    rougeL = sum(score['rougeL'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    
    return {'rouge1': rouge1, 'rouge2': rouge2, 'rougeL': rougeL}

    
def calc_metrics(answers, gts):
    
    answers_dict = defaultdict(list)
    gts_dict = defaultdict(list)
    
    for answer, gt in zip(answers, gts):
        answer_dict = parse_answer(answer)
        gt_dict = parse_answer(gt)
        
        if answer_dict and gt_dict:
            for k in answer_dict.keys():
                answers_dict[k].append(answer_dict[k])
                gts_dict[k].append(gt_dict[k])
    
    if not answers_dict['prediction']:
        return {}
    
    bin_acc = accuracy_score(gts_dict['prediction_binary'], answers_dict['prediction_binary'])
    mse = mean_squared_error(gts_dict['prediction'], answers_dict['prediction'])
    
    pros_rouge_scores = calc_rouge_score(gts_dict['positive developments'], answers_dict['positive developments'])
    cons_rouge_scores = calc_rouge_score(gts_dict['potential concerns'], answers_dict['potential concerns'])
    anal_rouge_scores = calc_rouge_score(gts_dict['analysis'], answers_dict['analysis'])
                              
    print(f"\nBinary Accuracy: {bin_acc:.2f}  |  Mean Square Error: {mse:.2f}")
    print(f"\nRouge Score of Positive Developments: {pros_rouge_scores}")
    print(f"\nRouge Score of Potential Concerns: {cons_rouge_scores}")
    print(f"\nRouge Score of Summary Analysis: {anal_rouge_scores}")
                              
    return {
        "valid_count": len(answers_dict['prediction']),
        "bin_acc": bin_acc,
        "mse": mse,
        "pros_rouge_scores": pros_rouge_scores,
        "cons_rouge_scores": cons_rouge_scores,
        "anal_rouge_scores": anal_rouge_scores
    }
    