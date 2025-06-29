import difflib
import re

def reward_boxed_answer(target_text):
    """
    Check if '\\boxed{...}' is present in the text and if the content inside the braces has less than 5 words.
    Return -1 if not matched or if content has 5 or more words, else 0.
    """
    match = re.search(r"\\boxed\{([^}]*)\}", target_text)
    if not match:
        return -1
    content = match.group(1).strip()
    if len(content.split()) >= 5:
        return -1
    return 0

def reward_answer_similarity(source_text, target_text):
    """
    Compute similarity between ground truth and extracted answer using difflib SequenceMatcher.
    
    Args:
        source_text (str): Ground truth answer
        target_text (str): Model's extracted answer
    
    Returns:
        float: Similarity ratio between 0 and 1
    """
    # Optionally, extract answer from \boxed{} if present
    match = re.search(r"\\boxed\{([^}]*)\}", target_text)
    if match:
        extracted = match.group(1).strip()
    else:
        extracted = target_text.strip()
    return difflib.SequenceMatcher(None, source_text.strip(), extracted).ratio()

def compute_vqa_rewards(prompts, completions, **kwargs):
    """
    Reward is -1 if reward_boxed_answer is -1, else reward is answer similarity.
    
    Args:
        prompts: List of prompts from GRPO trainer  
        completions: List of completions from GRPO trainer
    
    Returns:
        List[float]: Reward scores for each completion
    """
    scores = []
    
    for i, completion_batch in enumerate(completions):
        # Extract the source answer from the prompt
        answer = prompts[i]["conversations"][1]["content"]
        source_text = answer
        
        # Extract the model's answer from completion
        target_text = completion_batch[0]["content"].strip()
        
        # Compute reward
        boxed_reward = reward_boxed_answer(target_text)
        if boxed_reward == -1:
            reward = -1.0
        else:
            reward = reward_answer_similarity(source_text, target_text)
        
        scores.append(reward)
    
    return scores