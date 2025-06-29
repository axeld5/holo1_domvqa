from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
from tqdm import tqdm
import argparse
import os
from rl_rewards import compute_vqa_rewards, reward_boxed_answer, reward_answer_similarity
import numpy as np
import pandas as pd

def evaluate_single_model(
    model_path,
    eval_examples,
    model_name=None,
    temperature=0.1,
    max_new_tokens=256
):
    """
    Evaluate a single model on the provided examples.
    
    Args:
        model_path (str): Path to the model
        eval_examples (list): List of evaluation examples
        model_name (str): Display name for the model
        temperature (float): Temperature for generation
        max_new_tokens (int): Maximum tokens to generate
        
    Returns:
        dict: Evaluation results for this model
    """
    
    if model_name is None:
        model_name = os.path.basename(model_path)
    
    print(f"\nEvaluating model: {model_name}")
    print(f"Model path: {model_path}")
    
    # Load the model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None
    
    results = []
    rewards = []
    boxed_format_correct = 0
    similarity_scores = []
    
    for i, example in enumerate(tqdm(eval_examples, desc=f"Evaluating {model_name}")):
        try:
            # Get the user prompt and ground truth answer
            conversations = example["conversations"]
            user_prompt = None
            ground_truth = None
            
            for turn in conversations:
                if turn["role"] == "user":
                    user_prompt = turn["content"]
                elif turn["role"] == "assistant":
                    ground_truth = turn["content"]
            
            if not user_prompt or not ground_truth:
                continue
            
            # Generate model response
            messages = [{"role": "user", "content": user_prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False)
            model_inputs = tokenizer([input_text], return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode the response
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            model_response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            # Compute rewards
            boxed_reward = reward_boxed_answer(model_response)
            similarity = reward_answer_similarity(ground_truth, model_response)
            
            # Final reward (same logic as training)
            if boxed_reward == -1:
                final_reward = -1.0
            else:
                final_reward = similarity
                boxed_format_correct += 1
            
            rewards.append(final_reward)
            if final_reward != -1.0:
                similarity_scores.append(similarity)
            
            # Store detailed result
            result = {
                "example_id": i,
                "user_prompt": user_prompt,
                "ground_truth": ground_truth,
                "model_response": model_response,
                "boxed_reward": boxed_reward,
                "similarity": similarity,
                "final_reward": final_reward
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing example {i} for {model_name}: {e}")
            continue
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Compute metrics
    metrics = {
        "model_name": model_name,
        "total_examples": len(results),
        "avg_reward": np.mean(rewards) if rewards else 0,
        "boxed_format_accuracy": boxed_format_correct / len(results) if results else 0,
        "avg_similarity": np.mean(similarity_scores) if similarity_scores else 0,
        "reward_distribution": {
            "min": np.min(rewards) if rewards else 0,
            "max": np.max(rewards) if rewards else 0,
            "std": np.std(rewards) if rewards else 0
        }
    }
    
    return {
        "metrics": metrics,
        "results": results
    }

def evaluate_multiple_models(
    model_configs,
    dataset_file="vqa_rl_data.json",
    max_examples=None,
    temperature=0.1,
    max_new_tokens=256
):
    """
    Evaluate multiple models on the eval split of the dataset.
    
    Args:
        model_configs (list): List of dictionaries with 'path' and 'name' keys
        dataset_file (str): Path to the VQA dataset file
        max_examples (int): Maximum number of examples to evaluate (None for all)
        temperature (float): Temperature for generation
        max_new_tokens (int): Maximum tokens to generate
        
    Returns:
        dict: Evaluation results for all models
    """
    
    # Check if dataset exists
    if not os.path.exists(dataset_file):
        print(f"Dataset file {dataset_file} not found.")
        return None
    
    # Load the dataset
    print(f"Loading dataset from: {dataset_file}")
    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Filter for eval examples
    eval_examples = []
    for example in data:
        if example.get("split") == "eval":
            eval_examples.append(example)
    
    if max_examples:
        eval_examples = eval_examples[:max_examples]
    
    print(f"Found {len(eval_examples)} evaluation examples")
    
    if len(eval_examples) == 0:
        print("No evaluation examples found!")
        return None
    
    # Evaluate each model
    all_results = {}
    
    for config in model_configs:
        model_path = config["path"]
        model_name = config["name"]
        
        result = evaluate_single_model(
            model_path=model_path,
            eval_examples=eval_examples,
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        
        if result:
            all_results[model_name] = result
        else:
            print(f"Failed to evaluate model: {model_name}")
    
    return all_results

def print_comparative_summary(all_results):
    """Print a comparative summary of all model results."""
    if not all_results:
        print("No evaluation results to display.")
        return
    
    print("\n" + "="*80)
    print("COMPARATIVE EVALUATION SUMMARY")
    print("="*80)
    
    # Create comparison table
    comparison_data = []
    for model_name, result in all_results.items():
        metrics = result["metrics"]
        comparison_data.append({
            "Model": model_name,
            "Total Examples": metrics["total_examples"],
            "Avg Reward": f"{metrics['avg_reward']:.4f}",
            "Format Accuracy": f"{metrics['boxed_format_accuracy']:.2%}",
            "Avg Similarity": f"{metrics['avg_similarity']:.4f}",
            "Reward Std": f"{metrics['reward_distribution']['std']:.4f}"
        })
    
    # Print comparison table
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # Find best performing model
    best_model = max(all_results.keys(), 
                    key=lambda x: all_results[x]["metrics"]["avg_reward"])
    print(f"\nBest Overall Performance: {best_model}")
    print(f"Best Average Reward: {all_results[best_model]['metrics']['avg_reward']:.4f}")
    
    # Show detailed examples for each model
    print("\n" + "="*80)
    print("SAMPLE RESULTS BY MODEL")
    print("="*80)
    
    for model_name, result in all_results.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        
        results_list = result["results"]
        if results_list:
            # Show best example for this model
            best_example = max(results_list, key=lambda x: x["final_reward"])
            print(f"Best Example (Reward: {best_example['final_reward']:.4f}):")
            print(f"  Ground Truth: {best_example['ground_truth']}")
            print(f"  Model Response: {best_example['model_response']}")

def save_comparative_results(all_results, output_file):
    """Save detailed comparative results to a JSON file."""
    if not all_results:
        print("No results to save.")
        return
    
    print(f"Saving comparative results to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multiple VQA models on the eval split")
    parser.add_argument("--dataset_file", type=str, default="vqa_rl_data.json",
                        help="Path to the VQA dataset file")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to evaluate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--output_file", type=str, default="comparative_eval_results.json",
                        help="File to save detailed results")
    parser.add_argument("--save_results", action="store_true",
                        help="Save detailed results to file")
    
    args = parser.parse_args()
    
    # Configure models to evaluate
    model_configs = [
        {
            "path": "Qwen/Qwen2.5-VL-3B-Instruct",
            "name": "Qwen2.5-VL-3B-Instruct"
        },
        {
            "path": "HCompany/Holo1-3B",
            "name": "Holo1-3B"
        },
        {
            "path": "holo1-3b-vqa-rl",  # Default RL-trained model path
            "name": "Holo1-3B-VQA-RL"
        }
    ]
    
    print("Models to evaluate:")
    for config in model_configs:
        print(f"  - {config['name']}: {config['path']}")
    
    # Run comparative evaluation
    all_results = evaluate_multiple_models(
        model_configs=model_configs,
        dataset_file=args.dataset_file,
        max_examples=args.max_examples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )
    
    if all_results:
        # Print comparative summary
        print_comparative_summary(all_results)
        
        # Save detailed results if requested
        if args.save_results:
            save_comparative_results(all_results, args.output_file)
    else:
        print("Evaluation failed!") 