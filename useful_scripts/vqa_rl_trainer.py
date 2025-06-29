from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForImageTextToText
import json
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from rl_rewards import compute_vqa_rewards
import argparse
import os

def train_vqa_rl_model(
    model_name="HCompany/Holo1-3B", 
    max_steps=1000, 
    save_path="holo1-3b-vqa-rl", 
    dataset_file="vqa_rl_data.json"
):
    """
    Train a VQA model using GRPO (Generative Reinforcement Policy Optimization).
    
    Args:
        model_name (str): Name or path of the model to fine-tune
        max_steps (int): Maximum number of training steps
        save_path (str): Path to save the fine-tuned model
        dataset_file (str): Path to the VQA dataset file
        
    Returns:
        dict: Training statistics
    """
    # Check if dataset exists
    if not os.path.exists(dataset_file):
        print(f"Dataset file {dataset_file} not found. Please run prepare_dataset.py first.")
        return None
    
    # Load the dataset
    print(f"Loading dataset from: {dataset_file}")
    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Prepare dataset for GRPO training
    rows = []
    for example in data:
        if example["split"] == "train":
            for turn in example["conversations"]:
                if turn["role"] == "user":
                    question = turn["content"].strip()
                    rows.append({"question": question, "prompt": example["conversations"]})
                    break

    dataset = Dataset.from_list(rows)
    print(f"Prepared {len(rows)} training examples")
    
    # Load the model
    print(f"Loading model: {model_name}")
    max_seq_length = 33768  # Increased for longer DOM contexts
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure training parameters for 13B model
    max_prompt_length = 32768  # Increased for DOM content
    training_args = GRPOConfig(
        learning_rate=5e-6,  # Lower learning rate for larger model
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        logging_steps=10,
        per_device_train_batch_size=1,  # Smaller batch size for 13B model
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size
        num_generations=4,  # Reduced for larger model
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        num_train_epochs=1,
        max_steps=max_steps,
        save_steps=200,
        max_grad_norm=1.0,
        report_to="none",
        output_dir="outputs",
        remove_unused_columns=False,
        bf16=True,  # Use bf16 for better performance with large models
    )
    
    # Use VQA reward function
    reward_func = compute_vqa_rewards
    
    # Create the trainer
    print("Creating GRPO trainer")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train the model
    print(f"Starting VQA RL training for {max_steps} steps")
    print("Reward structure:")
    print("- Boxed answer format: -1 if no \\boxed{} or content has 5+ words, 0 otherwise")
    print("- Answer similarity: Similarity score between ground truth and extracted answer")
    print("- Final reward: -1 if format is wrong, else similarity score")
    
    trainer_stats = trainer.train()
    
    # Save the model
    print(f"Saving model to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("VQA RL training completed successfully!")
    return trainer_stats

def test_vqa_model(model_path, test_examples=None):
    """
    Test the trained VQA model on some example DOM questions.
    
    Args:
        model_path (str): Path to the trained model
        test_examples (list): List of test examples with DOM and questions
    """
    if test_examples is None:
        test_examples = [
            {
                "dom": "<html><body><h1>Welcome</h1><p>This is a test page.</p></body></html>",
                "question": "What is the main heading of the page?"
            },
            {
                "dom": "<html><body><div class='price'>$29.99</div><button>Buy Now</button></body></html>",
                "question": "What is the price displayed?"
            },
        ]
    
    print(f"Loading trained model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    print("\nTesting VQA model:")
    for i, example in enumerate(test_examples):
        dom = example["dom"]
        question = example["question"]
        
        prompt = f"""Given the following DOM of a page, answer the question that is asked.
<dom>{dom}</dom>
Question: {question}
Your answer must be a boolean, a word or a number, contained within $\\boxed{{}}$. Now answer the question.
Answer:"""
        
        messages = [{"role": "user", "content": prompt}]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = tokenizer([input_text], return_tensors="pt")
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        print(f"Example {i+1}:")
        print(f"Question: {question}")
        print(f"Answer: {output}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VQA model using GRPO")
    parser.add_argument("--model_name", type=str, default="HCompany/Holo1-3B", 
                        help="Name or path of the model to fine-tune")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of training steps")
    parser.add_argument("--save_path", type=str, default="holo1-3b-vqa-rl", 
                        help="Path to save the fine-tuned model")
    parser.add_argument("--dataset_file", type=str, default="vqa_rl_data.json",
                        help="Path to the VQA dataset file")
    parser.add_argument("--test_only", action="store_true", help="Only test a trained model")
    
    args = parser.parse_args()
    
    if args.test_only:
        test_vqa_model(args.save_path)
    else:
        # Run the training
        train_vqa_rl_model(
            model_name=args.model_name,
            max_steps=args.max_steps,
            save_path=args.save_path,
            dataset_file=args.dataset_file
        ) 