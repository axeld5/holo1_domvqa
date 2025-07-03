import torch
import json
import gc
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from dotenv import load_dotenv
from typing import Dict, List, Tuple
import numpy as np
import random

load_dotenv()

def clear_memory():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def setup_quantization():
    """Setup 4-bit quantization to reduce memory usage"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    return bnb_config

def find_target_modules(model):
    """Find all linear layers in the model for LoRA targeting"""
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Get the module name without the base model prefix
            if "model." in name:
                module_name = name.split(".")[-1]  # Get the last part
                target_modules.add(module_name)
    
    # Filter to common attention and MLP layers
    common_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found_modules = list(target_modules.intersection(common_modules))
    
    if not found_modules:
        # Fallback to all linear layers if common ones not found
        found_modules = list(target_modules)[:8]  # Limit to prevent too many modules
    
    print(f"Found target modules: {found_modules}")
    return found_modules

def setup_lora_config(target_modules=None):
    """Setup LoRA configuration for parameter-efficient fine-tuning"""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Conservative default
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Increased rank for better performance
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
        modules_to_save=None,
    )
    return lora_config

def load_holo1_model_optimized():
    """Load Holo1 model with memory optimizations"""
    print("Loading HCompany/Holo1 with memory optimizations...")
    
    # Setup quantization
    bnb_config = setup_quantization()
    
    # Load model with quantization
    model_id = "HCompany/Holo1"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training (required for quantized models)
    model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Find target modules for LoRA
    target_modules = find_target_modules(model)
    
    # Setup LoRA
    lora_config = setup_lora_config(target_modules)
    model = get_peft_model(model, lora_config)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    print(f"Model loaded with {model.num_parameters()} total parameters")
    print(f"Trainable parameters: {model.num_parameters(only_trainable=True)}")
    
    clear_memory()
    return model, processor

def load_webcode2m_dataset(num_samples=1000):
    """Load and prepare the webcode2m_purified dataset"""
    print(f"Loading webcode2m_purified dataset with {num_samples} samples...")
    
    # Load the dataset
    dataset = load_dataset("xcodemind/webcode2m_purified", split="train")
    
    # Sample the specified number of elements
    if len(dataset) > num_samples:
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = dataset.select(indices)
    
    print(f"Dataset loaded with {len(dataset)} samples")
    return dataset

def preprocess_dataset(dataset):
    """Preprocess the dataset for training"""
    def process_example(example):
        return {
            "image": example["image"],
            "question": "convert this image to html",
            "answer": example["text"]  # HTML code from the text column
        }
    
    processed_dataset = dataset.map(process_example, remove_columns=dataset.column_names)
    return processed_dataset

def get_image_token_id(processor):
    """Get image token ID for Holo1 model"""
    try:
        # Try common image tokens
        if hasattr(processor.tokenizer, 'additional_special_tokens'):
            for token in processor.tokenizer.additional_special_tokens:
                if 'image' in token.lower():
                    return processor.tokenizer.convert_tokens_to_ids(token)
        
        # Fallback to additional special tokens
        return processor.tokenizer.additional_special_tokens_ids[0] if processor.tokenizer.additional_special_tokens_ids else None
    except:
        return None

def collate_fn_factory(processor, image_token_id):
    """Factory function to create collate_fn for Holo1 with memory optimization"""
    
    def collate_fn(examples):
        # Process in smaller chunks to avoid OOM
        batch_size = len(examples)
        if batch_size > 2:  # If batch is too large, process in chunks
            mid = batch_size // 2
            chunk1 = collate_fn(examples[:mid])
            chunk2 = collate_fn(examples[mid:])
            
            # Combine chunks
            combined = {}
            for key in chunk1.keys():
                if key in chunk2:
                    combined[key] = torch.cat([chunk1[key], chunk2[key]], dim=0)
                else:
                    combined[key] = chunk1[key]
            return combined
        
        texts = []
        images = []
        
        for example in examples:
            image = example["image"]
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image to reduce memory usage
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            question = example["question"]
            answer = example["answer"]
            
            # Format for Holo1 model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image"},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
            
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        try:
            batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
            labels = batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            if image_token_id is not None:
                labels[labels == image_token_id] = -100
            batch["labels"] = labels
            return batch
        except torch.cuda.OutOfMemoryError:
            print("OOM in collate_fn, clearing memory and retrying with smaller batch...")
            clear_memory()
            # Try with just one example if we still have OOM
            if len(examples) > 1:
                return collate_fn(examples[:1])
            else:
                raise
    
    return collate_fn

def generate_response(model, processor, image: Image.Image, question: str) -> str:
    """Generate a response from the model for a given image and question."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Format for Holo1 model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,  # Increased for HTML generation
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode only the generated part
    generated_text = processor.batch_decode(
        generated_ids[:, inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )[0]
    
    return generated_text.strip()

def evaluate_model_subset(model, processor, test_dataset, max_samples: int = 10) -> Dict[str, float]:
    """Evaluate model on a subset of the test dataset"""
    print(f"Evaluating model on {max_samples} samples...")
    
    # Sample a subset for evaluation
    if len(test_dataset) > max_samples:
        indices = random.sample(range(len(test_dataset)), max_samples)
        eval_dataset = test_dataset.select(indices)
    else:
        eval_dataset = test_dataset
    
    total_samples = len(eval_dataset)
    successful_generations = 0
    
    for i, example in enumerate(eval_dataset):
        try:
            image = example["image"]
            question = example["question"]
            expected_answer = example["answer"]
            
            print(f"Evaluating sample {i+1}/{total_samples}")
            
            # Generate response
            generated_answer = generate_response(model, processor, image, question)
            
            # Check if generation was successful (non-empty and reasonable length)
            if generated_answer and len(generated_answer) > 10:
                successful_generations += 1
                print(f"✓ Generated HTML (length: {len(generated_answer)})")
            else:
                print(f"✗ Failed to generate valid HTML")
            
            # Clear memory after each evaluation
            clear_memory()
            
        except Exception as e:
            print(f"Error evaluating sample {i+1}: {e}")
            continue
    
    success_rate = successful_generations / total_samples
    print(f"Success rate: {success_rate:.2%} ({successful_generations}/{total_samples})")
    
    return {
        "success_rate": success_rate,
        "total_samples": total_samples,
        "successful_generations": successful_generations
    }

def main():
    """Main training function"""
    print("Starting Holo1 DOM finetuning...")
    
    # Load model and processor
    model, processor = load_holo1_model_optimized()
    
    # Load and preprocess dataset
    dataset = load_webcode2m_dataset(num_samples=1000)
    processed_dataset = preprocess_dataset(dataset)
    
    # Split dataset (80% train, 20% test)
    train_test_split = processed_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Get image token ID
    image_token_id = get_image_token_id(processor)
    print(f"Image token ID: {image_token_id}")
    
    # Create collate function
    collate_fn = collate_fn_factory(processor, image_token_id)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./holo1_dom_finetune_output",
        per_device_train_batch_size=1,  # Small batch size for memory efficiency
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Simulate larger batch size
        num_train_epochs=3,
        learning_rate=1e-4,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
    )
    
    # Start training
    print("Starting training...")
    try:
        trainer.train()
        print("Training completed successfully!")
        
        # Save the final model
        print("Saving final model...")
        trainer.save_model("./holo1_dom_finetune_final")
        processor.save_pretrained("./holo1_dom_finetune_final")
        
        # Evaluate the model
        print("Evaluating final model...")
        eval_results = evaluate_model_subset(model, processor, test_dataset, max_samples=10)
        
        print(f"\nFinal evaluation results:")
        print(f"Success rate: {eval_results['success_rate']:.2%}")
        print(f"Successful generations: {eval_results['successful_generations']}/{eval_results['total_samples']}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        # Save checkpoint even if training fails
        trainer.save_model("./holo1_dom_finetune_checkpoint")
        processor.save_pretrained("./holo1_dom_finetune_checkpoint")
        raise

if __name__ == "__main__":
    main() 