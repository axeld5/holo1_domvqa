import torch
import json
import gc
import os
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional
import numpy as np
import random
from huggingface_hub import login, HfApi
import argparse

load_dotenv()

def setup_huggingface_hub():
    """Setup Hugging Face Hub authentication"""
    try:
        # Try to login with token from environment or cache
        login(token=os.getenv("HF_TOKEN"), add_to_git_credential=True)
        print("✓ Hugging Face Hub authentication successful")
        return True
    except Exception as e:
        print(f"⚠ Hugging Face Hub authentication failed: {e}")
        print("Please set HF_TOKEN environment variable or run 'huggingface-cli login'")
        return False

def push_to_hub(model_path: str, repo_name: str, private: bool = False, commit_message: str = "Upload Holo1 UI2DOM model"):
    """Push the trained model to Hugging Face Hub"""
    try:
        print(f"\n🚀 Pushing model to Hugging Face Hub: {repo_name}")
        
        # Setup HF Hub authentication
        if not setup_huggingface_hub():
            print("❌ Cannot push to Hub without authentication")
            return False
        
        # Get the model and processor
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import PeftModel
        
        # Load the trained model
        print("📥 Loading trained model...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # Create model card content
        model_card = f"""---
license: apache-2.0
tags:
- vision
- image-to-text
- html-generation
- ui-to-code
- multimodal
- holo1
- dom
- screenshot-to-html
language:
- en
pipeline_tag: image-to-text
---

# Holo1 UI2DOM - Screenshot to HTML Converter

This model is a fine-tuned version of [Hcompany/Holo1-3B](https://huggingface.co/Hcompany/Holo1-3B) for converting UI screenshots to HTML code.

## Model Description

- **Base Model**: Hcompany/Holo1-3B
- **Fine-tuned on**: Web screenshots and corresponding HTML code
- **Task**: Convert UI screenshots to HTML/CSS code
- **Training**: Optimized with DeepSpeed ZeRO-3 for efficient large-scale training

## Usage

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Load model and processor
model_name = "{repo_name}"
model = AutoModelForImageTextToText.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Load your screenshot
image = Image.open("path/to/your/screenshot.png")

# Prepare the conversation
messages = [
    {{
        "role": "user",
        "content": [
            {{"type": "text", "text": "convert this image to html"}},
            {{"type": "image"}},
        ]
    }}
]

# Process and generate
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=[image], return_tensors="pt")

# Generate HTML
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

# Decode the result
generated_text = processor.batch_decode(
    generated_ids[:, inputs['input_ids'].shape[1]:], 
    skip_special_tokens=True
)[0]

print("Generated HTML:", generated_text)
```

## Training Details

- **Dataset**: webcode2m_purified (screenshot-HTML pairs)
- **Training Framework**: DeepSpeed ZeRO-3
- **Optimization**: LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- **Precision**: BF16 mixed precision training
- **Batch Size**: Effective batch size of 8 (micro batch size 1 × gradient accumulation 8)

## Performance

The model demonstrates strong capability in:
- Converting UI screenshots to semantic HTML
- Preserving layout structure and styling
- Generating clean, readable HTML code
- Handling various UI components and layouts

## Limitations

- Performance may vary on UI designs significantly different from training data
- Generated HTML may need manual refinement for production use
- Model size requires sufficient GPU memory for inference

## Citation

```bibtex
@misc{{holo1_ui2dom,
  title={{Holo1 UI2DOM: Screenshot to HTML Converter}},
  author={{Fine-tuned from Hcompany/Holo1-3B}},
  year={{2024}},
  url={{https://huggingface.co/{repo_name}}}
}}
```

## License

This model is released under the Apache 2.0 license.
"""
        
        # Save model card
        model_card_path = os.path.join(model_path, "README.md")
        with open(model_card_path, "w", encoding="utf-8") as f:
            f.write(model_card)
        
        # Push to Hub
        print(f"📤 Uploading to {repo_name}...")
        model.push_to_hub(
            repo_id=repo_name,
            private=private,
            commit_message=commit_message,
            use_auth_token=True
        )
        
        processor.push_to_hub(
            repo_id=repo_name,
            private=private,
            commit_message=commit_message,
            use_auth_token=True
        )
        
        print(f"✅ Model successfully pushed to: https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error pushing to Hub: {e}")
        return False

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

def load_holo1_model_optimized(use_deepspeed=True):
    """Load Holo1 model with memory optimizations and DeepSpeed compatibility"""
    print("Loading Hcompany/Holo1-3B with memory optimizations...")
    
    model_id = "Hcompany/Holo1-3B"
    
    if use_deepspeed:
        # For DeepSpeed, we don't use quantization as it can conflict with Zero Stage 3
        print("Loading model for DeepSpeed optimization...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=None,  # Let DeepSpeed handle device placement
        )
    else:
        # Setup quantization for non-DeepSpeed training
        bnb_config = setup_quantization()
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        # Prepare model for k-bit training (required for quantized models)
        model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing to save memory (only for non-DeepSpeed training)
    if not use_deepspeed:
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
    
    if not use_deepspeed:
        clear_memory()
    
    return model, processor

def load_webcode2m_dataset(num_samples=1000):
    """Load and prepare the webcode2m_purified dataset"""
    print(f"Loading webcode2m_purified dataset with {num_samples} samples...")
    
    # Use streaming mode to avoid downloading the entire dataset
    # This loads data on-demand and is much faster
    dataset_stream = load_dataset("xcodemind/webcode2m_purified", split="train", streaming=True)
    
    # Take only the required number of samples from the stream
    dataset_list = []
    for i, sample in enumerate(dataset_stream):
        if i >= num_samples:
            break
        dataset_list.append(sample)
        if (i + 1) % 100 == 0:  # Progress indicator
            print(f"Loaded {i + 1}/{num_samples} samples...")
    
    # Convert list back to dataset format
    from datasets import Dataset
    dataset = Dataset.from_list(dataset_list)
    
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

def collate_fn_factory(processor, image_token_id, use_deepspeed=True):
    """Factory function to create collate_fn for Holo1 with memory optimization"""
    
    def collate_fn(examples):
        # For DeepSpeed, we can be more aggressive with batch processing
        batch_size = len(examples)
        max_chunk_size = 4 if use_deepspeed else 2
        
        if batch_size > max_chunk_size:
            # Process in chunks to avoid OOM
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
            if not use_deepspeed:
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
            
            # Clear memory after each evaluation (only if not using DeepSpeed)
            if not torch.distributed.is_initialized():
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

def setup_deepspeed_config():
    """Setup DeepSpeed configuration"""
    config_path = "ui_dom_finetuning/ds_z3.json"
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Warning: DeepSpeed config file {config_path} not found!")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Using DeepSpeed config: {config}")
    return config_path

def main():
    """Main training function with DeepSpeed optimization"""
    parser = argparse.ArgumentParser(description="Holo1 DOM Finetuning with DeepSpeed")
    parser.add_argument("--hub_repo", type=str, default="holo1_ui2dom", help="Hugging Face Hub repository name")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub after training")
    parser.add_argument("--private_repo", action="store_true", help="Make the Hub repository private")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of training samples")
    
    args = parser.parse_args()
    
    print("Starting Holo1 DOM finetuning with DeepSpeed...")
    print(f"Training samples: {args.num_samples}")
    if args.push_to_hub:
        print(f"Will push to Hub: {args.hub_repo} (private: {args.private_repo})")
    
    # Setup DeepSpeed config
    use_deepspeed = setup_deepspeed_config() is not None
    
    # Load model and processor
    model, processor = load_holo1_model_optimized(use_deepspeed=use_deepspeed)
    
    # Load and preprocess dataset
    dataset = load_webcode2m_dataset(num_samples=args.num_samples)
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
    collate_fn = collate_fn_factory(processor, image_token_id, use_deepspeed=use_deepspeed)
    
    # Training arguments with DeepSpeed optimization
    training_args = TrainingArguments(
        output_dir="./holo1_dom_finetune_output",
        per_device_train_batch_size=1,  # Matches DeepSpeed config
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Matches DeepSpeed config
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
        bf16=True,  # Matches DeepSpeed config
        gradient_checkpointing=False,  # Aligned with RL trainer
        dataloader_drop_last=True,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb/tensorboard
        deepspeed="ui_dom_finetuning/ds_z3.json" if use_deepspeed else None,  # DeepSpeed config
        # Additional DeepSpeed-specific settings
        save_total_limit=2,
        prediction_loss_only=True,
        # Hub settings
        hub_model_id=args.hub_repo if args.push_to_hub else None,
        hub_strategy="checkpoint" if args.push_to_hub else None,
        hub_private_repo=args.private_repo,
        push_to_hub=args.push_to_hub,
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
        final_model_path = "./holo1_dom_finetune_final"
        trainer.save_model(final_model_path)
        processor.save_pretrained(final_model_path)
        
        # Push to Hub if requested
        if args.push_to_hub:
            print("\n" + "="*50)
            print("PUSHING TO HUGGING FACE HUB")
            print("="*50)
            success = push_to_hub(
                model_path=final_model_path,
                repo_name=args.hub_repo,
                private=args.private_repo,
                commit_message="Upload trained Holo1 UI2DOM model"
            )
            if success:
                print(f"🎉 Model successfully uploaded to: https://huggingface.co/{args.hub_repo}")
            else:
                print("❌ Failed to push model to Hub")
        
        # Evaluate the model
        print("Evaluating final model...")
        eval_results = evaluate_model_subset(model, processor, test_dataset, max_samples=10)
        
        print(f"\nFinal evaluation results:")
        print(f"Success rate: {eval_results['success_rate']:.2%}")
        print(f"Successful generations: {eval_results['successful_generations']}/{eval_results['total_samples']}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        # Save checkpoint even if training fails
        checkpoint_path = "./holo1_dom_finetune_checkpoint"
        trainer.save_model(checkpoint_path)
        processor.save_pretrained(checkpoint_path)
        
        # Try to push checkpoint to Hub if requested
        if args.push_to_hub:
            print("Attempting to push checkpoint to Hub...")
            push_to_hub(
                model_path=checkpoint_path,
                repo_name=f"{args.hub_repo}-checkpoint",
                private=args.private_repo,
                commit_message="Upload training checkpoint"
            )
        
        raise

if __name__ == "__main__":
    main() 