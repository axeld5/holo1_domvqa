import json
import os
import argparse

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)
from trl import GRPOConfig, GRPOTrainer
from rl_rewards import compute_vqa_rewards

# NEW: LoRA / PEFT imports
from peft import LoraConfig, get_peft_model, PeftModel


def train_vqa_rl_model(
    model_name: str = "HCompany/Holo1-3B",
    max_steps: int = 1_000,
    save_path: str = "holo1-3b-vqa-rl",
    dataset_file: str = "vqa_rl_data.json",
    lora_r: int = 64,
    push_to_hub: bool = False,
    hf_repo_id: str | None = None,
):
    """Train a VQA model using GRPO with a LoRA adapter (rank = ``lora_r``).

    The adapter is *merged* into the base weights before saving so the final
    model contains everything it needs — no separate LoRA files required.
    Optionally, push the merged weights to the HuggingFace Hub.
    """

    # ------------------------------------------------------------------
    # 0. Dataset ----------------------------------------------------------------
    # ------------------------------------------------------------------
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(
            f"Dataset file {dataset_file!r} not found. Please run prepare_dataset.py first."
        )

    print(f"[INFO] Loading dataset from: {dataset_file}")
    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Keep only the first user turn (question) for each training example
    train_rows = []
    for example in data:
        if example.get("split") != "train":
            continue
        for turn in example.get("conversations", []):
            if turn.get("role") == "user":
                train_rows.append(
                    {
                        "question": turn["content"].strip(),
                        "prompt": example["conversations"],
                    }
                )
                break
    dataset = Dataset.from_list(train_rows)
    print(f"[INFO] Prepared {len(dataset):,} training examples")

    # ------------------------------------------------------------------
    # 1. Model + LoRA ------------------------------------------------------------
    # ------------------------------------------------------------------
    print(f"[INFO] Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # NOTE: We load the ImageText2Text model since questions include DOM + text.
    base_model = AutoModelForImageTextToText.from_pretrained(model_name)

    # Ensure padding token is defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA configuration (rank = r)
    print(f"[INFO] Initialising LoRA (r = {lora_r})")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32 * 2,  # common heuristic
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # You can fine-tune which modules to target. For most transformer
        # models, q_proj & v_proj are sufficient — adapt as needed.
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
        ],
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 2. GRPO training configuration -----------------------------------
    # ------------------------------------------------------------------
    # Large context window for DOM + question
    max_prompt_length = 16_384
    max_seq_length = 16_484

    training_args = GRPOConfig(
        learning_rate=5e-5,  # Slightly higher LR for LoRA adapters
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        logging_steps=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        num_train_epochs=1,
        max_steps=max_steps,
        save_steps=200,
        max_grad_norm=1.0,
        report_to="none",
        output_dir="outputs",
        remove_unused_columns=True,
        deepspeed="useful_scripts/ds_z3.json",
        gradient_checkpointing=False,
    )

    # ------------------------------------------------------------------
    # 3. Training -------------------------------------------------------
    # ------------------------------------------------------------------
    reward_func = compute_vqa_rewards

    print("[INFO] Creating GRPO trainer")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    print(f"[INFO] Starting VQA RL training for {max_steps} steps")
    trainer_stats = trainer.train()

    # ------------------------------------------------------------------
    # 4. Merge LoRA weights + save -------------------------------------
    # ------------------------------------------------------------------
    print("[INFO] Merging LoRA adapter into base model weights …")
    merged_model = model.merge_and_unload() if isinstance(model, PeftModel) else model

    print(f"[INFO] Saving merged model to: {save_path}")
    merged_model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)

    # ------------------------------------------------------------------
    # 5. Optional: push to Hub -----------------------------------------
    # ------------------------------------------------------------------
    if push_to_hub:
        repo_id = hf_repo_id or os.path.basename(save_path.rstrip("/"))
        print(f"[INFO] Pushing model to Hub: {repo_id}")
        merged_model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)

    print("[SUCCESS] VQA RL training finished!")
    return trainer_stats


# --------------------------------------------------
# Quick test utility (unchanged apart from file paths)
# --------------------------------------------------

def test_vqa_model(model_path: str, test_examples: list | None = None):
    """Run a few manual VQA queries against the fine-tuned model."""

    if test_examples is None:
        test_examples = [
            {
                "dom": "<html><body><h1>Welcome</h1><p>This is a test page.</p></body></html>",
                "question": "What is the main heading of the page?",
            },
            {
                "dom": "<html><body><div class='price'>$29.99</div><button>Buy Now</button></body></html>",
                "question": "What is the price displayed?",
            },
        ]

    print(f"[INFO] Loading trained model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    print("\n==================== VQA test ====================")
    for i, example in enumerate(test_examples, 1):
        dom = example["dom"]
        question = example["question"]

        prompt = (
            "Given the following DOM of a page, answer the question that is asked.\n"
            f"<dom>{dom}</dom>\n"
            f"Question: {question}\n"
            "Your answer must be a boolean, a word or a number, contained within $\\boxed{}$. "
            "Now answer the question.\nAnswer:"
        )

        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = tokenizer([input_text], return_tensors="pt")

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        print(f"Example {i}")
        print(f"Q: {question}")
        print(f"A: {output}\n")


# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VQA model with GRPO + LoRA")
    parser.add_argument("--model_name", type=str, default="HCompany/Holo1-3B")
    parser.add_argument("--max_steps", type=int, default=1_000)
    parser.add_argument("--save_path", type=str, default="holo1-3b-vqa-rl")
    parser.add_argument("--dataset_file", type=str, default="vqa_rl_data.json")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_repo_id", type=str, default=None)
    parser.add_argument("--test_only", action="store_true")

    args = parser.parse_args()

    if args.test_only:
        test_vqa_model(args.save_path)
    else:
        train_vqa_rl_model(
            model_name=args.model_name,
            max_steps=args.max_steps,
            save_path=args.save_path,
            dataset_file=args.dataset_file,
            lora_r=args.lora_r,
            push_to_hub=args.push_to_hub,
            hf_repo_id=args.hf_repo_id,
        )
