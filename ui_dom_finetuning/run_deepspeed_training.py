#!/usr/bin/env python3

import subprocess
import sys
import os
import argparse

def run_deepspeed_training(
    num_gpus=1,
    master_port=29500,
    config_file="ds_z3.json",
    training_script="holo1_dom_finetune.py",
    hub_repo="holo1_ui2dom",
    push_to_hub=False,
    private_repo=False,
    num_samples=1000
):
    """Run DeepSpeed training with proper configuration"""
    
    # Check if DeepSpeed is installed
    try:
        import deepspeed
        print(f"DeepSpeed version: {deepspeed.__version__}")
    except ImportError:
        print("DeepSpeed not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "deepspeed"], check=True)
        import deepspeed
        print(f"DeepSpeed installed. Version: {deepspeed.__version__}")
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"Error: DeepSpeed config file {config_file} not found!")
        return False
    
    # Check if training script exists
    if not os.path.exists(training_script):
        print(f"Error: Training script {training_script} not found!")
        return False
    
    # Build DeepSpeed command
    cmd = [
        "deepspeed",
        "--num_gpus", str(num_gpus),
        "--master_port", str(master_port),
        training_script,
        "--deepspeed", config_file,
        "--hub_repo", hub_repo,
        "--num_samples", str(num_samples)
    ]
    
    # Add Hub-related arguments
    if push_to_hub:
        cmd.append("--push_to_hub")
    if private_repo:
        cmd.append("--private_repo")
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True)
        print("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run DeepSpeed training for Holo1 DOM finetuning")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--master_port", type=int, default=29500, help="Master port for distributed training")
    parser.add_argument("--config", type=str, default="ds_z3.json", help="DeepSpeed configuration file")
    parser.add_argument("--script", type=str, default="holo1_dom_finetune.py", help="Training script to run")
    parser.add_argument("--hub_repo", type=str, default="holo1_ui2dom", help="Hugging Face Hub repository name")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub after training")
    parser.add_argument("--private_repo", action="store_true", help="Make the Hub repository private")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of training samples")
    
    args = parser.parse_args()
    
    print("Starting DeepSpeed training...")
    print(f"GPUs: {args.num_gpus}")
    print(f"Master port: {args.master_port}")
    print(f"Config file: {args.config}")
    print(f"Training script: {args.script}")
    print(f"Training samples: {args.num_samples}")
    if args.push_to_hub:
        print(f"Will push to Hub: {args.hub_repo} (private: {args.private_repo})")
    print("=" * 50)
    
    success = run_deepspeed_training(
        num_gpus=args.num_gpus,
        master_port=args.master_port,
        config_file=args.config,
        training_script=args.script,
        hub_repo=args.hub_repo,
        push_to_hub=args.push_to_hub,
        private_repo=args.private_repo,
        num_samples=args.num_samples
    )
    
    if success:
        print("\n" + "=" * 50)
        print("Training completed successfully!")
        if args.push_to_hub:
            print(f"Model pushed to: https://huggingface.co/{args.hub_repo}")
        else:
            print("Check the output directory for saved models.")
    else:
        print("\n" + "=" * 50)
        print("Training failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 