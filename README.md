# holo1_domvqa

**GRPO-based reinforcement learning fine-tuning of Holo1-3B for DOM-based Visual Question Answering**

## Overview

This project implements Group Relative Policy Optimization (GRPO) training on the HCompany/Holo1-3B model to improve performance on DOM-based Visual Question Answering tasks. The model learns to answer questions about web pages by analyzing their DOM structure.

## 🚧 Current Status

**Experiment paused due to token count being too high.** The DOM structures are exceeding the model's context window, making training infeasible with current approach.

## Project Structure

```
holo1_domvqa/
├── useful_scripts/
│   ├── vqa_rl_trainer.py    # Main GRPO training script with LoRA
│   ├── rl_rewards.py        # Reward functions for RL training
│   └── vqa_eval.py         # Evaluation utilities
├── json_to_dataset/
│   └── prepare_dataset.py   # Dataset preparation utilities
├── generating_qa/
│   └── dom_vqa_gen.ipynb   # Question-answer generation notebook
├── labelling_difficulty/
│   ├── diff_labeling.ipynb # Difficulty assessment tools
│   └── fusing.ipynb        # Data fusion utilities
└── analyzing_outputs/      # Output analysis tools
```

## Key Features

### Training Pipeline
- **GRPO Training**: Uses Group Relative Policy Optimization for RL fine-tuning
- **LoRA Integration**: Efficient fine-tuning with Low-Rank Adaptation (rank=64)
- **Reward System**: Custom reward functions based on:
  - Answer format validation (requires `\boxed{}` format)
  - Semantic similarity to ground truth answers

### Model Details
- **Base Model**: HCompany/Holo1-3B (ImageText2Text architecture)
- **Context Window**: 16,384 tokens for prompts
- **Training Config**: 
  - Learning rate: 5e-5
  - Batch size: 1 (gradient accumulation: 8)
  - Max steps: 1,000

### Data Format
Training examples expect conversations in the format:
```json
{
  "conversations": [
    {
      "role": "user", 
      "content": "Given the following DOM... Question: What is X?"
    },
    {
      "role": "assistant",
      "content": "Answer text"
    }
  ],
  "split": "train"
}
```

## Usage

### Training
```bash
python useful_scripts/vqa_rl_trainer.py \
    --model_name HCompany/Holo1-3B \
    --max_steps 1000 \
    --dataset_file vqa_rl_data.json \
    --save_path holo1-3b-vqa-rl
```

### Testing
```bash
python useful_scripts/vqa_rl_trainer.py --test_only --save_path ./trained_model
```

## Reward Function

The reward system evaluates model outputs based on:

1. **Format Compliance**: Answers must be in `\boxed{content}` format with <5 words
2. **Semantic Similarity**: Uses difflib.SequenceMatcher to compare with ground truth
3. **Penalty System**: Returns -1.0 for format violations, similarity score [0,1] otherwise

## Technical Challenges

### Current Blocker: Token Count
- DOM structures from web pages are extremely verbose
- Many examples exceed the 16K token context window
- Need to implement DOM pruning/summarization strategies

### Potential Solutions
- Implement intelligent DOM filtering
- Use DOM-to-text summarization
- Hierarchical processing of DOM elements
- Multi-turn conversation approach

## Dependencies

- `transformers` - Hugging Face model loading
- `trl` - GRPO training implementation  
- `peft` - LoRA adapter support
- `datasets` - Data handling
- `torch` - PyTorch backend

*This project is part of research into improving multimodal AI capabilities for web-based question answering.*
