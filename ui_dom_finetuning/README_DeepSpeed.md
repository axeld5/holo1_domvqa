# DeepSpeed Optimized Holo1 DOM Finetuning

This directory contains the DeepSpeed-optimized version of the Holo1 DOM finetuning pipeline for converting screenshots to HTML code.

## Overview

The training pipeline has been optimized with DeepSpeed Zero Stage 3 (ZeRO-3) to enable efficient training of large vision-language models with reduced memory footprint and improved training speed.

## Key Optimizations

### 1. DeepSpeed ZeRO-3 Configuration
- **Zero Stage 3**: Partitions model parameters, gradients, and optimizer states across GPUs
- **Memory Offloading**: Configured to keep parameters on GPU for optimal performance
- **Communication Optimization**: Overlapped communication and contiguous gradients
- **Mixed Precision**: BF16 training for memory efficiency

### 2. Model Loading Optimizations
- **Conditional Quantization**: Disables quantization when using DeepSpeed (avoids conflicts)
- **Device Mapping**: Lets DeepSpeed handle device placement automatically
- **Gradient Checkpointing**: Saves memory by recomputing gradients during backward pass

### 3. Memory Management
- **Adaptive Batch Processing**: Larger chunks for DeepSpeed, smaller for regular training
- **Smart Memory Clearing**: Only clears memory when not using distributed training
- **Optimized Collate Function**: Handles OOM gracefully with chunked processing

## Files

- `holo1_dom_finetune.py`: Main training script with DeepSpeed integration and Hub upload
- `ds_z3.json`: DeepSpeed Zero Stage 3 configuration
- `run_deepspeed_training.py`: Wrapper script for launching DeepSpeed training
- `use_model_from_hub.py`: Example script for using the trained model from Hugging Face Hub
- `README_DeepSpeed.md`: This documentation file

## Configuration (`ds_z3.json`)

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "none"
        },
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 8,
    "bf16": { "enabled": true }
}
```

### Configuration Explanation:
- **Stage 3**: Partitions parameters, gradients, and optimizer states
- **No Offloading**: Keeps parameters on GPU for better performance
- **Overlap Communication**: Overlaps communication with computation
- **Contiguous Gradients**: Optimizes memory layout for gradients
- **Micro Batch Size**: 1 sample per GPU per step
- **Gradient Accumulation**: Effective batch size = 1 × 8 = 8
- **BF16**: Uses bfloat16 for mixed precision training

## Usage

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have CUDA-compatible GPU(s) available

3. Set up environment variables (if needed):
```bash
export CUDA_VISIBLE_DEVICES=0  # For single GPU
# or
export CUDA_VISIBLE_DEVICES=0,1,2,3  # For multi-GPU
```

### Running Training

#### Option 1: Using the Wrapper Script (Recommended)
```bash
# Single GPU training
python run_deepspeed_training.py

# Multi-GPU training
python run_deepspeed_training.py --num_gpus 4

# Training with Hub upload
python run_deepspeed_training.py --push_to_hub --hub_repo "your-username/holo1_ui2dom"

# Private repository on Hub
python run_deepspeed_training.py --push_to_hub --hub_repo "your-username/holo1_ui2dom" --private_repo

# Custom configuration
python run_deepspeed_training.py --num_gpus 2 --config ds_z3.json --master_port 29501
```

#### Option 2: Direct DeepSpeed Launch
```bash
# Single GPU
deepspeed --num_gpus=1 holo1_dom_finetune.py --deepspeed ds_z3.json

# Multi-GPU
deepspeed --num_gpus=4 holo1_dom_finetune.py --deepspeed ds_z3.json
```

#### Option 3: Without DeepSpeed (Fallback)
```bash
python holo1_dom_finetune.py
```

### Command Line Arguments

For `run_deepspeed_training.py`:
- `--num_gpus`: Number of GPUs to use (default: 1)
- `--master_port`: Master port for distributed training (default: 29500)
- `--config`: DeepSpeed configuration file (default: ds_z3.json)
- `--script`: Training script to run (default: holo1_dom_finetune.py)
- `--hub_repo`: Hugging Face Hub repository name (default: holo1_ui2dom)
- `--push_to_hub`: Push model to Hugging Face Hub after training
- `--private_repo`: Make the Hub repository private
- `--num_samples`: Number of training samples (default: 1000)

For `holo1_dom_finetune.py` (direct usage):
- `--hub_repo`: Hugging Face Hub repository name (default: holo1_ui2dom)
- `--push_to_hub`: Push model to Hugging Face Hub after training
- `--private_repo`: Make the Hub repository private
- `--num_samples`: Number of training samples (default: 1000)

## Performance Benefits

### Memory Efficiency
- **ZeRO-3**: Reduces per-GPU memory usage by ~3x compared to standard training
- **Parameter Partitioning**: Splits model parameters across available GPUs
- **Gradient Checkpointing**: Trades computation for memory savings

### Speed Improvements
- **Overlapped Communication**: Hides communication latency
- **Optimized Collectives**: Efficient gradient synchronization
- **Mixed Precision**: BF16 training for faster computation

### Scalability
- **Multi-GPU Support**: Seamless scaling across multiple GPUs
- **Automatic Load Balancing**: DeepSpeed handles parameter distribution
- **Efficient Gradient Aggregation**: Optimized across distributed setup

## Monitoring Training

The training script provides detailed logging:
- Model loading progress and memory usage
- Dataset loading and preprocessing status
- Training metrics (loss, learning rate, etc.)
- GPU memory utilization
- Evaluation results

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `per_device_train_batch_size` in training arguments
   - Increase `gradient_accumulation_steps` to maintain effective batch size
   - Enable parameter offloading in `ds_z3.json`

2. **DeepSpeed Import Error**
   - Install DeepSpeed: `pip install deepspeed`
   - Check CUDA compatibility: `python -c "import deepspeed; print(deepspeed.__version__)"`

3. **Model Loading Issues**
   - Ensure sufficient system RAM for model loading
   - Check internet connection for downloading pretrained models
   - Verify HuggingFace token if using gated models

4. **Multi-GPU Issues**
   - Ensure all GPUs are visible: `nvidia-smi`
   - Check NCCL installation for multi-GPU communication
   - Verify consistent CUDA versions across GPUs

### Debug Mode

To run in debug mode with more verbose output:
```bash
CUDA_LAUNCH_BLOCKING=1 python run_deepspeed_training.py --num_gpus 1
```

### Memory Optimization Tips

1. **Enable CPU Offloading** (if GPU memory is limited):
```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu"
        },
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}
```

2. **Reduce Batch Size and Increase Accumulation**:
```json
{
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 16
}
```

3. **Enable Memory Monitoring**:
```bash
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Output

The training will create the following outputs:
- `holo1_dom_finetune_output/`: Training checkpoints and logs
- `holo1_dom_finetune_final/`: Final trained model and processor
- `holo1_dom_finetune_checkpoint/`: Emergency checkpoint (if training fails)

## Model Evaluation

The script includes automatic evaluation on a test subset:
- Generates HTML from screenshot samples
- Measures success rate of valid HTML generation
- Reports generation statistics

## Using the Trained Model

### From Local Files
After training, you can use the model from the local directory:
```python
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# Load model and processor
model = AutoModelForImageTextToText.from_pretrained("./holo1_dom_finetune_final", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("./holo1_dom_finetune_final", trust_remote_code=True)

# Use the model for inference
image = Image.open("screenshot.png")
# ... (rest of inference code)
```

### From Hugging Face Hub
If you pushed the model to the Hub:
```python
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# Load model and processor from Hub
model = AutoModelForImageTextToText.from_pretrained("your-username/holo1_ui2dom", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("your-username/holo1_ui2dom", trust_remote_code=True)

# Use the model for inference
image = Image.open("screenshot.png")
# ... (rest of inference code)
```

### Using the Provided Example Script
We've included a ready-to-use script for inference:
```bash
# Convert a screenshot to HTML
python use_model_from_hub.py --image screenshot.png --output generated.html

# Use a specific model from Hub
python use_model_from_hub.py --image screenshot.png --model "your-username/holo1_ui2dom" --show_html

# Display help
python use_model_from_hub.py --help
```

## Hugging Face Hub Integration

### Authentication
Before pushing to Hub, authenticate with Hugging Face:
```bash
# Option 1: Set environment variable
export HF_TOKEN="your_huggingface_token"

# Option 2: Use huggingface-cli
huggingface-cli login

# Option 3: Use Python
from huggingface_hub import login
login("your_huggingface_token")
```

### Model Card
The training script automatically generates a comprehensive model card with:
- Model description and usage instructions
- Training details and performance metrics
- Code examples and citations
- Licensing information

### Repository Settings
- **Public Repository**: Default setting, model will be publicly accessible
- **Private Repository**: Use `--private_repo` flag to keep the model private
- **Repository Name**: Customize with `--hub_repo` parameter

## Next Steps

After training:
1. **Local Usage**: Load the trained model from `holo1_dom_finetune_final/`
2. **Hub Upload**: Push to Hugging Face Hub for easy sharing and deployment
3. **Inference**: Use the model for inference on new screenshots
4. **Fine-tuning**: Fine-tune further on domain-specific data if needed
5. **Production**: Deploy the model for production use

## Support

For issues related to:
- **DeepSpeed**: Check the [DeepSpeed documentation](https://deepspeed.readthedocs.io/)
- **Transformers**: Refer to [HuggingFace documentation](https://huggingface.co/docs/transformers/)
- **CUDA/GPU**: Verify your CUDA installation and GPU compatibility 