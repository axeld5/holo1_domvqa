#!/usr/bin/env python3
"""
Example script showing how to use the trained holo1_ui2dom model from Hugging Face Hub
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import argparse
import os

def load_model_from_hub(repo_name="holo1_ui2dom"):
    """Load the trained model from Hugging Face Hub"""
    print(f"Loading model from Hub: {repo_name}")
    
    try:
        # Load model and processor
        model = AutoModelForImageTextToText.from_pretrained(
            repo_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(repo_name, trust_remote_code=True)
        
        print("✅ Model loaded successfully!")
        return model, processor
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure the model exists on the Hub and you have access to it.")
        return None, None

def convert_image_to_html(model, processor, image_path, output_path=None):
    """Convert a UI screenshot to HTML code"""
    
    # Load and prepare image
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        print(f"📸 Loaded image: {image_path} ({image.size})")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None
    
    # Prepare the conversation
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "convert this image to html"},
                {"type": "image"},
            ]
        }
    ]
    
    # Process input
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt")
    
    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print("🔄 Generating HTML...")
    
    # Generate HTML
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # Decode the result
    generated_text = processor.batch_decode(
        generated_ids[:, inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )[0]
    
    # Clean up the generated text
    html_code = generated_text.strip()
    
    # Save to file if output path is provided
    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_code)
            print(f"💾 HTML saved to: {output_path}")
        except Exception as e:
            print(f"⚠ Error saving HTML: {e}")
    
    return html_code

def main():
    parser = argparse.ArgumentParser(description="Convert UI screenshots to HTML using holo1_ui2dom")
    parser.add_argument("--image", type=str, required=True, help="Path to the UI screenshot")
    parser.add_argument("--output", type=str, help="Output file path for the generated HTML")
    parser.add_argument("--model", type=str, default="holo1_ui2dom", help="Model name on Hugging Face Hub")
    parser.add_argument("--show_html", action="store_true", help="Print the generated HTML to console")
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"❌ Image file not found: {args.image}")
        return
    
    print("🚀 Starting UI to HTML conversion...")
    print("=" * 50)
    
    # Load model
    model, processor = load_model_from_hub(args.model)
    if model is None or processor is None:
        return
    
    # Convert image to HTML
    html_code = convert_image_to_html(model, processor, args.image, args.output)
    
    if html_code:
        print("\n" + "=" * 50)
        print("✅ Conversion completed successfully!")
        print(f"📊 Generated HTML length: {len(html_code)} characters")
        
        if args.show_html:
            print("\n" + "=" * 50)
            print("GENERATED HTML:")
            print("=" * 50)
            print(html_code)
        
        if not args.output:
            print("\n💡 Tip: Use --output to save the HTML to a file")
            print("💡 Tip: Use --show_html to display the generated HTML")
    else:
        print("❌ Conversion failed")

if __name__ == "__main__":
    main() 