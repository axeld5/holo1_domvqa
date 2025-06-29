import json
import random
import os

def get_dom_messages(question, dom):
    return [
        {
            "role": "user",
            "content": """Given the following DOM of a page, answer the question that is asked.
            <dom>""" + dom + """</dom>
            Question: """ + question + r"""
            Your answer must be a boolean, a word or a number, contained within $\boxed{}$. Now answer the question.
            Answer:"""
        }
    ]


def prepare_vqa_dataset(output_file="vqa_rl_data.json"):
    """
    Prepare the VQA dataset for RL training.
    
    Args:
        num_samples (int): Number of samples to extract
        output_file (str): Output file path
    """
    print("Loading VQA dataset...")
    with open("json_to_dataset/domvqa_fused_filtered.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # Prepare data in the format expected by the RL trainer
    rl_data = []
    
    for idx in range(len(dataset)):
        example = dataset[idx]
        question = example["question"]
        dom = example["cleaned_html"]
        answer = example["answer"]
        img_difficulty = example["img_difficulty"]
        dom_difficulty = example["dom_difficulty"]
        if img_difficulty == 0 or dom_difficulty == 0:
            print(f"Skipping example {idx} because it is too hard")
            continue
        
        # Create the conversation format
        conversation = {
            "conversations": get_dom_messages(question, dom)+ [{"role": "assistant", "content": answer}]
        }        
        rl_data.append(conversation)
    
    # Shuffle and split into train/eval
    random.seed(42)
    random.shuffle(rl_data)
    n_total = len(rl_data)
    n_eval = n_total // 5
    eval_data = rl_data[:n_eval]
    train_data = rl_data[n_eval:]

    # Add split label
    for ex in train_data:
        ex["split"] = "train"
    for ex in eval_data:
        ex["split"] = "eval"

    final_data = train_data + eval_data

    # Save the dataset
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved to {output_file}")
    print(f"Number of samples: {len(final_data)}")
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")
    
    # Show some examples
    print("\nFirst 3 examples:")
    for i, example in enumerate(final_data[:3]):
        print(f"Example {i+1}:")
        print(f"Split: {example['split']}")
        print(f"Question: {example['conversations'][0]['content']}")
        print(f"Answer: {example['conversations'][1]['content']}")
        print()

if __name__ == "__main__":
    prepare_vqa_dataset(output_file="vqa_rl_data.json") 