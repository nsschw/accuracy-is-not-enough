import os
import json
import yaml
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

# Inject environment variables before library initializations
load_dotenv()

# Deferred imports respect the CUDA_VISIBLE_DEVICES mask injected above
from likelux.models.huggingface import HFModelMistral

def load_config(config_path: str) -> dict:
    """Safely parse the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_prompts(data: list, max_num_samples: int, system_prompt: str) -> tuple[list, list]:
    prompts = []
    true_values = []

    for user in data:
        prompt = [{"role": "system", "content": system_prompt}]
        
        for sample in user["input"]:
            prompt.append({"role": "user", "content": sample["text"]})
            prompt.append({"role": "assistant", "content": str(sample["label"])})

        for i, sample in enumerate(user.get("output", [])):
            if i >= max_num_samples:
                break
            
            # 1. Append the target text for the current prediction
            prompt.append({"role": "user", "content": sample["text"]})
            true_values.append("1" if sample["label"] == 1 else "0")
            
            # 2. Snapshot the prompt (frozen at the user prompt) for generation
            prompts.append(prompt.copy())

            # 3. Autoregressively append the true label to the master prompt 
            #    so that step n+1 sees the historical interaction
            prompt.append({"role": "assistant", "content": true_values[-1]})

    return prompts, true_values


def main():
    parser = argparse.ArgumentParser(description="Next Like Prediction inference for open-weight models")
    parser.add_argument("--config", type=str, required=True, help="Path to execution config YAML")
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"Instantiating model: {config['model_path']}")
    model = HFModelMistral(config['model_path'])

    print(f"Loading dataset: {config['data_path']}")
    with open(config['data_path'], "r") as f:
        data = json.load(f) # Dataset remains JSON

    created_prompts, true_values = create_prompts(
        data=data, 
        max_num_samples=config['max_num_samples'], 
        system_prompt=config['system_prompt']
    )

    results = []
    os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)

    print("Extracting probabilities...")
    for true_val, prompt in tqdm(zip(true_values, created_prompts), total=len(true_values)):
        
        # 1. Extract certainty distribution
        prob_dist = model.generate(prompt, options=["0", "1"])
        
        # 2. Argmax classification over valid option space
        discrete_pred = max(["0", "1"], key=lambda k: prob_dist[k])
        
        results.append({
            "true_label": true_val,
            "prediction": discrete_pred,
            "probabilities": prob_dist,
            "history_length": (len(prompt) - 2) // 2
        })

        # 3. Iterative checkpointing to JSON
        with open(config['output_path'], "w") as f:
            json.dump(results, f, indent=4)

    print(f"Evaluation complete. Results safely written to {config['output_path']}")

if __name__ == "__main__":
    main()