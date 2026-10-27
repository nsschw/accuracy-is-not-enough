import argparse
import os
import yaml
import wandb
from dotenv import load_dotenv
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Mistral3ForConditionalGeneration
from trl import SFTConfig, SFTTrainer

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main(config_path: str):
    config = load_config(config_path)

    # 1. WandB Setup (API key is read from WANDB_API_KEY in .env)
    load_dotenv()
    wandb.login()
    os.environ["WANDB_PROJECT"] = config["wandb"]["project"]

    # 2. Model & Tokenizer Init
    model_config = config["model"]
    if "mistral" in model_config["name"]:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_config["name"],
            device_map=model_config["device_map"],
            dtype="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        device_map=model_config["device_map"],
        dtype="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"])

    # 3. Data Init
    dataset = load_dataset('json', data_files={'train': config["data"]["train_file"]})

    # 4. Training Config via Dictionary Unpacking
    training_args = SFTConfig(**config["training_args"])

    # 5. Trainer Initialization
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        args=training_args,
    )

    # 6. Execution
    trainer.train()
    trainer.save_model(config["output"]["save_model_path"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT Training via YAML config.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    
    main(args.config)