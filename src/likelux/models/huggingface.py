import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Mistral3ForConditionalGeneration
from typing import List, Dict, Union

class HFModel:
    """
    A class for loading a base language model and evaluating single-token probabilities.
    """

    def __init__(self, model_name: str, temperature: float = 1.0):
        """
        Initialize a base model without fine-tuning for logit extraction.
        
        Args:
            model_name: The name or path of the pre-trained model.
            temperature: Softmax temperature scaling factor.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="cuda:0",
            torch_dtype="auto" 
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.temperature = temperature
        self.model.eval()

    def generate(self, prompt: Union[List[dict], str], options: List[str] = ["0", "1"]) -> Dict[str, float]:
        """
        Evaluate the probability distribution for specific next-token options.
        
        Args:
            prompt: The chat dictionary or string prompt.
            options: List of single-token strings to evaluate probabilities for.
            
        Returns:
            Dictionary mapping options to their absolute probabilities, 
            along with the residual Out-of-Option probability mass.
        """
        formatted_prompt = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False 
        )
        
        input_ids = self.tokenizer.encode(
            formatted_prompt, 
            return_tensors="pt", 
            add_special_tokens=False
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            
        # Isolate logits for the final token in the prompt sequence
        next_token_logits = outputs.logits[0, -1, :] / self.temperature
        
        # Calculate probability distribution across the entire vocabulary
        vocab_probs = F.softmax(next_token_logits, dim=-1)

        results = {}
        option_prob_sum = 0.0

        for option in options:
            option_token_ids = self.tokenizer.encode(option, add_special_tokens=False)
            
            if len(option_token_ids) != 1:
                raise ValueError(f"Optimization Failure: Option '{option}' tokenizes to {len(option_token_ids)} tokens. Expected exactly 1.")
                
            target_token_id = option_token_ids[0]
            prob = vocab_probs[target_token_id].item()
            
            results[option] = prob
            option_prob_sum += prob

        results["Out-of-Option-Probability"] = max(0.0, 1.0 - option_prob_sum)
        return results
    
    def push_to_hub(self, repo_name: str, organization: str = None):
        """
        Push the model to the Hugging Face Hub.
        """
        self.model.push_to_hub(repo_name, organization=organization, private=False)
        self.tokenizer.push_to_hub(repo_name, organization=organization, private=False)


class HFModelMistral(HFModel):
    """
    Mistral-specific implementation leveraging Mistral3ForConditionalGeneration.
    """

    def __init__(self, model_name: str, temperature: float = 1.0):
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="cuda:0",
            torch_dtype="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.temperature = temperature
        self.model.eval()