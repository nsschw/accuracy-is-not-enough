import os
import math
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
from typing import List, Dict

class OpenRouterClient:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER"),
        )

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def call(self, model_id: str, messages: list[dict], temp: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temp,
            )
        
        print(response.choices[0])
        return response.choices[0].message.content

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def generate_with_certainty(
        self, 
        model_id: str, 
        messages: list[dict], 
        options: List[str] = ["0", "1"],
        temp: float = 0.0,
        reasoning: dict = None
    ) -> Dict:
        """
        Calls OpenRouter, forces logprob extraction, and returns linear probabilities 
        for specific target tokens.
        """
        kwargs = {
            "model": model_id,
            "messages": messages,
            "temperature": temp,
            "logprobs": True,
            "top_logprobs": 8,
        }

        if reasoning:
            kwargs["extra_body"] = {"reasoning": reasoning}

        response = self.client.chat.completions.create(**kwargs)
        
        choice = response.choices[0]
        print(choice)
        
        # Defensive check: The provider must support logprobs
        if choice.logprobs is None or not choice.logprobs.content:
            raise ValueError(f"API Provider constraint: Model '{model_id}' does not support logprobs.")

        # Isolate the logprobs for the very first generated token
        first_token_logprobs = choice.logprobs.content[0].top_logprobs
        
        results = {}
        option_prob_sum = 0.0
        
        # Convert logarithmic probabilities to linear space (0.0 to 1.0)
        for token_data in first_token_logprobs:
            token_str = token_data.token
            linear_prob = math.exp(token_data.logprob)
            
            if token_str in options:
                results[token_str] = linear_prob
                option_prob_sum += linear_prob
                
        # Handle options that were not highly probable enough to make the Top 20 cutoff
        for option in options:
            if option not in results:
                results[option] = 0.0

        # Calculate residual probability mass consumed by unexpected tokens
        results["Out-of-Option-Probability"] = max(0.0, 1.0 - option_prob_sum)
        
        return {
            "prediction": choice.logprobs.content[0].token,
            "probabilities": results
        }
