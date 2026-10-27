import os
import json

# --- Configuration ---
DATA_PATH = "data/processed/next-like-prediction-test-filtered.json"
OUTPUT_PATH = "outputs/results/statistical-baseline.json"
MAX_NUM_SAMPLES = 75

def evaluate_baseline(data: list, max_num_samples: int) -> list:
    """
    Computes a mathematical baseline relying strictly on the dynamic historical 
    distribution of labels. The history grows autoregressively with each sequential prediction.
    """
    results = []

    for user in data:
        # 1. Initialize running counts from the initial input context
        history = user.get("input", [])
        running_likes = sum(1 for sample in history if sample.get("label") == 1)
        running_total = len(history)

        # 2. Evaluate the target samples sequentially
        for i, sample in enumerate(user.get("output", [])):
            if i >= max_num_samples:
                break
            
            # 3. Calculate probability based strictly on all previous interactions (n-1)
            if running_total == 0:
                prob_1 = 0.5 
            else:
                prob_1 = running_likes / running_total
            
            prob_0 = 1.0 - prob_1
            
            # 4. Determine discrete prediction via majority class
            discrete_pred = "1" if prob_1 >= 0.5 else "0"
            true_val = "1" if sample.get("label") == 1 else "0"
            
            results.append({
                "true_label": true_val,
                "prediction": discrete_pred,
                "probabilities": {
                    "0": prob_0,
                    "1": prob_1,
                    "Out-of-Option-Probability": 0.0
                },
                "history_length": running_total
            })

            # 5. Append the current true label to the running history for step n+1
            running_total += 1
            if sample.get("label") == 1:
                running_likes += 1

    return results

if __name__ == "__main__":
    print(f"Loading dataset: {DATA_PATH}")
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    print("Calculating statistical baseline...")
    results = evaluate_baseline(data, MAX_NUM_SAMPLES)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Statistical baseline complete. Results safely written to {OUTPUT_PATH}")