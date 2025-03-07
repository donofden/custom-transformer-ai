import os
import json
from fastapi import FastAPI
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from scripts.train_transformer import TransformerModel

app = FastAPI()

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "../config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Load tokenizer
tokenizer_path = os.path.join(os.path.dirname(__file__), config["tokenizer_path"])
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")

tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = "[PAD]"

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), config["model_save_path"])
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = TransformerModel(vocab_size=500)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

@app.get("/generate")
def generate_text(prompt: str):
    try:
        input_ids = tokenizer.encode(prompt, truncation=True, padding="max_length", max_length=64)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        print(f"Encoded Input: {input_ids}")  # Debug print

        with torch.no_grad():
            output = model(input_tensor, input_tensor)

        probabilities = F.softmax(output, dim=-1)  # Convert logits to probabilities
        generated_ids = torch.multinomial(probabilities.squeeze(), num_samples=1).squeeze().tolist()


        # generated_ids = output.argmax(dim=-1).squeeze().tolist()

        print(f"Generated Token IDs: {generated_ids}")  # Debug print

        # Ensure generated IDs are properly decoded
        if isinstance(generated_ids, int):
            generated_ids = [generated_ids]

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"Generated Text: {generated_text}")  # Debug print

        return {"generated_text": generated_text if generated_text.strip() else "(No meaningful output)"}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)