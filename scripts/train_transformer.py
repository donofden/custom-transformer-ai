import os
import json
import torch
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "../config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Define hyperparameters for easy modification
learning_rate = config["learning_rate"] # Lower the learning rate to improve stability
batch_size = config["batch_size"] # Increase batch size for more efficient learning
epochs = config["epochs"] # Train for more epochs
vocab_size = config["vocab_size"] # Set vocabulary size to match the tokenizer
max_length = config["max_length"] # Maximum sequence length
batch_first = config["batch_first"] # Ensures batch-first processing for transformer model

# Load custom tokenizer
tokenizer_path = os.path.join(os.path.dirname(__file__), config["tokenizer_path"])
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")

tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

# Ensure padding and EOS token is set
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # Explicitly add pad token if missing
    tokenizer.pad_token = "[PAD]"
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({"eos_token": "[EOS]"})  # Add EOS token if missing

def encode(text):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
    if tokens is None:
        tokens = [tokenizer.unk_token_id]  # Use UNK token if encoding fails
    tokens.append(tokenizer.eos_token_id or tokenizer.pad_token_id)  # Ensure the model knows when to stop
    return tokens

# Debug Tokenizer Before Training
test_text = "What is Python?"
test_encoded = encode(test_text)
test_decoded = tokenizer.decode(test_encoded)
print(f"Test Input: {test_text}")
print(f"Encoded: {test_encoded}")
print(f"Decoded: {test_decoded}")
print(f"Tokenizer vocab size: {len(tokenizer.get_vocab())}")

# Define Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers, dim_feedforward, batch_first=batch_first)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        return self.fc_out(output)

# Load Dataset
dataset = load_dataset("json", data_files=os.path.join(os.path.dirname(__file__), config["dataset_path"]))["train"]

train_data = [{"input_ids": encode(item["prompt"]), "output_ids": encode(item["response"])} for item in dataset]

# Convert dataset to tensors
train_tensors = [(torch.tensor(item["input_ids"], dtype=torch.long), torch.tensor(item["output_ids"], dtype=torch.long)) for item in train_data]

# Model, Loss, Optimizer
model = TransformerModel(vocab_size=vocab_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # Ignore padding tokens in loss calculation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Debug Model Output Before Training
random_input = torch.randint(0, vocab_size, (1, max_length), dtype=torch.long)
with torch.no_grad():
    random_output = model(random_input, random_input)
print(f"Random Output Token IDs Before Training: {random_output.argmax(dim=-1).squeeze().tolist()}")

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Starting Transformer Training... Start Time: {start_time}")
    
    # Training Loop
    for epoch in range(epochs):  # Increased epochs for better learning
        epoch_start = datetime.now()
        epoch_loss = 0
        for src, tgt in train_tensors:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src.unsqueeze(0), tgt.unsqueeze(0))
            loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_end = datetime.now()
        epoch_duration = epoch_end - epoch_start
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Duration: {epoch_duration}")
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    print(f"Training complete. End Time: {end_time}, Total Duration: {training_duration}")

    # Ensure the models directory exists
    models_dir = os.path.join(os.path.dirname(__file__), os.path.dirname(config["model_save_path"]))
    os.makedirs(models_dir, exist_ok=True)

    # Save Model
    torch.save(model.state_dict(), os.path.join(models_dir, os.path.basename(config["model_save_path"])))
    print("Model saved successfully.")
