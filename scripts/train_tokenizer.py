import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "../config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Initialize tokenizer
tokenizer = Tokenizer(models.BPE())

# Define Pre-tokenizer
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Trainer settings
trainer = trainers.BpeTrainer(special_tokens=["<pad>", "<unk>", "<sos>", "<eos>"])

# Load dataset
dataset_file = os.path.join(os.path.dirname(__file__), config["dataset_path"])

with open(dataset_file, "r") as f:
    dataset_texts = [line.split(":")[1].strip() for line in f.readlines()]

# Train tokenizer
tokenizer.train_from_iterator(dataset_texts, trainer)

# Save tokenizer
tokenizer.save(os.path.join(os.path.dirname(__file__), config["tokenizer_path"]))

print("Tokenizer training completed.")
# Debugging: Print tokenizer vocab size
print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")

