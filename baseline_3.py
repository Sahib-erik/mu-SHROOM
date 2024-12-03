from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import Dataset, DataLoader
import torch
import json

# Load T5 tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Custom Dataset
class SpanLabelDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                input_text = (
                    f"Premise: {example['model_input']} Hypothesis: {example['model_output_text']} "
                    f"Identify spans and probabilities that contradict the premise."
                )
                spans = example['soft_labels']
                target_text = " ".join(
                    [f"[{example['model_output_text'][s['start']:s['end']]}: {s['prob']}]" for s in spans]
                )
                self.data.append({"input": input_text, "target": target_text})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data[idx]["input"]
        target_text = self.data[idx]["target"]
        input_enc = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        target_enc = self.tokenizer(target_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": target_enc["input_ids"].squeeze()
        }

# Hyperparameters
batch_size = 4
epochs = 10
learning_rate = 5e-5

# Load dataset
train_dataset = SpanLabelDataset('/kaggle/input/muval1en/mushroom.en-val.v1.jsonl', tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

# Save fine-tuned model
model.save_pretrained("t5-span-labeling")
tokenizer.save_pretrained("t5-span-labeling")

# Inference and Save Predictions
model.eval()

# Prepare a list to hold predictions
predictions = []

with torch.no_grad():
    # Iterate through the dataset
    with open('/kaggle/input/muval1en/mushroom.en-val.v1.jsonl', 'r') as f:
        for line in f:
            example = json.loads(line.strip())
            input_text = (
                f"Premise: {example['model_input']} Hypothesis: {example['model_output_text']} "
                f"Identify spans and probabilities that contradict the premise."
            )

            # Tokenize the input
            input_enc = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

            # Print tokenized input length for debugging
            print(f"Input Length: {len(input_enc['input_ids'][0])}")

            # Generate the output
            outputs = model.generate(input_enc["input_ids"], max_length=150)

            # Check if the model is generating outputs
            if outputs is not None and len(outputs) > 0:
                print(f"Generated token IDs: {outputs[0]}")  # Print the generated token IDs
                predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Predicted Text: {predicted_text}")  # Print the predicted text
            else:
                predicted_text = "No prediction generated"  # Handle case where no output is generated

            # Save the input, prediction, and gold target in a dictionary
            predictions.append({
                "input_text": input_text,
                "predicted_text": predicted_text,
                "gold_target": " ".join(
                    [f"[{example['model_output_text'][s['start']:s['end']]}: {s['prob']}]" for s in example['soft_labels']]
                )
            })

# Save predictions to a JSON file
with open('predictions.json', 'w') as f:
    json.dump(predictions, f, indent=4)

print("Predictions saved to 'predictions.json'.")
