from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import json
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the T5 model and tokenizer
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

pairs = []
# Read data from sample_test.v1.json
with open('val/mushroom.en-val.v1.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        # Collect id, model_output_text, model_input, and soft_labels as pairs
        pairs.append((data['id'], data['model_output_text'], data['model_input'], data['soft_labels']))

# Define the modified prompt template
prompt_template = "Identify the part of the hypothesis that contradicts the premise.\n\nPremise: {text1}\n\nHypothesis: {text2}"

# List to hold all predicted spans
predicted_spans = []

# Helper functions for evaluation
def compute_exact_match(pred_start, pred_end, true_start, true_end):
    """Check if predicted span exactly matches the ground truth span."""
    return int(pred_start == true_start and pred_end == true_end)

def compute_f1(pred_start, pred_end, true_start, true_end):
    """Compute the F1 score for a predicted span vs ground truth span."""
    pred_span = set(range(pred_start, pred_end + 1))
    true_span = set(range(true_start, true_end + 1))
    overlap = pred_span.intersection(true_span)
    
    if len(overlap) == 0:
        return 0.0
    precision = len(overlap) / len(pred_span)
    recall = len(overlap) / len(true_span)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

# Evaluation metrics storage
exact_matches = 0
total_examples = 0
f1_scores = []

# Process each pair
for pair in pairs:
    id, model_output_text, model_input, soft_labels = pair
    prompt = prompt_template.format(text1=model_input, text2=model_output_text)
    
    # Tokenize and generate prediction
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    outputs = model.generate(input_ids)
    predicted_span_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Find character indices of the predicted span in the original text
    start_char_idx = model_output_text.find(predicted_span_text)
    end_char_idx = start_char_idx + len(predicted_span_text)
    
    # Debugging output
    print(f"Input Text: {model_output_text}")
    print(f"Predicted Span Text: '{predicted_span_text}'")
    print(f"Start Char Index: {start_char_idx}, End Char Index: {end_char_idx}")
    
    if start_char_idx != -1 and end_char_idx != -1 and start_char_idx < end_char_idx:
        predicted_spans.append({
            'id': id,
            'model_output_text': model_output_text,
            'target_text': model_input,
            'predicted_span': predicted_span_text,
            'hard_labels': [{'start': start_char_idx, 'end': end_char_idx}],
            'soft_labels': soft_labels
        })
    else:
        # Handle case where no valid span was found
        predicted_spans.append({
            'id': id,
            'model_output_text': model_output_text,
            'target_text': model_input,
            'predicted_span': None,
            'hard_labels': [],
            'soft_labels': soft_labels
        })

# Print or save the predicted spans with indices
for prediction in predicted_spans:
    print(json.dumps(prediction, indent=2))

# Save results to a file called predictions.jsonl
with open('predictions.jsonl', 'w') as outfile:
    for prediction in predicted_spans:
        json.dump(prediction, outfile)
        outfile.write("\n")