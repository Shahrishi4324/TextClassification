import torch
from datasets import load_dataset
from transformers import BertTokenizer

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the IMDb dataset from Hugging Face Datasets
dataset = load_dataset('imdb')

# Display dataset structure
print("Dataset Structure:")
print(dataset)

# Initialize the BERT tokenizer from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example of tokenizing a single sentence
example_sentence = dataset['train'][0]['text']
tokens = tokenizer.tokenize(example_sentence)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\nExample Sentence Tokenization:")
print(f"Sentence: {example_sentence}")
print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")

from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

# Define a function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

# Apply the tokenization to the entire dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare the DataCollator to dynamically pad the inputs during batching
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create PyTorch DataLoaders for training and validation
train_loader = DataLoader(tokenized_datasets['train'], batch_size=16, shuffle=True, collate_fn=data_collator)
test_loader = DataLoader(tokenized_datasets['test'], batch_size=16, shuffle=False, collate_fn=data_collator)

# Display the structure of the tokenized dataset
print("\nTokenized Dataset Structure:")
print(tokenized_datasets)

from transformers import BertForSequenceClassification, AdamW

# Load the pre-trained BERT model for sequence classification (2 classes for sentiment analysis)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# Define the optimizer (AdamW) for fine-tuning BERT
optimizer = AdamW(model.parameters(), lr=2e-5)

# Fine-tuning loop (simplified for demonstration purposes)
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'label'}
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}/{num_epochs} completed. Loss: {loss.item():.4f}")

# Save the fine-tuned model
model.save_pretrained('fine_tuned_bert_imdb')
tokenizer.save_pretrained('fine_tuned_bert_imdb')