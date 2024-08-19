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

from sklearn.metrics import classification_report

# Set the model to evaluation mode
model.eval()

# Store predictions and true labels
all_preds = []
all_labels = []

# No gradient calculation needed during evaluation
with torch.no_grad():
    for batch in test_loader:
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'label'}
        labels = batch['label'].to(device)
        
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Generate a classification report
report = classification_report(all_labels, all_preds, target_names=['Negative', 'Positive'])
print("\nClassification Report:")
print(report)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate a confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for IMDb Sentiment Analysis with BERT')
plt.show()

# Save the confusion matrix as an image file
plt.savefig('confusion_matrix_imdb_bert.png')
print("Confusion matrix saved as 'confusion_matrix_imdb_bert.png'.")