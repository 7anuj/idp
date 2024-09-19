import re
import pandas as pd
import emoji
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch

# Load dataset from txt file (assuming each line is a tweet)
file_path = 'telugu.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    data = f.readlines()

# Convert to DataFrame for easy processing
df = pd.DataFrame(data, columns=['original_tweet'])

# Initialize the normalizer for Telugu text
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("te")

# Function to clean the text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # Remove special characters, numbers, and punctuations (preserve Telugu characters)
    text = re.sub(r'[^\w\s\'’\u0C00-\u0C7F]+', '', text)  # Retain Telugu Unicode range (U+0C00 to U+0C7F)

    # Normalize Telugu text
    text = normalizer.normalize(text)

    # Convert English words to lowercase
    text = text.lower()

    # Replace emojis with a placeholder or remove them
    text = emoji.replace_emoji(text, replace='')

    return text

# Apply the cleaning function to the entire dataset
df['cleaned_tweet'] = df['original_tweet'].apply(clean_text)

# Display cleaned data
print(df['cleaned_tweet'].head())



# Assuming your DataFrame is already loaded as df
# It should have columns like: 'original_tweet' and 'cleaned_tweet'
# Ensure you have 'label' for indicating duplicates

# Initialize the normalizer for Telugu text
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("te")

# Function to clean the text (if you haven't cleaned yet)
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # Remove special characters, numbers, and punctuations (preserve Telugu characters)
    text = re.sub(r'[^\w\s\'’\u0C00-\u0C7F]+', '', text)  # Retain Telugu Unicode range (U+0C00 to U+0C7F)

    # Normalize Telugu text
    text = normalizer.normalize(text)

    # Convert English words to lowercase
    text = text.lower()

    # Replace emojis with a placeholder or remove them
    text = emoji.replace_emoji(text, replace='')

    return text

# If your 'cleaned_tweet' column is already clean, no need to clean again.
# You can clean the original column if needed (if 'cleaned_tweet' is not already clean)
df['cleaned_tweet'] = df['cleaned_tweet'].apply(clean_text)

df['label']=0

# Ensure you have a 'label' column for duplicate detection
# If you don't have labels, you need to create a 'label' column manually.
# E.g., 1 for duplicate, 0 for non-duplicate

# Split the dataset into training and validation sets (80/20 split)
train_df, val_df = train_test_split(df[['cleaned_tweet', 'label']], test_size=0.2, random_state=42)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")




# Load pre-trained mBERT tokenizer and model for binary classification
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2).cuda()

# Custom Dataset class for handling tweet pairs
class TwitterDuplicateDataset(Dataset):
    def _init_(self, tweets, labels):
        self.tweets = tweets
        self.labels = labels

    def _len_(self):
        return len(self.labels)

    def _getitem_(self, idx):
        encoded = tokenizer(self.tweets[idx],
                            truncation=True, padding='max_length', max_length=128,
                            return_tensors='pt')
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0), torch.tensor(self.labels[idx])

# Create train and validation datasets
train_dataset = TwitterDuplicateDataset(train_df['cleaned_tweet'].tolist(), train_df['label'].tolist())
val_dataset = TwitterDuplicateDataset(val_df['cleaned_tweet'].tolist(), val_df['label'].tolist())

# DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=2e-5)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Number of epochs
epochs = 3

# Fine-tuning loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    total_loss = 0

    # Training loop
    for batch in train_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Average Training Loss: {avg_train_loss}")

# Save the fine-tuned model
model.save_pretrained("mbert_finetuned_duplicate_detection")
tokenizer.save_pretrained("mbert_finetuned_duplicate_detection")

print("Model saved!")


model.eval()  # Set model to evaluation mode
total_preds = []
total_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        total_preds.extend(preds)
        total_labels.extend(labels.cpu().numpy())

# Calculate evaluation metrics
accuracy = accuracy_score(total_labels, total_preds)
precision = precision_score(total_labels, total_preds)
recall = recall_score(total_labels, total_preds)
f1 = f1_score(total_labels, total_preds)

print(f"Validation Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")