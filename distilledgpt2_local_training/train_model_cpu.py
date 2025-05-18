import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Load recession data
print("Loading recession data...")
df = pd.read_csv("historical_recessions.csv")

# Create training examples
training_texts = []
for _, row in df.iterrows():
    # Create a question-answer pair for each recession
    question = f"What were the causes and impacts of the {row['name']}?"
    answer = f"The {row['name']} occurred from {row['start_date']} to {row['end_date']} and lasted {row['duration_months']} months. " \
             f"Its primary causes were {row['primary_causes']}. " \
             f"It resulted in a GDP decline of {row['gdp_decline_percent']}% and peak unemployment of {row['peak_unemployment_rate']}%. " \
             f"Notable policy responses included {row['notable_policy_responses']}."
    
    # Format as instruction-following example
    example = f"Question: {question}\nAnswer: {answer}"
    training_texts.append(example)

# Create a Hugging Face dataset
dataset = Dataset.from_dict({"text": training_texts})

# Load a smaller model that's easier to fine-tune
model_name = "distilgpt2"  # Much smaller than Gemma
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Make sure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./recession_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
)

# Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train the model
print("Training model...")
trainer.train()

# Save the model
print("Saving model...")
model.save_pretrained("./recession_model")
tokenizer.save_pretrained("./recession_model")

print("Model trained and saved to ./recession_model")