import pandas as pd
import torch
import platform
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Set environment variables to optimize MPS performance
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Check for available GPU acceleration
def check_gpu():
    print(f"PyTorch version: {torch.__version__}")
    print(f"System: {platform.system()} {platform.machine()}")
    
    # Check for CUDA (NVIDIA GPUs)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Check for MPS (Apple Silicon)
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS (Apple Metal) available: {mps_available}")
    
    # Determine device
    if cuda_available:
        device = "cuda"
    elif mps_available:
        # Try to create an MPS tensor to verify it actually works
        try:
            test_tensor = torch.zeros(1).to("mps")
            device = "mps"
            print("MPS is working correctly")
        except Exception as e:
            print(f"MPS reported as available but failed: {e}")
            device = "cpu"
            print("Falling back to CPU")
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    return device

def main():
    # Get the appropriate device
    device = check_gpu()

    # Load recession data
    print("Loading recession data...")
    df = pd.read_csv("historical_recessions.csv")

    # Create more training examples to increase workload
    print("Creating training examples...")
    training_texts = []
    for _, row in df.iterrows():
        # Create multiple question-answer pairs for each recession with variations
        questions = [
            f"What were the causes and impacts of the {row['name']}?",
            f"Explain the economic effects of the {row['name']}.",
            f"What led to the {row['name']} and how did it affect unemployment?",
            f"Describe the policy responses during the {row['name']}.",
            f"How long did the {row['name']} last and what were its consequences?"
        ]
        
        answer = f"The {row['name']} occurred from {row['start_date']} to {row['end_date']} and lasted {row['duration_months']} months. " \
                f"Its primary causes were {row['primary_causes']}. " \
                f"It resulted in a GDP decline of {row['gdp_decline_percent']}% and peak unemployment of {row['peak_unemployment_rate']}%. " \
                f"Notable policy responses included {row['notable_policy_responses']}."
        
        # Add all variations to increase dataset size
        for question in questions:
            example = f"Question: {question}\nAnswer: {answer}"
            training_texts.append(example)

    # Create a Hugging Face dataset
    dataset = Dataset.from_dict({"text": training_texts})

    # Load a smaller model that's easier to fine-tune
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move model to the appropriate device
    model.to(device)

    # Verify model is on MPS
    if device == "mps":
        print(f"Model device check: {next(model.parameters()).device}")
        
        # Test tensor operations on MPS
        print("Testing MPS tensor operations...")
        test_input = torch.rand(1, 768).to(device)  # Match embedding dimension
        try:
            test_output = model.forward(input_ids=torch.ones((1, 10), dtype=torch.long).to(device))
            print(f"Test output device: {test_output.logits.device}")
            print("MPS tensor test successful")
        except Exception as e:
            print(f"MPS test failed: {e}")

    # Make sure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./recession_model",
        overwrite_output_dir=True,
        num_train_epochs=5,  # More epochs
        per_device_train_batch_size=8,  # Larger batch size
        gradient_accumulation_steps=4,  # Accumulate gradients
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,  # More frequent logging
        fp16=False,  # Disable fp16 which might not work well with MPS
        dataloader_num_workers=0,  # Disable multiprocessing
        # Use appropriate device settings
        no_cuda=True  # Disable CUDA
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

if __name__ == "__main__":
    main()