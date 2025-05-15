#!/usr/bin/env python3

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(question, max_length=200, temperature=0.7):
    # Load the model and tokenizer
    model_path = "./recession_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Set pad token properly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Format the input
    prompt = f"Question: {question}\nAnswer:"
    
    # Tokenize with explicit attention mask
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Make sure attention mask is properly set
    attention_mask = inputs.get("attention_mask", None)
    
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Process the answer
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:", 1)[1].strip()
    else:
        answer = generated_text.replace(prompt, "").strip()
    
    return answer

def main():
    # Default prompt if none provided
    prompt = "What happens during an economic recession?"
    
    # Use command line argument as prompt if provided
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    
    print(f"Input question: {prompt}")
    print("-" * 40)
    
    print("Loading model and generating response...")
    # Generate text using the model
    answer = generate_text(prompt)
    
    print("\nGenerated answer:")
    print(answer)

if __name__ == "__main__":
    main()