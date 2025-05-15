import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Check if a question was provided
    if len(sys.argv) < 2:
        print("Usage: python query_model.py 'Your question here'")
        return
    
    # Get the question from command line arguments
    question = " ".join(sys.argv[1:])
    
    # Load the model and tokenizer
    print("Loading model...")
    model_path = "./recession_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Set pad token properly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Format the input
    prompt = f"Question: {question}\nAnswer:"
    
    # Generate the answer
    print(f"\nQuestion: {question}")
    print("Generating answer...")
    
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
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Print the answer
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:", 1)[1].strip()
    else:
        answer = generated_text.replace(prompt, "").strip()
    
    print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()
