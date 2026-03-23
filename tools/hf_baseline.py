import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def main():
    print("Loading HuggingFace GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a galaxy far far away",
        "Alan Turing was a"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors='pt')
        
        # Generate 50 tokens with greedy decoding to match C++ engine
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated:")
        print(generated_text[len(prompt):])

if __name__ == "__main__":
    main()
