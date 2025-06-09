from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("../model/gpt2-hp")
tokenizer = GPT2Tokenizer.from_pretrained("../model/gpt2-hp")
model.eval()

while True:
    prompt = input("You: ")
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(input_ids, max_length=150, pad_token_id=tokenizer.eos_token_id)
    print("Bot:", tokenizer.decode(output[0], skip_special_tokens=True))