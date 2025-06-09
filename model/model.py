import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from gpt_config import config

model = GPT2LMHeadModel(config)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="../data/tokenized_dataset.txt",
    block_size=128
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir="./gpt2-hp",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

trainer.train()
model.save_pretrained("./gpt2-hp")
tokenizer.save_pretrained("./gpt2-hp")
"""
saving so that we can dso this later directly 
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./gpt2-hp")
tokenizer = AutoTokenizer.from_pretrained("./gpt2-hp")
"""