# Format the data into a huggingface dataset for fine-tuning

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Load Cleaned Data
df = pd.read_csv("data/fed_speeches_cleaned.csv")
print(f"Loaded {len(df)} speeches")

# Convert to a hugging face dataset
dataset = Dataset.from_pandas(df[["text"]])

# Load Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token # GPT2 doesn't have a pad token by default

# Tokenize the text
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True, # Cut the input sequence to the model's max size
        max_length=512, # BERT uses 512 tokens, GPT-2 can handle more but we'll keep it manageable
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
print(f"Tokenized dataset: {tokenized_dataset}")

# Split into train and eval
split = tokenized_dataset.train_test_split(test_size=0.2, seed=67)
train_dataset= split["train"]
eval_dataset = split["test"]
print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")


# Import the model and add the training configuration

from peft import LoraConfig, get_peft_model, TaskType

# load the model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Configure LoRA
lora_config = LoraConfig(
    r=16, # rank -> higher means more parameters and better performance but higher computational cost
    lora_alpha=32, # The scaling factor for the LoRA updates. Higher values mean more aggressive updates, but can lead to instability. 32 is a common choice for language models.
    lora_dropout=0.1, #dropout rate
    target_modules=["c_attn", "c_proj"], # We are only using the attention layers for fine tuning here as it is the most common minimal choice
    task_type=TaskType.CAUSAL_LM # use causal, not masked as GPT is causal
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False # Again, GPT2 is causal, not masked
)

# Training arguments
training_args = TrainingArguments(
    output_dir = "./gpt2-finetuned-fedspeeches",
    num_train_epochs = 3,
    per_device_train_batch_size = 4, # Number of samples per batch
    per_device_eval_batch_size = 4,
    warmup_steps = 100, # Learning rate gradually increases from 0 over 100 steps, rather than jumping to 5e-4
    learning_rate = 5e-4,
    weight_decay = 0.01,
    logging_dir = "./logs", # Where TensorBoard logs will be saved
    logging_steps = 50, # Print loss every 50 steps
    eval_strategy = "epoch", # Evaluate at the end of each epoch (could also use "steps" or "no")
    save_strategy = "epoch", # Save the model at the end of each epoch
    load_best_model_at_end=True,
    fp16=True # Change to 16 bit precision for faster training
)

# Trainer
trainer= Trainer(
    model=model,
    args = training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained("./gpt2-finetuned-fedspeeches")
tokenizer.save_pretrained("./gpt2-finetuned-fedspeeches")
print("Training Complete")

# By including c_proj and raising r to 16, the eval loss reduced to 2.811, compared to 2.871 for just c_attn.
# This isn't really important for the overall paper, but it is good to have a reduction in perplexity