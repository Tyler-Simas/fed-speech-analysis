import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
from tqdm import tqdm

# Load data
df = pd.read_csv("data/fed_speeches_cleaned.csv")

# Setup devicve
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer
tokenizer =GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def compute_perplexity(text, model, tokenizer, max_length = 512):
    encodings = tokenizer(
        text,
        return_tensors = "pt",
        truncation = True,
        max_length = max_length
    )
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    return torch.exp(loss).item() # Perplexity calculation

# Load base model
print("Computing base model perplexity...")
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
base_model.eval()

# Calculate perplexity across speeches with base model
base_perplexities = []
for text in tqdm(df["text"]):
    ppl = compute_perplexity(text, base_model, tokenizer)
    base_perplexities.append(ppl)

df["perplexity_base"] = base_perplexities

# Free GPU memory before loading the next model
del base_model
torch.cuda.empty_cache()

# Load the fine tuned model
ft_base = GPT2LMHeadModel.from_pretrained("gpt2")
ft_model = PeftModel.from_pretrained(ft_base, "./gpt2-finetuned-fedspeeches").to(device)
ft_model.eval()

# Calculate perplexity across speeches with finetuned model
ft_perplexities = []
for text in tqdm(df["text"]):
    ppl = compute_perplexity(text, ft_model, tokenizer)
    ft_perplexities.append(ppl)

df["perplexity_finetuned"] = ft_perplexities

# Save results
df.to_csv("fed_speeches_perplexity.csv", index=False)
print("Saved to fed_speeches_perplexity.csv")

# Summary Stats
print("\nBase Model Perplexity:")
print(df["perplexity_base"].describe())
print("\nFinetunes Model Perplexity:")
print(df["perplexity_finetuned"].describe())