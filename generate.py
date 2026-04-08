from peft import PeftModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# load base model and tokenizer
base_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# load LoRA weights on top of base model
model = PeftModel.from_pretrained(base_model, "./gpt2-finetuned-fedspeeches")
model.eval()

# now use pipeline with the explicitly loaded model
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

output = generator(
    "The Federal Reserve remains committed to",
    max_length=100,
    num_return_sequences=1
)

print(output[0]["generated_text"])