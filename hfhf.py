from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-v0.1" 
save_path = "./mistral-7b-local" 

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model.save_pretrained(save_path)
print("Model saved")
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)

print(f"Model and tokenizer saved at: {save_path}")
