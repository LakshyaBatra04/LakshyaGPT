from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define the exact local model path
local_model_path = r"D:\huggingface_cache\hub\models--mistralai--Mistral-7B-v0.1\snapshots\7231864981174d9bee8c7687c24c8344414eae6b"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,  # Use FP16 for lower memory usage
    device_map="auto" , # Automatically use GPU if available
    low_cpu_mem_usage=True

)

# Function to generate responses
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output = model.generate(**inputs, max_new_tokens=100, temperature=0.7, top_p=0.9)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
prompt = "What is the significance of AI in healthcare?"
response = generate_response(prompt)
print(response)
