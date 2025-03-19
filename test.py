from transformers import pipeline
import torch
print("eee")
pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it",
    device="cuda",
    torch_dtype=torch.bfloat16,
    cache_dir="D:/huggingface_cache"  # Change this to your preferred directory
)
print("fucky")

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    }
]

output = pipe(text=messages)
print(output[0][0]["generated_text"][-1]["content"]+"1") 

# Okay, let's take a look! 
# Based on the image, the animal on the candy is a **turtle**. 
# You can see the shell shape and the head and legs.
