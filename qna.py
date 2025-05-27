from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
import bitsandbytes as bnb  # Import bitsandbytes for 4-bit quantization

# Check if GPU is available and assign the appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically choose CUDA if available, otherwise CPU

# Load the model and processor
model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
processor = LlavaProcessor.from_pretrained(model_id)

# Enable 4-bit quantization and FlashAttention 2, and move the model to the appropriate device
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use float16 precision
    low_cpu_mem_usage=True,     # Optimize CPU memory usage
    load_in_4bit=True,          # Enable 4-bit quantization using bitsandbytes
    use_flash_attention_2=True  # Enable FlashAttention 2 for faster attention calculation
).to(device)  # Move model to either CUDA or CPU based on availability

def generate_answer_from_frames(prompt, video_frames):
    """
    Generate an answer using Llava model from a given prompt and video frames.
    
    Parameters:
    - prompt (str): The question to ask.
    - video_frames (list): List of frames from the video.

    Returns:
    - answer (str): The generated answer from the model.
    """
    # Ensure the number of tokens matches the number of frames
    toks = "<image>" * len(video_frames)
    print(f"Number of video frames: {len(video_frames)}")

    # Add explicit instructions to the conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Please answer the following question based on the images shown below. {prompt}"},
                *[{"type": "image"} for _ in video_frames]  # Add frames as images
            ],
        }
    ]

    # Apply the chat template to format the conversation
    formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Preprocess inputs
    inputs = processor(text=formatted_prompt, images=video_frames, return_tensors="pt").to(device, torch.float16)

    # Generate the model's answer
    output = model.generate(**inputs, max_new_tokens=500, do_sample=False)

    # Decode and return the output
    answer = processor.decode(output[0][2:], skip_special_tokens=True)
    return answer[len(prompt) + 10:]  # Removing the prompt portion
