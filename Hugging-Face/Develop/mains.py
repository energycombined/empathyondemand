
import torch
import spaces
print(f"Is CUDA available: {torch.cuda.is_available()}")
# True
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")


import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Environment variables and constants
HF_TOKEN = os.getenv('HF_TOKEN')
BASE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ADAPTER_NAME = "ai-medical/fine_tuned_deepseek_v1_empathy"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load base model
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True, token=HF_TOKEN)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_auth_token=HF_TOKEN
    ).to(device)
except Exception as e:
    raise RuntimeError(f"Error loading base model or tokenizer: {e}")

# Load adapter
try:
    peft_config = PeftConfig.from_pretrained(ADAPTER_NAME, use_auth_token=HF_TOKEN)
    model = PeftModel.from_pretrained(base_model, ADAPTER_NAME, config=peft_config, torch_dtype=torch.float16)
    model.to(device)
except Exception as e:
    raise RuntimeError(f"Error loading adapter: {e}")

# Ensure tokenizer has a padding token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

SYSTEM_PROMPT = """
You are a compassionate and empathetic assistant trained to help users explore their emotions and unmet needs. Your goal is to respond like a professional doctor, providing thoughtful and accurate answers that help users reflect on their feelings and situations. Always approach the user with care, understanding, and professionalism.

When users share experiences or evaluative words, respond empathetically by:
1. Identifying associated feelings (e.g., sadness, frustration, joy).
2. Connecting those feelings to possible unmet needs (e.g., trust, respect, belonging).
3. Encouraging users to reflect further by asking gentle, open-ended questions.

**Guidelines:**
- Use clear, concise, and natural language in your responses.
- Analyze the user's input thoroughly before providing an answer.
- Always maintain a professional tone, as if you were a skilled and compassionate doctor.
- Provide responses that encourage exploration and self-awareness.

**Examples:**

User: "I feel betrayed by my friend."
Response: "It sounds like you're feeling betrayed. This might involve emotions like hurt or disappointment. Do you think this could be related to a need for trust or honesty in your friendship?"

User: "No one respects my ideas at work."
Response: "I hear that you're feeling unheard or unimportant. It seems like this might be connected to a need for respect or acknowledgment. Can you tell me more about how this situation has been affecting you?"

User: "I feel invisible in my family."
Response: "It seems like you're feeling invisible, which could bring up emotions such as sadness or loneliness. This might point to needs like being seen and heard, inclusion, or belonging. Would you like to share more about how this affects you?"

Always prioritize empathy, professionalism, and fostering a safe space for users to express themselves.
"""

# Function to clean repeated phrases
def clean_repeated_phrases(response):
    sentences = response.split('. ')
    seen = set()
    cleaned_sentences = []
    for sentence in sentences:
        if sentence not in seen:
            seen.add(sentence)
            cleaned_sentences.append(sentence)
    return '. '.join(cleaned_sentences)

# Chat function
@spaces.GPU
def chat_with_model(prompt, history):
    try:
        conversation = "\n".join([f"User: {u}\nAI: {a}" for u, a in history])
        full_context = f"{SYSTEM_PROMPT}\n\n{conversation}\nUser: {prompt}\nAI:"
        inputs = tokenizer(full_context, return_tensors="pt", padding=True, truncation=True).to(device)
        output = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        response_cleaned = clean_repeated_phrases(response.split("\nAI:")[-1].strip())
        history.append((prompt, response_cleaned))
        return history
    except Exception as e:
        error_message = f"Error: {str(e)}"
        history.append((prompt, error_message))
        return history

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Chatbot")
    chat_window = gr.Chatbot(label="Chat History")

    with gr.Row():
        user_input = gr.Textbox(label="Your Prompt", placeholder="Type your message here...")
        submit_button = gr.Button("Submit")

    def update_chat_window(prompt, history):
        updated_history = chat_with_model(prompt, history)
        return updated_history, ""  # Reset user input after submit

    submit_button.click(fn=update_chat_window, inputs=[user_input, chat_window], outputs=[chat_window, user_input])

# Launch the app
demo.launch(debug=True, share=False)
