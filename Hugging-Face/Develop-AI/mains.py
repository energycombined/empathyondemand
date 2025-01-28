import os
import threading
import torch
import gradio as gr

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)
from peft import PeftModel, PeftConfig

import spaces
print(f"Is CUDA available: {torch.cuda.is_available()}")
# True
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

###############################################################################
# Model and Tokenizer Loading
###############################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv('HF_TOKEN')
HF_TOKEN_ORG = os.getenv('HF_TOKEN_ORG')
BASE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ADAPTER_NAME = "ai-medical/fine_tuned_deepseek_v2_empathy"

try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True, token=HF_TOKEN)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_auth_token=HF_TOKEN
    ).to(device)
    peft_config = PeftConfig.from_pretrained(ADAPTER_NAME, use_auth_token=HF_TOKEN_ORG)
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_NAME,
        config=peft_config,
        torch_dtype=torch.float16
    ).to(device)
except Exception as e:
    raise RuntimeError(f"Model or adapter loading error: {e}")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Explicitly set max_seq_length
model.config.max_seq_length = 2048  # Adjust as needed

###############################################################################
# System Prompt (No extra "###" tokens)
###############################################################################

SYSTEM_PROMPT = """
You are a compassionate and empathetic assistant trained to help users explore their emotions and unmet needs.
Your goal is to respond like a professional doctor, providing thoughtful and accurate answers that help users
reflect on their feelings and situations. Always approach the user with care, understanding, and professionalism.
"""

"""
When users share experiences or evaluative words, respond empathetically by:
1. Identifying associated feelings (e.g., sadness, frustration, joy).
2. Connecting those feelings to possible unmet needs (e.g., trust, respect, belonging).
3. Encouraging users to reflect further by asking gentle, open-ended questions.

**Guidelines:**
- Use clear, concise, and natural language in your responses.
- Analyze the user's input thoroughly before providing an answer.
- Always maintain a professional tone, as if you were a skilled and compassionate doctor.
- Provide responses that encourage exploration and self-awareness.

Always prioritize empathy, professionalism, and fostering a safe space for users to express themselves.
"""

###############################################################################
# Chat Function (Single Prompt → Single Answer)
###############################################################################
# Chat function
@spaces.GPU
def chat_with_model(prompt, history):
    """
    Generate a single response (no multi-turn context).
    Uses a streaming approach so partial output can be shown.
    """
    try:
        # Form the full context once for the single user prompt
        full_context = f"{SYSTEM_PROMPT}\nUser: {prompt}\nAI:"

        inputs = tokenizer(
            full_context,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.config.max_seq_length
        ).to(device)

        text_streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id
        )

        # Generate in a separate thread for streaming
        thread = threading.Thread(target=lambda: model.generate(**generation_kwargs))
        thread.start()

        partial_response = ""
        for new_token in text_streamer:
            # Accumulate tokens as they stream in
            partial_response += new_token

            # Real-time cleanup:
            # Remove any leftover references to "### Instruction," etc., if they appear
            for unwanted in ["### Instruction:", "### Input:", "### Response:"]:
                partial_response = partial_response.replace(unwanted, "")

            # Check if the chatbot has generated a question
            if "?" in partial_response:
                # Pause further token generation
                yield partial_response
                break

            yield partial_response

        thread.join()

    except Exception as e:
        yield f"Error: {str(e)}"

###############################################################################
# Custom CSS Styling
###############################################################################

css_content = '''
/* Ensure the main background and text colors adapt to the theme */
body {
  font-family: Arial, sans-serif;
}

/* Main container adjustments */
.gradio-container {
  max-width: 800px;
  margin: auto;
  padding: 20px;
}

/* Chatbox messages */
.chatbox .message {
  background: var(--background-color-primary);
  color: var(--text-color-primary);
  padding: 10px;
  border-radius: 8px;
  margin-bottom: 8px;
}

/* User messages can have a different background color if desired */
.chatbox .user {
  background: var(--background-color-secondary);
  color: var(--text-color-primary);
}

/* Textbox and button styling */
.textbox, .button {
  border-radius: 5px;
  padding: 10px;
  margin-top: 10px;
}

/* Button styling */
.button {
  background: var(--button-primary-background-fill);
  color: var(--button-primary-text-color);
  border: none;
  cursor: pointer;
}

.button:hover {
  background: var(--button-primary-background-fill-hover);
}
'''

def write_temp_css(css_str, filename="style.css"):
    with open(filename, "w") as f:
        f.write(css_str)

write_temp_css(css_content)

def read_css_from_file(filepath="style.css"):
    with open(filepath, "r") as f:
        return f"<style>\n{f.read()}\n</style>"

css = read_css_from_file()

###############################################################################
# Gradio Interface
###############################################################################

welcome_message = """<div style='text-align:center;'>
<h2>AI Chatbot</h2>
<p>Ask a single question to receive a single empathetic answer.</p>
</div>"""
with gr.Blocks(theme=gr.themes.Default(), css=css) as demo:
    gr.Markdown(welcome_message)
    chat_window = gr.Chatbot(label="Q&A History", elem_classes="chatbox", value=[])

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type your message here...",
            label="Your Prompt",
            elem_classes="textbox"
        )
        submit_button = gr.Button("Submit", elem_classes="button")

    def update_chat_window(prompt, history):
        # Even though we keep a "chat window," each new prompt → new single response
        # No multi-turn logic or reusing of older conversation
        history = history or []
        history.append([prompt, ""])  # Add new user query with empty AI response

        partial_response = ""
        for new_text in chat_with_model(prompt, history):
            partial_response = new_text
            # Update the last AI message in the display
            history[-1][1] = partial_response
            yield history, ""  # Clear the input field after submission

        # If the chatbot generated a question, wait for the user to respond
        if "?" in partial_response:
            yield history, ""  # Clear the input field after submission

    submit_button.click(
        update_chat_window,
        inputs=[user_input, chat_window],
        outputs=[chat_window, user_input]  # Clear the input field after submission
    )
    user_input.submit(
        update_chat_window,
        inputs=[user_input, chat_window],
        outputs=[chat_window, user_input]  # Clear the input field after submission
    )

demo.launch(debug=True)