# NVC_DeepSeek_R1_Distill_Llama_8B_DATA.py
# -*- coding: utf-8 -*-
"""
This script generates a dataset for training the DeepSeek model on chain-of-thought (CoT) reasoning
using Nonviolent Communication (NVC) principles. It includes:
1. A generator for user prompts (questions).
2. A generator for chain-of-thought responses based on NVC principles.
"""

from unsloth import FastLanguageModel
import torch
import csv
import pandas  # Import pandas

# --- 1. Load Model with Memory Optimization Parameters ---
max_seq_length = 2048
dtype = None  # Auto-detect dtype
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Enable faster inference for generation
FastLanguageModel.for_inference(model)

# --- 2. User Input Prompts for Data Generation ---
questions_df = pandas.read_csv("synthetic_questions_llm_nvc_real_scenario.csv") # Reads the CSV file
user_prompts = questions_df['question'].tolist() # Extracts questions into a list

# --- 3. NVC Chain-of-Thought Generation Prompt Template ---
nvc_cot_prompt_template = """
Respond based on Nonviolent Communication principles. Translate the user's input into feelings and needs, and generate a chain-of-thought response.

**Steps:**
1. Identify the feeling and need.
2. Co-create a request with the speaker.
3. Formulate a sentence according to NVC principles.

**Rules:**
- Do not give advice.
- Use only real feelings (e.g., sadness, frustration) and universal needs (e.g., connection, understanding).
- Avoid pseudo-feelings (e.g., rejected, attacked).
- Keep responses under 100 words.

**User Input:**
{}

**Chain of Thought:**
"""

# --- 4. Generate Chain-of-Thought Responses ---
def generate_cot_response(user_input):
    prompt_text = nvc_cot_prompt_template.format(user_input)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device) # Use model.device instead of "cuda"

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,  # Limit response length
        use_cache=True,
        temperature=0.7,  # Add some randomness
        top_p=0.9,
    )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extract only the chain-of-thought part, after the prompt
    response_start_index = response.find("Chain of Thought:")
    if response_start_index != -1:
        cot_response = response[response_start_index + len("Chain of Thought:"):].strip()
    else:
        cot_response = response.strip()  # If prompt format is slightly off

    return cot_response

# --- 5. Generate Dataset and Save to CSV ---
output_data = []
for user_input in user_prompts:
    cot_response = generate_cot_response(user_input)
    print(f"**User Input:** {user_input}")
    print(f"**Chain of Thought:** {cot_response}\n")
    output_data.append({"instruction": user_input, "chain_of_thought": cot_response})

# Save to CSV
csv_filename = "nvc_cot_dataset.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["instruction", "chain_of_thought"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(output_data)

print(f"Dataset generated and saved to '{csv_filename}'.")
print("NVC Chain-of-Thought Dataset Generation Completed.")
