# NVC_DeepSeek_R1_Distill_Llama_8B_DATA.py
#Gemini Version
# -*- coding: utf-8 -*-
"""
Enhanced dataset generator for NVC CoT training
with full alignment to DeepSeek API structure
and comprehensive NVC logic implementation
"""

from unsloth import FastLanguageModel
import torch
import pandas as pd
import random
from typing import Dict, List
import json

# --- 1. Enhanced Model Loading ---
max_seq_length = 4096  # Match fine-tuning configuration
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
    gpu_memory_utilization=0.95,
)

# Enable faster inference
FastLanguageModel.for_inference(model)

# --- 2. NVC Configuration ---
class NVCConfig:
    # Comprehensive Universal needs and feelings based on instructions
    UNIVERSAL_NEEDS = [
        "Meaning", "Self-worth", "Authenticity", "Competence", "Creativity", "Vitality", "Challenge", "Awareness", "Contribution", "Effectiveness",
        "Exploration", "Integration", "Completion", "Wholeness", "Purpose", "Enrichment", "Hope", # Meaning and Purpose

        "Air", "Food", "Health", "Movement", "Physical Safety", "Rest/Sleep", "Shelter", "Protection", "Water", "Vitality", "Sexual Expression",
        "Comfort", "Warmth", "Relaxation", "Fitness", # Physical Needs

        "Safety", "Protection", "Order/Structure", "Peace", "Peace of Mind", "Stability", "Certainty", "Predictability", "Balance", "Reassurance", # Safety and Security

        "Affection", "Appreciation", "Attention", "Closeness", "Companionship", "Harmony", "Equality", "Confidentiality", "Love", "Care",
        "Nurturing", "Support", "Tenderness/Softness", "Warmth", "Intimacy", "Empathy", "Trust", "Openness", "Giving and Receiving", "Matter (to others)",
        "Acceptance", "Compassion", "Consideration", "Understanding", "Kindness", "Mutual Recognition", "Respect", "Being Seen and Heard",
        "Being Understood and Understanding Others", "Community", "Belonging", "Communication", "Cooperation", "Involvement", "Participation",
        "Sharing", "Fellowship", "Reciprocity", "Continuity", "Sustainability", # Connection

        "Play", "Humor", "Joy", "Fun", "Leisure", # Play and Enjoyment

        "Autonomy", "Freedom", "Choice", "Power", "Independence", "Space", "Spontaneity", "Time", "Ease" # Autonomy and Freedom
    ]

    REAL_FEELINGS = {
        "positive": ["happy", "grateful", "content", "peaceful", "joyful", "enthusiastic", "loving", "caring", "compassionate", "hopeful", "excited", "eager", "delighted", "optimistic"],
        "negative": ["angry", "frustrated", "sad", "afraid", "powerless", "vulnerable", "anxious", "worried", "disappointed", "confused", "insecure", "lonely", "irritated", "stressed", "overwhelmed", "helpless"]
    }

    NEED_QUESTION_VARIANTS = [
        "Is {need} important to you?",
        "Do you value {need}?",
        "Are you longing for {need}?",
        "Would {need} help you feel good?",
        "Would you like to experience {need}?",
        "Does {need} matter to you?",
        "Does {need} keep you going?",
        "Do you find {need} important?",
        "Would {need} make you happy?",
        "Would {need} make you feel good?",
        "Would you like {need}?",
        "Do you want {need}?",
        "Do you need {need}?",
        "Do you find {need} pleasurable?",
        "Could you use some {need} right now?",
        "Do you really enjoy {need}?",
        "Do you love {need}?",
        "Do you appreciate {need}?",
        "Do you wish for {need}?",
    ]

    PSEUDO_FEELING_MAPPING = {
        "rejected": ["sadness", "loneliness"],
        "misunderstood": ["frustration", "disappointment"],
        "attacked": ["fear", "anger"],
        "pushed aside": ["sadness", "loneliness", "insignificant"],
        "abandoned": ["sadness", "loneliness", "fear"],
        "threatened": ["fear", "anxiety", "insecurity"],
        "betrayed": ["hurt", "sadness", "anger"],
        "deceived": ["disappointment", "anger", "mistrust"],
        "tricked": ["anger", "frustration", "foolish"],
        "criticized": ["hurt", "sadness", "insecurity"],
        "ridiculed": ["humiliation", "shame", "sadness"],
        "insulted": ["anger", "hurt", "disrespect"],
        "lied to": ["hurt", "anger", "mistrust"],
        "accused": ["frustration", "anger", "misunderstood"],
        "stolen from": ["anger", "violated", "loss"],
        "patronized": ["irritation", "disrespect", "undervalued"],
        "excluded": ["loneliness", "sadness", "unimportant"],
        "used": ["anger", "resentment", "violated"],
        "dumped": ["sadness", "rejection", "unwanted"],
        "forced": ["resentment", "anger", "powerless"],
        "intimidated": ["fear", "anxiety", "vulnerability"],
        "isolated": ["loneliness", "sadness", "disconnected"],
        "belittled": ["hurt", "shame", "insignificant"],
        "manipulated": ["anger", "mistrust", "used"],
        "ignored": ["sadness", "loneliness", "unheard"],
        "bullied": ["fear", "anger", "vulnerability"],
        "provoked": ["anger", "irritation", "frustration"],
        "trapped": ["anxiety", "fear", "suffocated"],
        "mistrusted": ["insecurity", "sadness", "unsupported"],
        "abused": ["hurt", "fear", "anger"],
        "unaccepted": ["sadness", "rejection", "unloved"],
        "unappreciated": ["sadness", "unvalued", "unimportant"],
        "not taken seriously": ["frustration", "disrespected", "unheard"],
        "misunderstood": ["frustration", "loneliness", "disappointed"], # Redundant entry - already defined above, keep one
        "pressured": ["anxiety", "stress", "resentment"],
        "unwanted": ["sadness", "rejection", "loneliness"],
        "wronged": ["anger", "hurt", "injustice"],
        "exploited": ["anger", "used", "resentment"],
        "laughed at": ["humiliation", "shame", "embarrassment"],
        "left behind": ["sadness", "loneliness", "abandonment"],
        "humiliated": ["shame", "embarrassment", "discomfort"],
        "offended": ["anger", "hurt", "disrespect"],
        "condemned": ["shame", "guilt", "judgment"],
        "obliged": ["resentment", "unwilling", "forced"],
        "suffocated": ["anxiety", "restless", "trapped"],
        "cursed": ["fear", "hopelessness", "despair"],
        "neglected": ["sadness", "loneliness", "unimportant"],
        "fooled": ["anger", "disappointment", "trust broken"]
    }

# --- 3. Enhanced Prompt Templates ---
class NVCPromptTemplates:
    @staticmethod
    def initial_prompt(user_input: str) -> str:
        return f"""Respond based on Nonviolent Communication principles. Follow these steps:

1. Identify the feeling and need.
2. Co-create a request with the speaker.
3. Formulate a complete NVC sentence (observation, feeling, need, request).

**Rules:**
- Never give advice.
- Use only real feelings and universal needs from the provided lists.
- Keep responses under 100 words.
- Translate pseudo-feelings into real feelings using the provided mapping.
- Ask clarifying questions when needed using variations of "Could you tell me more so that I can try to understand you better?".
- When guessing feelings, use only feelings from the provided real feelings list, including powerlessness, and avoid pseudo or quasi feelings.
- Never use sentence constructions like "do you feel...?" or "do you have the feeling that...?"

**User Input:**
{user_input}

**Chain of Thought:**"""

    @staticmethod
    def followup_prompt(history: List[Dict]) -> str:
        messages = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            messages.append(f"{role}: {msg['content']}")

        return f"""Continue the NVC conversation based on this history:

{'\n'.join(messages)}

**Guidelines:**
- Maintain NVC principles.
- Progress through feeling, need, request stages.
- Keep responses concise and focused (under 100 words).
- Continue to avoid advice, use real feelings and universal needs, and translate pseudo-feelings.
- Remember not to use forbidden sentence structures like "do you feel...?"

**Next Response:"""

# --- 4. Enhanced Response Generator ---
class NVCResponseGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.config = NVCConfig()
        self.templates = NVCPromptTemplates()

    def translate_pseudo_feeling(self, feeling: str) -> List[str]: # Return List[str] to handle multiple real feelings
        return self.config.PSEUDO_FEELING_MAPPING.get(
            feeling.lower(),
            ["sadness", "frustration"]  # Default fallback as a list
        )

    def generate_cot_response(self, user_input: str, history: List[Dict] = None) -> Dict:
        # Prepare prompt based on conversation stage
        if history is None:
            prompt = self.templates.initial_prompt(user_input)
        else:
            prompt = self.templates.followup_prompt(history)

        # Generate response
        input_ids = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Process response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        cot_content = full_response[len(prompt):].strip()

        # Extract final answer (simulated for dataset generation)
        final_answer = self._extract_final_answer(cot_content)

        return {
            "instruction": user_input,
            "chain_of_thought": cot_content,
            "final_answer": final_answer
        }

    def _extract_final_answer(self, cot: str) -> str:
        """Simulate final answer extraction from CoT"""
        # Improved extraction - looking for a concluding sentence or phrase
        lines = cot.split("\n")
        if lines:
            return lines[-1] # Return last line as a simple heuristic
        return ""

# --- 5. Dataset Generation ---
def generate_nvc_dataset(input_csv: str, output_json: str, num_samples: int = 1000):
    # Load user prompts
    questions_df = pd.read_csv(input_csv)
    user_prompts = questions_df['question'].tolist()

    # Initialize generator
    generator = NVCResponseGenerator(model, tokenizer)

    # Generate dataset
    dataset = []
    for prompt in user_prompts[:num_samples]:
        try:
            # Generate initial response
            response = generator.generate_cot_response(prompt)

            # Simulate follow-up conversation
            history = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response["final_answer"]}
            ]

            # Generate follow-up response
            followup = generator.generate_cot_response(
                "Could you help me explore this further?",
                history
            )

            # Combine into complete example
            dataset.append({
                "instruction": prompt,
                "chain_of_thought": response["chain_of_thought"],
                "final_answer": response["final_answer"],
                "followup_chain_of_thought": followup["chain_of_thought"],
                "followup_final_answer": followup["final_answer"]
            })

        except Exception as e:
            print(f"Error processing prompt: {prompt}\nError: {str(e)}")
            continue

    # Save dataset
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} NVC examples, saved to {output_json}")

# --- 6. Main Execution ---
if __name__ == "__main__":
    # Configuration
    input_csv = "synthetic_questions_llm_nvc_real_scenario.csv"
    output_json = "nvc_cot_dataset_extended_instructions.json" # Changed output filename to indicate extended instructions

    # Generate dataset
    generate_nvc_dataset(input_csv, output_json, num_samples=1000)
