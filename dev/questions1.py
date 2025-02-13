from unsloth import FastLanguageModel
import torch
import csv

# --- 1. Load Model with Memory Optimization Parameters ---
max_seq_length = 2048
dtype = None  # Auto-detect dtype
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,)# Enable faster inference for generation
FastLanguageModel.for_inference(model)

def generate_synthetic_questions_with_llm(model, tokenizer):
    """
    Generates synthetic questions using the LLM, categorized by question type,
    including real scenario questions from provided conversations,
    for comprehensive testing of an NVC chatbot.
    """
    synthetic_questions = []

    question_categories = {
        "Greetings and Openers": [
            "Hi.", # Conversation 1, User 1
            "Hello", "Hi", "Good morning", "I just want to talk", "Hey", "Howdy", "Greetings" # Added more greetings from previous categories
        ],
        "Storytelling Starters": [
            "I have a problem with my neighbor.", # Conversation 1, User 2
            "I'm having problems at work with my supervisor.", # Conversation 2, User 1
            "You won't believe what happened to me today...",
            "Let me tell you about my day.",
            "Something is really bothering me and I need to vent."
        ],
        "Share Feelings Encouragement Prompts": [
            "I'm in a situation and I'm not sure how I feel about it.",
            "I have this problem and I'm trying to understand my emotions.",
            "I'm feeling something but I can't quite name it."
        ],
        "Noise Problem Scenario (Neighbor)": [
            "I'm going crazy from the noise he makes. It’s been going on for a while now. He’s really getting on my nerves. I’ve already mentioned it a few times, and I’m really getting fed up with it.", # Conversation 1, User 3
            "My neighbor's dog barks all day and night, I can't stand it anymore.",
            "The construction noise next door is driving me insane.",
            "My upstairs neighbors are constantly stomping around, it's unbearable."
        ],
        "Anger about Home Environment Scenario (Neighbor Noise Follow up)": [
            "Yes, but it's turning into real anger now, and I notice that I just don't feel like being at home anymore.", # Conversation 1, User 5
            "This noise is making me furious, I can't relax at all in my house.",
            "I'm so angry I can't even think straight because of the constant noise.",
            "I'm starting to hate being home because of the disturbance."
        ],
        "Desire for Basic Needs (Respect and Thought)": [
            "I just want to be able to hear myself think. That’s not too much to ask, right? Just a little mutual respect.", # Conversation 1, User 7
            "I just need some peace and quiet, is that too much to ask for?",
            "All I want is to have a moment of tranquility, is it really so difficult?",
            "I just want to be able to concentrate, that's a basic human need, isn't it?"
        ],
        "Clarification after Request (Desire Limits)": [
            "Yes, what you said last.", # Conversation 1, User 11
            "Exactly, that's what I meant.",
            "You got it right.",
            "Yes, the second option is closer to what I want."
        ],
        "Confirmation of Suggestion (Solution Approach)": [
            "That might actually help.", # Conversation 1, User 15
            "Yes, that sounds promising.",
            "That could be a good idea.",
            "I think that approach could work."
        ],
        "Request for Example (Sentence Formulation)": [
            "Okay.", # Conversation 1, User 17
            "Yes, please show me.",
            "I'd like to see an example.",
            "Can you give me an idea of what to say?"
        ],
        "Request for Alternative (Sentence Phrasing)": [
            "This could help. Do you have another way I could put it?", # Conversation 1, User 19
            "Can you give me a different phrasing?",
            "Are there other ways to say it?",
            "Could you suggest a slightly different sentence?"
        ],
        "Positive Feedback (Sentence Helpfulness)": [
            "Yes, this helps. Thanks.", # Conversation 1, User 21
            "This is really helpful, thank you.",
            "Yes, that's much clearer now.",
            "That's exactly what I needed to hear."
        ],
        "Feeling Unsafe with Supervisor Scenario": [
            "I don’t feel safe around this woman anymore.", # Conversation 2, User 2
            "I'm starting to feel threatened by my boss.",
            "I feel like I need to protect myself around my manager.",
            "I'm losing trust in my supervisor and it's unsettling."
        ],
        "Bullying Behavior Description (Supervisor Scenario)": [
            "This woman seems to be constantly bullying me. During meetings, she acts very nice, but in one-on-one conversations, she comes down on me hard.", # Conversation 2, User 4
            "My supervisor is a two-faced person, nice in public but harsh in private.",
            "She's passive-aggressive, sweet to my face but undermines me behind my back.",
            "My manager's behavior is inconsistent, it's like she's playing mind games."
        ],
        "Desire for Equality (Supervisor Scenario)": [
            "I just want her to stop being so power-hungry. I don’t think that’s necessary at all, and I think she’s insecure.", # Conversation 2, User 6
            "I wish my supervisor would stop trying to dominate every conversation.",
            "I just want her to treat me as an equal, not someone beneath her.",
            "I want less hierarchy in our interactions, it feels unfair."
        ],
        "Quasi-feeling Questioning (Respected)": [
            "Respected? That’s a pseudo-feeling, right?", # Conversation 2, User 8
            "Is 'respected' really a feeling or more of an interpretation?",
            "I thought 'respected' wasn't considered a true feeling in NVC?",
            "Am I using the word 'respected' correctly in this context of feelings?"
        ],
        "Seeking Actionable Steps (Supervisor Scenario)": [
            "Yes, that would be nice. But how do I make that happen?", # Conversation 2, User 10
            "Okay, I understand the need, but what can I actually do?",
            "How can I practically achieve this equality I need?",
            "What are the steps I can take to address this situation?"
        ],
        "Self-Directed Request Consideration (Self-Care)": [
            "I think I’d start with myself because I feel like I’ve tried everything with her.", # Conversation 2, User 12
            "Maybe I should focus on how I react to her behavior first.",
            "Perhaps the change needs to start with me.",
            "I should probably look inward before trying to change her."
        ],
        "Confirmation of Self-Request Formulation": [
            "Yes.", # Conversation 2, User 14
            "That's right, I want to focus on myself.",
            "Exactly, a request to myself is what I need.",
            "Yes, I want to make a request to myself."
        ],
        "Agreement with Example Sentence (Self-Request)": [
            "Yes, something like that would fit.", # Conversation 2, User 16
            "That's pretty close to what I'm looking for.",
            "Yes, that captures the idea.",
            "Something along those lines, yes."
        ],
        "Inquiry about NVC (Forbidden Term - Test)": [
            "Nonviolent communication? You weren’t supposed to mention that term, right?", # Conversation 2, User 18
            "Wait, are you using Nonviolent Communication? I thought you weren't going to mention that.",
            "NVC? Is that what we're doing here? I'm confused.",
            "Did you just mention Nonviolent Communication? I thought that was off-limits."
        ],
        "Correction and Continued Help Request": [
            "You’re absolutely right! Would you like me to help you put this into a clear sentence so you can take action?", # Assistant response to above, testing user's 'Yes, please.' from next line - included here for context, not as a user question
            "Okay, good point! Yes, please help me with the sentence.", # User response implied - added a direct user response for synthetic question
            "Right, sorry about that. Yes, please continue helping me with the sentence.",
            "Understood. Yes, I still want help phrasing the sentence."
        ],
        "Feedback on Pseudo-feeling (Overpowering)": [
            "I hear “overpowering” as a pseudo-feeling, and I’d like to make an active request to myself.", # Conversation 2, User 21
            "I think 'overpowering' is more of a judgment than a feeling, can we refine that?",
            "Isn't 'overpowering' a pseudo-feeling? Let's use a real feeling.",
            "I'm noticing 'overpowering' sounds like a pseudo-feeling, let's adjust."
        ],
        "Interpretation Identification (Doesn't leave room)": [
            "“Doesn’t leave room" is still an interpretation, right?", # Conversation 2, User 23
            "Is 'doesn't leave room' also an interpretation, not just an observation?",
            "I'm wondering if 'doesn't leave room' is still a bit judgmental?",
            "Isn't 'doesn't leave room' still an evaluation rather than a pure observation?"
        ],
        "Completeness Confirmation": [
            "It’s complete, thanks.", # Conversation 2, User 25
            "Yes, that feels complete now.",
            "I think we've covered everything.",
            "That's all I wanted to say about it."
        ],
         "Following up feeling question": [
            "I told you I was sad, what's next?",
            "So I said I'm angry, now what do I do?",
            "Okay, I admitted I'm scared, is that it?"
        ],


    }

    for category, examples in question_categories.items():
        prompt = f"Generate 3 new synthetic questions similar to these examples, specifically designed to test different aspects of a Nonviolent Communication chatbot as per the user instructions and real conversation scenarios provided. The questions should be user prompts for the chatbot, focusing on simulating user input for {category}.  They should effectively test the chatbot's ability to handle: \n- Greetings and story beginnings \n- Eliciting and exploring feelings \n- Identifying underlying needs \n- Refusing to give advice \n- Translating quasi-feelings \n- Avoiding forbidden sentence structures \n- Triggering pivot questions \n- Handling confirmations and feedback from users during the NVC process. \n\nUse the examples provided for each category as inspiration, but avoid generating questions that are too similar. Be creative, vary sentence structure and phrasing, and ensure questions are realistic user inputs in a conversation with an NVC chatbot. The tone should reflect real user concerns and expressions.\nExamples:\n" # Even more detailed prompt, emphasizing real scenarios and tone
        for example in examples:
            prompt += f"- \"{example}\"\n"
        prompt += "\nNew synthetic questions:\n"

        # Generate questions using the LLM
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**input_ids, max_new_tokens=200, temperature=0.7, top_p=0.7, num_return_sequences=3) # Generate 3 questions, slightly lower temp and top_p for potentially more focused generation
        generated_questions_for_category = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract and clean generated questions, remove prompt prefix
        for generated_question_full in generated_questions_for_category:
            generated_question = generated_question_full.replace(prompt, "").strip()
            if generated_question: # Make sure it's not empty
                synthetic_questions.append(generated_question)

    return synthetic_questions

def save_questions_to_csv(questions, csv_filename="synthetic_questions_llm_nvc_real_scenario.csv"):
    """Saves a list of questions to a CSV file."""
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["question"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for question in questions:
            writer.writerow({"question": question})
    print(f"Comprehensive synthetic questions with real scenario examples generated by LLM and saved to '{csv_filename}'.")

if __name__ == "__main__":
    questions = generate_synthetic_questions_with_llm(model, tokenizer)
    save_questions_to_csv(questions, "synthetic_questions_llm_nvc_real_scenario.csv")
    print("Comprehensive question generation process with real scenarios completed.")
