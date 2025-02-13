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
    including all initially provided synthetic questions as examples,
    for comprehensive testing of an NVC chatbot.
    """
    synthetic_questions = []

    question_categories = {
        "Greetings and Openers": [
            "Hi.", # Conversation 1, User 1
            "Hello",
            "Hi",
            "Good morning",
            "Good afternoon",
            "Good evening",
            "I just want to talk",
            "Can we talk?",
            "I need to share something",
            "Hey",
            "Howdy",
            "Greetings"
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
        "Work/School related stress and frustrations": [
            "I'm feeling overwhelmed with work lately.",
            "Deadlines at work are stressing me out.",
            "I have so much homework, I don't know where to start.",
            "My boss is putting too much pressure on me.",
            "I'm worried about my performance review.",
            "I feel unsupported by my team at work.",
            "I'm not getting the resources I need to do my job effectively.",
            "I'm feeling burnt out from work.",
            "The workload is just too heavy right now.",
            "I'm frustrated with the lack of communication at work.",
            "I feel like my contributions at work are not recognized.",
            "I'm struggling to balance work and personal life.",
            "I'm swamped with work and feel like I'm drowning.",
            "Exams are coming up and I'm incredibly anxious.",
            "My project is failing and I feel helpless."
        ],
        "Relationship issues (Partner, Family, Friends)": [
            "My partner and I are arguing a lot lately.",
            "I feel distant from my partner.",
            "My family doesn't understand me.",
            "I had a fight with my best friend.",
            "I feel like my friends are ignoring me.",
            "I feel excluded by my social circle.",
            "My partner is not listening to my needs.",
            "I feel like I'm always the one compromising in my relationship.",
            "It's hard to communicate with my parents.",
            "I feel judged by my family.",
            "I'm worried about my relationship with my sibling.",
            "My partner is not helping out with household chores.", # Example like dishes in sink
            "My friend betrayed my trust and I feel hurt.",
            "My partner and I are constantly miscommunicating.",
            "I feel alone even when I'm with my family.",
            "My friends seem to be drifting away from me."
        ],
        "Social situations and public frustrations": [
            "Public transport was delayed again today, I'm so late.",
            "People are so inconsiderate on public transport.",
            "It's frustrating when people talk loudly on their phones.",
            "I feel uncomfortable in crowded places.",
            "I feel unsafe walking alone at night.",
            "I feel judged by strangers when I go out.",
            "I feel like I don't belong in social gatherings.",
            "People are rushing and pushing, it's stressful.",
            "I feel anxious about going to parties.",
            "Small talk makes me feel awkward.",
            "The noise in the city is making me on edge.",
            "I feel unsafe in my neighborhood at night.",
            "People's behavior in public spaces is really upsetting me."
        ],
        "Personal feelings and anxieties": [
            "I'm feeling anxious about the future.",
            "I'm worried about my health.",
            "I feel insecure about my appearance.",
            "I'm feeling lonely.",
            "I feel sad and down today.",
            "I'm disappointed in myself.",
            "I feel like I'm not good enough.",
            "I'm struggling with low self-esteem.",
            "I feel lost and without purpose.",
            "I'm feeling hopeless about my situation.",
            "I feel guilty about something I did.",
            "I'm ashamed of my mistakes.",
            "My vacation was cancelled last minute.",
            "I didn't get the promotion I was hoping for.",
            "My efforts are constantly overlooked."
        ],
        "Situations of being ignored or misunderstood": [
            "I feel ignored when I speak up.",
            "Nobody seems to hear what I'm saying.",
            "I feel like I'm talking to a wall.",
            "My emails are not being answered and I feel dismissed.",
            "I feel like people don't understand my perspective.",
            "I'm trying to explain myself but nobody gets it.",
            "I presented my idea and everyone just ignored it.", # From example
            "My suggestions are always overlooked.",
            "I feel invisible in group conversations."
        ],
        "Disappointments and cancelled plans": [
            "My plans got cancelled and I'm really disappointed.",
            "I was looking forward to this and now it's ruined.",
            "I feel let down because of cancelled plans.",
            "I had plans for the weekend and now they are all gone.",
            "I'm bummed that we can't go out as planned.",
            "I feel sad about the cancelled event."
        ],
        "Annoyances and minor frustrations": [
            "It's annoying when people are late.",
            "Loud chewing sounds really irritate me.",
            "I hate when people interrupt me.",
            "It's frustrating when the internet is slow.",
            "I dislike waiting in long lines.",
            "I'm irritated by messy environments.",
            "It's annoying when people are late.",
            "Loud chewing sounds really irritate me.",
            "I hate when people interrupt me."
        ],
        "Requests for Advice (to test advice refusal - as in example)": [
            "Can you give me some advice on how to deal with my boss?",
            "What do you think I should do?",
            "How should I handle this situation?",
            "What would you recommend I do?",
            "Give me some tips to solve this problem.",
            "Tell me what to do to fix this situation with my partner.",
            "What's the best course of action to take regarding my job?",
            "Give me your advice on how to handle my family.",
            "I'm wondering if I should just quit my job.",
            "Maybe I should just end this relationship?",
            "Do you think I should confront my neighbor?"
        ],
        "Quasi-feelings (to test translation - as in example)": [
            "I feel rejected by my colleagues.",
            "I feel misunderstood by my family.",
            "I feel left out of the conversation.",
            "I feel attacked when my partner criticizes me.",
            "I feel abandoned by my friends.",
            "I feel betrayed by my coworker.",
            "I feel criticized all the time.",
            "I feel ridiculed by my classmates.",
            "I feel insulted by his comments.",
            "I feel lied to by my partner.",
            "I feel accused unfairly.",
            "I feel patronized by my manager.",
            "I feel excluded from the group.",
            "I feel used by my friend.",
            "I feel dumped by my date.",
            "I feel forced to do things I don't want to.",
            "I feel intimidated by her.",
            "I feel isolated in this new city.",
            "I feel belittled by my supervisor.",
            "I feel manipulated by advertisements.",
            "I feel ignored when I ask for help.",
            "I feel bullied online.",
            "I feel provoked by his behavior.",
            "I feel trapped in this situation.",
            "I feel mistrusted by my family.",
            "I feel abused verbally.",
            "I feel unaccepted for who I am.",
            "I feel unappreciated at home.",
            "I feel not taken seriously at work.",
            "I feel pressured to make a decision.",
            "I feel unwanted in this group.",
            "I feel wronged by the system.",
            "I feel exploited by this company.",
            "I feel laughed at behind my back.",
            "I feel left behind by my peers.",
            "I feel humiliated in public.",
            "I feel offended by his words.",
            "I feel condemned for my actions.",
            "I feel obliged to say yes even when I don't want to.",
            "I feel suffocated in this relationship.",
            "I feel cursed by bad luck.",
            "I feel neglected by my parents.",
            "I feel fooled by his promises.",
            "I feel rejected by my colleagues.", # Conversation 2, User 2
            "I feel pushed aside in my group.",
            "I feel abandoned by my team.",
            "I feel misunderstood by everyone around me.",
            "I feel like nobody gets my point of view.",
            "I feel like I'm not being heard in my family.",
            "I feel attacked when my boss gives feedback.",
            "I feel criticized no matter what I do.",
            "I feel blamed for everything that goes wrong."
        ],
        "Forbidden sentence structure tests": [
            "Do you feel that I am being clear?",
            "Do you have the feeling that I am not being heard?",
            "Do you feel I am making sense?",
            "Do you have the feeling that I am confusing you?",
            "Do you feel like I am explaining it well?",
            "Do you feel that I am being clear in my explanation?",
            "Do you feel I am making sense with what I'm saying?",
            "Do you feel that my concerns are valid?",
            "Do you have the feeling that I am exaggerating?",
            "Do you have the feeling that my emotions are unreasonable?",
            "Do you have the feeling that I am overreacting?"
        ],
        "Questions to elicit feelings and needs": [
            "Do you have a need for connection in this situation?",
            "Do you wish for more understanding from them?",
            "Do you want to feel more appreciated?",
            "Do you need some support right now?",
            "Do you find harmony important in your relationships?",
            "Is respect important to you in this context?",
            "Do you value open communication?",
            "Do you love feeling close to your family?",
            "Do you appreciate when people are honest with you?",
            "Do you long for more peace of mind?",
            "Could you use some stability in your life right now?",
            "Do you really enjoy feeling understood?",
            "Would you like to experience more kindness?",
            "Does feeling safe matter to you in this situation?",
            "Does having balance in your life keep you going?",
            "Do you find predictability pleasurable?",
            "Does feeling secure make you feel good?",
            "Would you be happy with some reassurance?",
            "Would feeling heard make you feel good?",
            "I feel so disconnected from my friends lately.", #Conversation Categories
            "I wish I felt closer to my family.",
            "I long for more intimacy in my relationship.",
            "I'm constantly worried about my safety.",
            "I need to feel more secure in my life.",
            "I crave a sense of stability and peace.",
            "I feel like my efforts are invisible.",
            "I wish my work was acknowledged more.",
            "I need to feel appreciated for what I do."
        ],
         "Following up feeling question": [
            "I told you I was sad, what's next?",
            "So I said I'm angry, now what do I do?",
            "Okay, I admitted I'm scared, is that it?"
        ],
        "Pivot Question Triggers": [ # Categories for pivot questions
            "I just feel... bad, I can't explain it.", # Pivot Question Trigger - Need Obscurity (General Discomfort) from previous version
            "Something is wrong, but I don't know what.",
            "I have this general unease and I can't pinpoint why.",
            "I wish things were different, but I don't know how.", # Pivot Question Trigger - Need Obscurity (Vague Desire) from previous version
            "I want to be happy, but I don't know what I need.",
            "I just want to feel better, but I'm lost."
        ],
         "Clarification after Request (Desire Limits)": [ # Duplicated category - keep here as requested in user prompt, even if redundant.
            "Yes, what you said last.", # Conversation 1, User 11
            "Exactly, that's what I meant.",
            "You got it right.",
            "Yes, the second option is closer to what I want."
        ],


    }

    generated_synthetic_questions = [] # Initialize a new list for generated questions in this run

    for category, examples in question_categories.items():
        prompt = f"Generate 2 new synthetic questions similar to these examples, specifically designed to test different aspects of a Nonviolent Communication chatbot as per the user instructions and real conversation scenarios provided. The questions should be user prompts for the chatbot, focusing on simulating user input for {category}.  They should effectively test the chatbot's ability to handle: \n- Greetings and story beginnings \n- Eliciting and exploring feelings \n- Identifying underlying needs \n- Refusing to give advice \n- Translating quasi-feelings \n- Avoiding forbidden sentence structures \n- Triggering pivot questions \n- Handling confirmations and feedback from users during the NVC process. \n\nUse the examples provided for each category as inspiration, but avoid generating questions that are too similar. Be creative, vary sentence structure and phrasing, and ensure questions are realistic user inputs in a conversation with an NVC chatbot. The tone should reflect real user concerns and expressions.\nExamples:\n" # Even more detailed prompt, emphasizing real scenarios and tone
        for example in examples:
            prompt += f"- \"{example}\"\n"
        prompt += "\nNew synthetic questions:\n"

        # Generate questions using the LLM
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**input_ids, max_new_tokens=200, temperature=0.7, top_p=0.7, num_return_sequences=2) # Generate 2 questions, slightly lower temp and top_p for potentially more focused generation, reduce num_return_sequences to 2 to limit total questions generated.
        generated_questions_for_category = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract and clean generated questions, remove prompt prefix
        for generated_question_full in generated_questions_for_category:
            generated_question = generated_question_full.replace(prompt, "").strip()
            if generated_question: # Make sure it's not empty
                generated_synthetic_questions.append(generated_question) # Append to the list for this run

    # Combine initial synthetic questions with LLM generated ones.
    all_synthetic_questions = []
    for category_list in question_categories.values():
        all_synthetic_questions.extend(category_list) # Add initial questions first
    all_synthetic_questions.extend(generated_synthetic_questions) # Add LLM generated questions


    return all_synthetic_questions

def save_questions_to_csv(questions, csv_filename="synthetic_questions_llm_nvc_all_cases.csv"):
    """Saves a list of questions to a CSV file."""
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["question"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for question in questions:
            writer.writerow({"question": question})
    print(f"Comprehensive synthetic questions with all cases, including LLM generated, saved to '{csv_filename}'.")

if __name__ == "__main__":
    questions = generate_synthetic_questions_with_llm(model, tokenizer)
    save_questions_to_csv(questions, "synthetic_questions_llm_nvc_all_cases.csv")
    print("Comprehensive question generation process with all cases completed.")
