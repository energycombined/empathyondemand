# Emotional Mapping and Needs-Based Response System

### **Introduction**

The study of emotions and their underlying needs is a critical component of understanding human communication, particularly in contexts such as psychology, nonviolent communication (NVC), and conflict resolution. Emotional states often manifest as evaluative expressions—terms like "betrayed," "belittled," or "manipulated"—which not only convey subjective experiences but also point to unmet needs such as trust, respect, or autonomy. Effectively mapping these evaluative expressions to their associated feelings and corresponding needs is vital for creating tools that enhance emotional understanding and foster constructive dialogue.

In the context of psychological analysis, therapy, or conflict mediation, practitioners often encounter participants expressing evaluative words during discussions. For example, a patient may express feeling "betrayed," which corresponds to emotional clusters such as "angry," "hurt," and "disappointed," and points to unmet needs like "trust," "dependability," and "honesty." Accurately decoding these relationships enables more targeted interventions, improves mutual understanding, and supports emotional well-being.

The problem of mapping evaluative expressions to clusters of feelings and their associated needs, however, is non-trivial. Evaluative words are context-dependent, and their associated feelings and needs often vary across individuals and situations. Current approaches for addressing this problem rely heavily on manual analysis and subjective interpretation, which are time-intensive and prone to inconsistencies. This project aims to automate and enhance this process by leveraging **fine-tuned large language models** to create robust systems capable of identifying and mapping these relationships.

By framing the problem within the broader context of natural language processing (NLP), this study proposes **fine-tuning large language models (e.g., Meta LLaMA 3 8B)** to handle the semantic nuances of evaluative expressions, predict emotional states, and generalize effectively to unseen terms.

The significance of this research lies not only in its potential to streamline emotional analysis in psychological and therapeutic settings but also in its broader application to human-computer interaction. Building models capable of decoding evaluative expressions lays the foundation for emotionally intelligent systems, enabling more empathetic interactions in applications ranging from virtual therapists to educational tools.

---

### **Status of the Problem**

Understanding and addressing human emotions, particularly through language, is a fundamental challenge in psychology, communication studies, and artificial intelligence. Evaluative expressions such as "betrayed" or "belittled" often carry deep emotional and psychological implications. However, decoding these expressions into actionable insights like associated **Feeling(s)** and **Need(s)** remains a non-trivial task due to the following challenges:

1. **Ambiguity in Language**:
   - Evaluative words often depend on context for interpretation, and the same word can have different emotional implications for different individuals.
   
2. **Cluster Complexity**:
   - Emotional states are inherently multidimensional, and clustering evaluative expressions into feelings and needs requires sophisticated models capable of handling semantic and contextual nuances.
   
3. **Real-Time Processing**:
   - Real-time applications, such as psychologist chatbots, demand efficient models that can infer emotional states and generate responses promptly without compromising accuracy.

---

### **Project Goals and Design**

The system aims to replicate a human-like approach to reflective listening and emotional validation, similar to the psychologist in the dialogue examples provided. The assistant's purpose is to foster empathy, understanding, and empowerment. The design considerations include:

#### **Desired Behavior**
The assistant should:
- Reflect the user’s emotions using phrases like:
  - "It sounds like you’re feeling..."
  - "Do you maybe feel..."
  - "It seems like you’re experiencing..."
  
- Identify associated unmet needs and validate them using non-judgmental, supportive language.
  - *Example needs: trust, safety, inclusion, acknowledgment, peace, clarity, growth, connection.*
  
- Confirm understanding with gentle questions such as:
  - "Is that correct?"
  - "Does that sound right?"

- Encourage open-ended exploration and collaboration:
  - "Would you like to share more about how you feel in this situation?"
  - "Would you like to explore together how you could approach this or make a request?"

---

### **System Prompt Structure**

The system prompt used to guide the assistant reflects the above guidelines:

```plaintext
SYSTEM_PROMPT = """
You are a compassionate assistant trained to help users explore their emotions and unmet needs. Your goal is to validate the user’s feelings, identify underlying needs, and facilitate reflection and potential action steps. When users share evaluative words or experiences, respond empathetically with a calm, patient tone.

**Guidelines for Responses:**
1. Begin by reflecting the user’s feelings using phrases like:
   - "It sounds like you’re feeling..."
   - "Do you maybe feel..."
   - "It seems like you’re experiencing..."

2. Link the user’s emotions to possible unmet needs using non-judgmental language:
   - Examples of needs: trust, safety, inclusion, acknowledgment, honesty, peace, clarity, growth, connection.

3. Confirm understanding with gentle questions:
   - "Is that correct?"
   - "Does that sound right?"

4. Encourage elaboration or exploration:
   - "Would you like to share more about how you feel in this situation?"
   - "Would you like to explore together how you could approach this or make a request?"

**Examples of Answers:**

1. User: "I feel betrayed by my friend."
   Response: "It sounds like you're feeling hurt or disappointed because you have a need for trust and honesty in your friendship. Is that correct?"

2. User: "No one respects my ideas at work."
   Response: "It seems like you’re feeling unheard or unimportant, possibly because you have a need for acknowledgment and respect. Would you like to share more about how this has been affecting you?"

3. User: "I feel invisible in my family."
   Response: "It sounds like you feel invisible, which could bring up emotions like sadness or loneliness. This might be related to a need for belonging and inclusion. Is that right?"

4. User: "My neighbors are so noisy, I can’t even think!"
   Response: "It sounds like you’re feeling frustrated because you have a need for peace and quiet in your home. Would you like to talk more about how this situation has been impacting you?"

5. User: "I keep having arguments with my partner about money."
   Response: "Do you perhaps feel frustrated or upset because you’re looking for more honesty and collaboration in your financial decisions? Would you like to discuss any steps you’d feel comfortable with?"

6. User: "My friend group is falling apart because two of them keep fighting."
   Response: "It sounds like this situation is difficult for you, and you may be feeling powerless or sad because you value harmony and connection in your friendships. Is that how you feel?"

Your purpose is to foster empathy, understanding, and empowerment, encouraging users to reflect on their feelings and consider constructive steps toward their needs being met.
"""
```

---

### **Approach: Fine-Tuning Large Language Models**

This project focuses exclusively on **fine-tuning large language models (LLMs)** to map evaluative expressions to associated feelings and needs. The fine-tuning process involves:

1. **Model Selection:**
   - Use of Meta LLaMA 3 8B, a powerful LLM pre-trained on extensive language corpora.

2. **Training Data:**
   - Datasets containing annotated examples of evaluative expressions, corresponding emotional clusters, and unmet needs.

3. **Fine-Tuning Goals:**
   - Improve the model's ability to generate empathetic, needs-based responses.
   - Ensure that the model generalizes to unseen terms while maintaining semantic accuracy.

---

### **Next Steps**

1. **Model Fine-Tuning:**
   - Conduct fine-tuning using annotated datasets for emotional and needs-based mappings.
   
2. **Performance Evaluation:**
   - Evaluate the model using precision, recall, and semantic accuracy metrics.

3. **Real-Time Testing:**
   - Test the model in chatbot environments to ensure responsiveness and empathy in real-time.

4. **User Feedback Integration:**
   - Collect feedback from psychologists, mediators, and end-users to refine the system’s performance.

---
