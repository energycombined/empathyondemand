# Emotional Mapping and Needs-Based Response System

## Introduction

The study of emotions and their underlying needs is a critical component of understanding human communication, particularly in contexts such as psychology, nonviolent communication (NVC), and conflict resolution. Emotional states often manifest as evaluative expressions—terms like "betrayed," "belittled," or "manipulated"—which not only convey subjective experiences but also point to unmet needs such as trust, respect, or autonomy. Effectively mapping these evaluative expressions to their associated feelings and corresponding needs is vital for creating tools that enhance emotional understanding and foster constructive dialogue.

In the context of psychological analysis, therapy, or conflict mediation, practitioners often encounter participants expressing evaluative words during discussions. For example, a patient may express feeling "betrayed," which corresponds to emotional clusters such as "angry," "hurt," and "disappointed," and points to unmet needs like "trust," "dependability," and "honesty." Accurately decoding these relationships enables more targeted interventions, improves mutual understanding, and supports emotional well-being.

The problem of mapping evaluative expressions to clusters of feelings and their associated needs, however, is non-trivial. Evaluative words are context-dependent, and their associated feelings and needs often vary across individuals and situations. Current approaches for addressing this problem rely heavily on manual analysis and subjective interpretation, which are time-intensive and prone to inconsistencies. This project aims to automate and enhance this process by leveraging **fine-tuned large language models** to create robust systems capable of identifying and mapping these relationships.

By framing the problem within the broader context of natural language processing (NLP), this study proposes **fine-tuning large language models (e.g., Meta LLaMA 3 8B, DeepSeek 7B/13B/33B)** to handle the semantic nuances of evaluative expressions, predict emotional states, and generalize effectively to unseen terms.

The significance of this research lies not only in its potential to streamline emotional analysis in psychological and therapeutic settings but also in its broader application to human-computer interaction. Building models capable of decoding evaluative expressions lays the foundation for emotionally intelligent systems, enabling more empathetic interactions in applications ranging from virtual therapists to educational tools.

## Collaborating Institution: Tilburg University

This project is a collaborative effort with Tilburg University, leveraging their expertise in psychology, ethics, and AI to enhance the depth and applicability of our research. Tilburg University's involvement ensures that the project is grounded in sound psychological principles and ethical considerations, aligning with their focus on responsible AI development.

## Status of the Problem

Understanding and addressing human emotions, particularly through language, is a fundamental challenge in psychology, communication studies, and artificial intelligence. Evaluative expressions such as "betrayed" or "belittled" often carry deep emotional and psychological implications. However, decoding these expressions into actionable insights like associated **Feeling(s)** and **Need(s)** remains a non-trivial task due to the following challenges:

1.  **Ambiguity in Language:** Evaluative words often depend on context for interpretation, and the same word can have different emotional implications for different individuals.
2.  **Cluster Complexity:** Emotional states are inherently multidimensional, and clustering evaluative expressions into feelings and needs requires sophisticated models capable of handling semantic and contextual nuances.
3.  **Real-Time Processing:** Real-time applications, such as psychologist chatbots, demand efficient models that can infer emotional states and generate responses promptly without compromising accuracy.

## Project Overview

This project focuses on building an AI system that can understand and respond to human emotions expressed through language, particularly "evaluative expressions" like "betrayed," "belittled," or "manipulated." These words often indicate unmet needs such as trust, respect, or autonomy. By accurately mapping these expressions to their associated feelings and needs, we aim to create a tool that enhances emotional understanding and fosters constructive dialogue.

## Goals

*   Develop an AI that can identify and reflect user emotions.
*   Link expressed emotions to underlying unmet needs.
*   Facilitate user reflection and exploration of solutions.
*   Reduce digital polarization through empathetic responses.
*   Create open-source tools that are accessible and ethically sound.

## Applications

*   **Psychology & Therapy:** Assisting therapists in understanding and responding to patients.
*   **Conflict Resolution:** Facilitating communication and empathy between parties in conflict.
*   **Human-Computer Interaction:** Creating emotionally intelligent virtual assistants and chatbots.
*   **Social Media Moderation:**  Reducing toxicity and promoting constructive online interactions.
*   **Education:** Teaching emotional intelligence and communication skills.

## Research Stages & Timeline

### Stage 1: Foundation & Data Preparation (Months 1-3)

**Objective:** Develop a linguistically grounded dataset for empathy modeling.

**Tasks:**

*   Define NVC-based empathy metrics with Tilburg University psychologists.
*   Annotate 1,000 samples linking evaluative terms to emotions and needs (e.g., "betrayed" → "hurt/angry" → "trust/honesty").
*   Create a diverse and representative dataset that covers a wide range of emotional expressions.

**Tools:**

*   Prodigy, Doccano for annotation.
*   Hugging Face datasets for data management.

### Stage 2: Model Development (Months 4-6)

**Objective:** Fine-tune Meta LLaMA 3 and DeepSeek models for empathy-aware text transformation.

**Tasks:**

*   Train Meta LLaMA 3 8B and DeepSeek 7B/13B/33B using 4-bit QLoRA + DeepSpeed.
*   Optimize for real-time inference on AWS spot instances.
*   Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.

**Tools:**

*   Hugging Face `peft`, `transformers` for model training.
*   AWS SageMaker for cloud-based training and deployment.
*   DeepSpeed for distributed training.

### Stage 3: Deployment & Validation (Months 7-9)

**Objective:** Deploy models and evaluate real-world impact.

**Tasks:**

*   Deploy API for chatbot and social media moderation applications.
*   Conduct user studies (N=200) to measure conflict reduction and user satisfaction.
*   Gather qualitative feedback to understand the user experience and identify areas for improvement.

**Tools:**

*   FastAPI, Docker for API deployment.
*   Qualtrics for user feedback collection.
*   Gradio for creating user-friendly interfaces.

### Stage 4: Dissemination (Months 10-12)

**Objective:** Share research findings and open-source tools.

**Tasks:**

*   Publish research findings in relevant conferences and journals (e.g., ACL, EMNLP).
*   Release open-source pipeline, dataset, and trained models on platforms like Hugging Face Hub and GitHub.
*   Host workshops at Tilburg University and other institutions to share knowledge and foster collaboration.

# Program Outline: Empathetic Framework for Open-Source Language Models  

 For more details [here](./program/README.md)


## Technical Approach

### Model Architecture

*   **Base Models:**
    *   Meta LLaMA 3 8B
    *   DeepSeek 7B/13B/33B (4-bit quantized)
*   **Fine-Tuning:**
    *   QLoRA: Train only 1-10% of parameters for efficiency.
    *   LoRA (Low-Rank Adaptation):  Allows efficient fine-tuning by modifying only a small portion of the model's parameters.
    *   DeepSpeed ZeRO-3: Distributed training across 8x A100 GPUs for faster training.
*   **Inference:**
    *   Optimized with vLLM for <500ms latency.

### Key Techniques

*   **4-bit Quantization:** Reduces memory usage for efficient operation and deployment.
*   **Hugging Face Transformers Library:** Used for model training, management, and deployment.
*   **Emotion-Need Mapping:**
    ```python
    # Example: Transform "I feel betrayed" → "I feel hurt; I need trust"
    output = model.generate("I feel betrayed", max_length=50)
    print(output)  # Output: "I feel hurt and angry. I need trust and honesty."
    ```
*   **Real-Time API:** FastAPI endpoint for seamless integration into various applications.

## Ethical Considerations

*   **Bias Mitigation:**  We will actively work to identify and mitigate biases in the training data and model outputs to ensure fairness and inclusivity.
*   **Data Privacy:**  User data will be handled with utmost care, following strict privacy guidelines and regulations.
*   **Transparency:** The project will be open-source, and our methods and findings will be transparently documented and shared.
*   **Responsible Use:** We will provide guidelines and recommendations for responsible use of the developed tools to prevent misuse or harm.

## Expected Outcomes

*   **Enhanced Emotional Understanding:** The system will provide more accurate and nuanced understanding of emotional expressions.
*   **Improved Communication:** The tools will facilitate more empathetic and constructive communication in various settings.
*   **Reduced Conflict:** By addressing underlying needs, the system will help reduce conflict and promote resolution.
*   **Open-Source Contribution:** The project will contribute valuable resources (datasets, models, tools) to the NLP and AI community.
*   **Advancement in Empathetic AI:** This research will push the boundaries of empathetic AI, leading to more human-centered technology.

## Current Status

We have successfully fine-tuned the Meta LLaMA 3 8B model and developed a user-friendly demonstration using Gradio. The model can now engage in empathetic conversations, reflecting user emotions and suggesting underlying needs. We are currently in the process of integrating and evaluating the DeepSeek models.

## Future Steps

*   Continue refining the models (Meta LLaMA 3 and DeepSeek) through further fine-tuning and expanded datasets.
*   Conduct more comprehensive evaluations to assess performance and identify areas for improvement, particularly focusing on the DeepSeek models' performance.
*   Integrate the models into real-world applications, including a pilot study with therapists or mediators.
*   Gather user feedback to improve the system's effectiveness and user experience.
*   The fine-tuned models will be made available on the **Hugging Face Hub** for wider access and testing.

## How to Use

The project includes a Gradio demo that allows you to interact with the AI chatbot. Simply type in your feelings or experiences, and the chatbot will respond empathetically, helping you explore your emotions and underlying needs.  A link to the demo will be provided here upon deployment.

## Conclusion

This project demonstrates the potential of fine-tuned large language models (Meta LLaMA 3 and DeepSeek) to create emotionally intelligent systems that can understand and respond to human emotions in a meaningful way. This has significant implications for various fields, paving the way for more empathetic and effective human-computer interactions and contributing to a more understanding and compassionate digital world. Our collaboration with Tilburg University ensures that this project is not only technically sound but also ethically grounded and socially responsible.
## License
This work is licensed under a 
Creative Commons Attribution 4.0 International License 
(http://creativecommons.org/licenses/by/4.0/).
