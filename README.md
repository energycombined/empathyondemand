# Emotional Mapping and Needs-Based Response System

## Introduction

The study of emotions and their underlying needs is a critical component of understanding human communication, particularly in contexts such as psychology, nonviolent communication (NVC), and conflict resolution. Emotional states often manifest as evaluative expressions—terms like "betrayed," "belittled," or "manipulated"—which not only convey subjective experiences but also point to unmet needs such as trust, respect, or autonomy. Effectively mapping these evaluative expressions to their associated feelings and corresponding needs is vital for creating tools that enhance emotional understanding and foster constructive dialogue.

In the context of psychological analysis, therapy, or conflict mediation, practitioners often encounter participants expressing evaluative words during discussions. For example, a patient may express feeling "betrayed," which corresponds to emotional clusters such as "angry," "hurt," and "disappointed," and points to unmet needs like "trust," "dependability," and "honesty." Accurately decoding these relationships enables more targeted interventions, improves mutual understanding, and supports emotional well-being.

The problem of mapping evaluative expressions to clusters of feelings and their associated needs, however, is non-trivial. Evaluative words are context-dependent, and their associated feelings and needs often vary across individuals and situations. Current approaches for addressing this problem rely heavily on manual analysis and subjective interpretation, which are time-intensive and prone to inconsistencies. This project aims to automate and enhance this process by leveraging **fine-tuned large language models** to create robust systems capable of identifying and mapping these relationships.

By framing the problem within the broader context of natural language processing (NLP), this study proposes **fine-tuning large language models (e.g., Meta LLaMA 3 8B)** to handle the semantic nuances of evaluative expressions, predict emotional states, and generalize effectively to unseen terms.

The significance of this research lies not only in its potential to streamline emotional analysis in psychological and therapeutic settings but also in its broader application to human-computer interaction. Building models capable of decoding evaluative expressions lays the foundation for emotionally intelligent systems, enabling more empathetic interactions in applications ranging from virtual therapists to educational tools.

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

## Applications

*   **Psychology & Therapy:** Assisting therapists in understanding and responding to patients.
*   **Conflict Resolution:** Facilitating communication and empathy between parties in conflict.
*   **Human-Computer Interaction:** Creating emotionally intelligent virtual assistants and chatbots.

## Technical Approach

We are fine-tuning **Meta LLaMA 3 8B**, a powerful large language model (LLM), to achieve our goals. Fine-tuning involves training the model on a specialized dataset to improve its performance on a specific task.

**Key Techniques:**

*   **4-bit Quantization:**  Reduces memory usage for efficient operation.
*   **LoRA (Low-Rank Adaptation):**  Allows efficient fine-tuning by modifying only a small portion of the model's parameters.
*   **Hugging Face Transformers Library:** Used for model training and management.

## Current Status

We have successfully fine-tuned the Meta LLaMA 3 8B model and developed a user-friendly demonstration using Gradio. The model can now engage in empathetic conversations, reflecting user emotions and suggesting underlying needs.

## Future Steps

*   Continue refining the model through further fine-tuning and expanded datasets.
*   Conduct more comprehensive evaluations to assess performance and identify areas for improvement.
*   Integrate the model into real-world applications.
*   Gather user feedback to improve the system's effectiveness.
*   The fine-tuned model is available on the **Hugging Face Hub** for wider access and testing.

## How to Use

The project includes a Gradio demo that allows you to interact with the AI chatbot. Simply type in your feelings or experiences, and the chatbot will respond empathetically, helping you explore your emotions and underlying needs.

## Conclusion

This project demonstrates the potential of fine-tuned large language models to create emotionally intelligent systems that can understand and respond to human emotions in a meaningful way. This has significant implications for various fields, paving the way for more empathetic and effective human-computer interactions.