# Program Outline: Empathetic Framework for Open-Source Language Models  
**Collaborating Institution**: Tilburg University  
**Objective**: Develop a scalable, open-source framework based on Nonviolent Communication (NVC) to transform evaluative language into empathetic communication using fine-tuned DeepSeek models.  

---

## Research Stages & Costs  

### Stage 1: Foundation & Data Preparation (Months 1-3)  
**Description**:  
- Define NVC-based empathy metrics and annotate a **1,000-sample dataset** with emotions, needs, and non-evaluative rephrasing.  
- Collaborate with Tilburg University to map evaluative language (e.g., “I feel betrayed”) to underlying feelings/needs.  

**Cost Breakdown**:  
| Category              | Cost (€) | Details                                                                 |  
|-----------------------|----------|-------------------------------------------------------------------------|  
| Data Annotation        | 4,000    | Expert linguists + NVC practitioners to label/curate dataset.          |  
| Cloud Compute (AWS)    | 1,500    | Initial experiments with DeepSeek 7B (4-bit QLoRA).                    |  
| Personnel              | 3,500    | Student researchers for pipeline setup and baseline testing.           |  

---

### Stage 2: Model Development (Months 4-6)  
**Description**:  
- Fine-tune **DeepSeek 13B–33B** using 4-bit QLoRA and DeepSpeed.  
- Train models to recognize emotions and propose NVC-aligned rephrasing (e.g., “I feel hurt” → “I need trust”).  

**Cost Breakdown**:  
| Category              | Cost (€) | Details                                                                 |  
|-----------------------|----------|-------------------------------------------------------------------------|  
| Cloud Compute (AWS)    | 12,000   | Spot instances for distributed training (8x A100 GPUs).                |  
| Data Augmentation      | 2,000    | Synthetic data generation to improve generalization.                   |  
| Personnel              | 6,000    | Researchers for hyperparameter tuning and evaluation.                  |  

---

### Stage 3: Deployment & Validation (Months 7-9)  
**Description**:  
- Integrate models into real-world applications (e.g., chatbots, social media moderation tools).  
- Conduct user studies to measure conflict reduction and empathy improvement.  

**Cost Breakdown**:  
| Category              | Cost (€) | Details                                                                 |  
|-----------------------|----------|-------------------------------------------------------------------------|  
| API Integration        | 5,000    | Deploy production-ready model with Docker + FastAPI.                   |  
| User Testing           | 3,500    | Participant compensation and analysis.                                 |  
| Contingency            | 2,000    | Unforeseen scaling/optimization costs.                                 |  

---

### Stage 4: Dissemination (Months 10-12)  
**Description**:  
- Publish results in peer-reviewed journals (e.g., ACL, EMNLP).  
- Release open-source training pipeline and public dataset.  

**Cost Breakdown**:  
| Category              | Cost (€) | Details                                                                 |  
|-----------------------|----------|-------------------------------------------------------------------------|  
| Academic Publishing    | 1,500    | Article processing charges (APCs) for open-access journals.            |  
| Documentation          | 1,000    | Technical writing + open-source repository maintenance.                |  

---

## Total Budget Tiers  
| Tier  | Scope                          | Model               | Total Cost (€) | Key Deliverables                                      |  
|-------|--------------------------------|---------------------|----------------|------------------------------------------------------|  
| **1** | Pilot (Validation)             | DeepSeek 7B         | 20,000         | Proof-of-concept model + benchmarking report.        |  
| **2** | Intermediate (Deployment)      | DeepSeek 13B–33B    | 40,000         | Deployable model + open-source pipeline + user study.|  
| **3** | Advanced (Research Impact)     | DeepSeek 33B–67B    | 50,000         | Peer-reviewed paper + public dataset + API.          |  

---

## Cost-Saving Strategies  
1. **QLoRA + 4-bit Quantization**: Reduces training costs by 80% while maintaining performance.  
2. **AWS Spot Instances**: Lowers cloud expenses by 70% for GPU workloads.  
3. **Open-Source Tools**: Hugging Face `peft`, `transformers`, and `llama.cpp` eliminate licensing fees.  

---

## Alignment with University Goals  
- **Research Excellence**: Advances AI for social good, aligning with Tilburg’s focus on ethics and technology.  
- **Student Training**: Provides hands-on experience in NLP, empathy modeling, and scalable AI.  
- **Societal Impact**: Reduces digital polarization through empathetic communication tools.  

---