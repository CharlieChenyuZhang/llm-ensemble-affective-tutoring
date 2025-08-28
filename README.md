# LLM Ensemble Affective Tutoring

This repository contains the code and resources for the paper:

**"Ensembling Large Language Models to Characterize Affective Dynamics in Student–AI Tutor Dialogues"**

_Accepted at ACII 2025 Late Breaking_

# Pipeline Overview

## Data

For **FERPA compliance**, all original student text has been replaced with masked placeholders.

## 1. Labeling with LLMs

- `analyze_VAL_with_llms.py` generates labels and scores using **Claude** and **GPT-4o**.
- `analyze_VAL_with_gemini.py` generates labels and scores using **Gemini**.  
  _Note: The prompts used are embedded directly in these scripts._

## 2. Ensemble Fusion

- `fuse_ensemble.py` merges outputs from all models.
- Produces: **`pytutor_chat_messages_with_ensemble.json`**

## 3. Affect Analysis

- `pytutor_affect_analysis.py` performs affect analysis over the fused dataset.

## 4. Temporal Dynamics

- `temporal_affect_analysis.py` runs temporal analyses over the grouped dataset.

---

**Full Flow:**  
LLM labeling → ensemble fusion → affect analysis → temporal analysis

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{llm-ensemble-affective-tutoring-2025,
  title={Ensembling Large Language Models to Characterize Affective Dynamics in Student–AI Tutor Dialogues},
  author={Chenyu Zhang and Sharifa Alghowinem and Cynthia Breazeal},
  booktitle={Proceedings of the ACII 2025 Late Breaking},
  year={2025}
}
```
