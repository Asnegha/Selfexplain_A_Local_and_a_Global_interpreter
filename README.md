# Selfexplain: A Local and a Global interpreter
- This project enhances the **interpretability** of transformer-based text classifiers by using **phrase-level concepts**, offering both **local (per-sample)** and **global (dataset-level)** insights into model decisions.
- Report can be found at: **Self_Explain.pdf**
---

## Models & Methods

- **Transformer Models Evaluated:**  
  - BERT  
  - RoBERTa  
  - XLNet  
  - XLM-R  

- **Tuning Strategies:**  
  - Full fine-tuning  
  - LoRa (Low-Rank Adaptation)  
  - Quantization-aware tuning  

- **Model Sizes:**  
  - small
  - medium
  - large

---

## 📊 Datasets

- **SST** – Sentiment classification  
- **TREC** – Question classification  
- **Banking77** – Customer support intent classification  
- **GoEmotions** – Fine-grained emotion classification (27 emotion labels + neutral)

---

## 🚀 Key Results

- Achieved **97.5% accuracy on TREC** using **XLM-R + LoRa**, with **top-5 phrase-level explanations**
- Phrase-level rationales successfully highlight key decision-driving tokens
- Framework supports both **local explanations** (per instance) and **global phrase trends** (dataset-wide)
