# 🤖 AI/ML Engineering Internship Tasks

This repository contains completed tasks for the AI/ML Engineering Internship program.

---

## ✅ Completed Tasks

| # | Task | Tech Stack | Status |
|---|------|-----------|--------|
| 1 | News Topic Classifier using BERT | PyTorch, Hugging Face, Gradio | ✅ Done |
| 2 | End-to-End ML Pipeline (Customer Churn) | Scikit-learn, joblib | ✅ Done |
| 5 | Auto-Tagging Support Tickets using LLM | OpenAI API, Prompt Engineering | ✅ Done |

---

## 📁 Repository Structure

```
ai-ml-internship/
├── task1_bert_classifier/
│   ├── news_classifier.py     ← Main script
│   ├── requirements.txt
│   └── README.md
│
├── task2_ml_pipeline/
│   ├── pipeline.py            ← Main script
│   ├── requirements.txt
│   └── README.md
│
└── task5_auto_tagging/
    ├── auto_tagger.py         ← Main script
    ├── requirements.txt
    └── README.md
```

---

## 🚀 Quick Start

### Task 1 — BERT News Classifier
```bash
cd task1_bert_classifier
pip install -r requirements.txt
python news_classifier.py
```

### Task 2 — ML Pipeline (Customer Churn)
```bash
cd task2_ml_pipeline
pip install -r requirements.txt
python pipeline.py
```

### Task 5 — Auto-Tag Support Tickets
```bash
cd task5_auto_tagging
pip install -r requirements.txt

# Demo mode (no API key needed):
python auto_tagger.py

# With OpenAI:
export OPENAI_API_KEY="sk-your-key"
python auto_tagger.py
```

---

## 🛠️ Skills Demonstrated

- NLP with Transformers (BERT fine-tuning)
- Transfer learning & few-shot learning
- Scikit-learn Pipeline API
- Hyperparameter tuning with GridSearchCV
- Model export and reusability (joblib)
- Prompt engineering (zero-shot & few-shot)
- LLM-based text classification
- Production-readiness practices
