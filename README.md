# 🌍 PerTrans – Context-Aware Multilingual Chat Translation & Evaluation

<div align="center">

![PerTrans](https://img.shields.io/badge/Project-PerTrans-00C853?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-FastAPI-blue?style=for-the-badge)
![Frontend](https://img.shields.io/badge/Frontend-HTML%20CSS%20JavaScript-orange?style=for-the-badge)
![LLM](https://img.shields.io/badge/OpenRouter-Multi--Model-purple?style=for-the-badge)

**A Context-Aware Multilingual Translation System with Multi-Model Evaluation**

</div>

---

# 📖 Overview

PerTrans is a research prototype that performs **context-aware multilingual translation** for chat conversations using multiple Large Language Models (LLMs).

Unlike traditional sentence-level translation systems, PerTrans considers the surrounding conversation to preserve:

- Context
- Tone
- Meaning
- Conversation flow

The system also includes a comprehensive evaluation dashboard that compares translation quality and efficiency across multiple LLMs.

---

# ✨ Features

## 🌐 Context-Aware Translation

- Translate chat conversations instead of isolated sentences
- Adjustable context window
- Multiple target languages
- Preserves conversational meaning

---

## 🤖 Multi-Model Support

Compare translations using:

- Meta Llama 3.3 70B
- DeepSeek Chat
- Qwen 2.5 72B

Users can switch models directly from the interface.

---

## 📊 Evaluation Dashboard

PerTrans automatically evaluates generated translations using standard NLP metrics.

Supported metrics include:

- BLEU
- COMET
- ROUGE-L
- BERTScore

Efficiency metrics:

- Latency
- GPU Memory
- Tokens Used
- Cost
- Overall Score

---

## 📈 Interactive Visualizations

The evaluation dashboard contains:

- Radar Chart
- Token Distribution Chart
- Overall Performance Score
- Translation Output Viewer
- Detailed Metric Table
- Strengths & Weaknesses Analysis
- Automated Feedback

---

# 🖥️ System Interface

## Translation Interface

Features:

- Context Window Slider
- Target Language Selection
- Model Selection
- Translation
- Summarization
- Live Performance Metrics

---

## Evaluation Dashboard

Displays:

- Overall Score
- BLEU
- COMET
- ROUGE-L
- BERTScore
- GPU Memory Usage
- Tokens Used
- Latency
- Translation Quality Analysis

---

# 🏗️ Project Architecture

```
                    User
                      │
                      ▼
          HTML / CSS / JavaScript
                      │
                      ▼
              FastAPI Backend
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
   OpenRouter API             Evaluation Engine
        │                           │
        ▼                           ▼
 Multiple LLM Models        NLP Quality Metrics
        │                           │
        └─────────────┬─────────────┘
                      ▼
             Evaluation Dashboard
```

---

# ⚙️ Technologies Used

## Frontend

- HTML5
- CSS3
- JavaScript

## Backend

- Python
- FastAPI
- Uvicorn

## APIs

- OpenRouter API

## Supported Models

- Meta Llama 3.3 70B
- DeepSeek Chat
- Qwen 2.5 72B

## NLP Evaluation

- BLEU
- COMET
- ROUGE-L
- BERTScore

## Visualization

- Chart.js

---

# 📂 Project Structure

```
PerTrans/
│
├── frontend/
│   ├── index.html
│   ├── evaluate.html
│   ├── style.css
│   ├── script.js
│   ├── evaluate.js
│   └── assets/
│
├── backend/
│   ├── app.py
│   ├── evaluate.py
│   ├── models.py
│   ├── metrics.py
│   └── requirements.txt
│
├── dataset/
│   └── messages.json
│
├── screenshots/
│
└── README.md
```

---

# 📊 Evaluation Metrics

| Metric | Purpose |
|---------|----------|
| BLEU | Measures n-gram overlap with reference translation |
| COMET | Neural metric evaluating semantic quality |
| ROUGE-L | Longest Common Subsequence similarity |
| BERTScore | Contextual semantic similarity |
| Latency | Translation response time |
| GPU Memory | Runtime memory usage |
| Tokens Used | Prompt + Completion tokens |
| Cost | API inference cost |

---

# 🚀 Getting Started

## Clone Repository

```bash
git clone https://github.com/yourusername/PerTrans.git

cd PerTrans
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure API

Create a `.env` file.

```
OPENROUTER_API_KEY=your_api_key_here
```

---

## Start Backend

```bash
uvicorn app:app --reload
```

---

## Launch Frontend

Simply open

```
index.html
```

or run using VSCode Live Server.

---

# 🔄 Workflow

```
User selects messages
          │
          ▼
Choose Context Window
          │
          ▼
Select Language
          │
          ▼
Select Model
          │
          ▼
Generate Translation
          │
          ▼
Run Evaluation
          │
          ▼
Compare Metrics
          │
          ▼
View Dashboard
```

---

# 📈 Model Comparison

PerTrans enables side-by-side comparison of different LLMs using:

- Translation Quality
- Context Preservation
- Semantic Similarity
- Computational Cost
- Response Time
- Token Consumption
- Translation Consistency

---

# 🔬 Research Contributions

This project demonstrates:

- Context-aware multilingual translation
- Comparative evaluation of multiple LLMs
- Integrated translation quality assessment
- Interactive visualization dashboard
- Cost-performance analysis for translation models

---

# 📷 Screenshots

## Translation Interface

> Add screenshot here

```
screenshots/interface.png
```

---

## Evaluation Dashboard

> Add screenshot here

```
screenshots/dashboard.png
```

---

## Translation Results

> Add screenshot here

```
screenshots/output.png
```

---

# 📌 Future Improvements

- Voice Translation
- Speech-to-Speech Translation
- Emotion-aware Translation
- Translation Memory
- Custom Fine-tuned Models
- Real-time Chat Integration
- Additional Evaluation Metrics
- PDF Report Export
- Model Leaderboard

---

# 👨‍💻 Authors

**Srutilaya Rajaraman**

B.Tech Computer Science & Engineering

VIT Chennai

---

# 📜 License

This project is developed for academic and research purposes.

---

<div align="center">

### ⭐ If you like this project, consider giving it a Star!

</div>
