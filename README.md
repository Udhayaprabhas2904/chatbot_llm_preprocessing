# LLM Preprocessing Comparison System

## Overview

The LLM Preprocessing Comparison System is a Python-based application developed using FastAPI to evaluate the impact of text preprocessing techniques on Large Language Model (LLM) performance.

The system compares baseline and preprocessed queries across five locally hosted LLMs using Ollama, measuring performance metrics and generating comprehensive PDF reports for analysis.

## Supported LLM Models

- SmolLM2
- Qwen 2.5
- TinyLlama
- Llama 3.2
- Gemma 3

## Key Features

- **Text Preprocessing:** Applies multiple text cleaning and normalization techniques.
- **Baseline vs. Preprocessed Evaluation:** Compares model responses using original and preprocessed queries.
- **Performance Monitoring:** Measures preprocessing time, generation time, and end-to-end latency.
- **Experiment Management:** Executes and tracks experiments across multiple models and queries.
- **Data Persistence:** Stores experiment results and evaluation metrics in SQLite.
- **Automated Reporting:** Generates PDF reports with performance summaries, comparison tables, and visual analytics.

## Performance Metrics

- Preprocessing Time
- Model Generation Time
- End-to-End Processing Time
- Prompt Length
- Response Length
- Word Count
- Sentence Count
- Lexical Diversity

## Technology Stack

| Technology | Application |
|---|---|
| Python | Core development |
| FastAPI | REST API development |
| Ollama | Local LLM integration |
| SQLite | Data storage |
| Matplotlib | Data visualization |
| ReportLab | PDF report generation |
| Swagger UI | API documentation and testing |

## System Workflow

1. Submit user queries through the API.
2. Process the original query as the baseline.
3. Apply selected preprocessing techniques.
4. Submit the preprocessed query to the selected LLM.
5. Collect and compare performance metrics.
6. Store experiment results in the SQLite database.
7. Generate PDF reports with comparison charts and performance summaries.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd llm-preprocessing-comparison-system
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

Windows:

```powershell
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Set Up Ollama

Install Ollama and download the required models before running experiments.

### 6. Start the Application

```bash
uvicorn app.main:app --reload
```

## API Documentation

Once the application is running, access the interactive Swagger documentation at:

`http://127.0.0.1:8000/docs`

## Project Objective

To systematically evaluate how text preprocessing techniques influence the performance and response characteristics of locally hosted Large Language Models through comparative experimentation and quantitative analysis.

## Future Enhancements

- Automated response quality evaluation
- Accuracy, relevance, and completeness scoring
- Repeated experiments for statistical reliability
- Advanced performance analytics
- Model-wise comparative evaluation

