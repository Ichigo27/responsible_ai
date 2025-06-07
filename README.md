# Responsible AI Evaluation API Documentation

## Overview

The Responsible AI Evaluation API provides endpoints for evaluating AI-generated responses across multiple responsible AI metrics including bias/fairness, hallucination detection, toxicity, relevance, and explainability.

## Base URL

```
http://localhost:9500/api/v1
```

## Available Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| `bias_fairness` | Evaluates responses for demographic bias, stereotyping, and unfair treatment | 0.7 |
| `hallucination` | Detects factual inaccuracies and unsupported claims | 0.8 |
| `toxicity` | Identifies harmful, offensive, or inappropriate content | 0.9 |
| `relevance` | Measures how well the response addresses the given prompt | 0.7 |
| `explainability` | Provides detailed explanation of the AI's reasoning process | N/A |

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /api/v1/health`

**Description:** Check the health status of the API service.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "service": "responsible_ai",
  "python_version": "3.13"
}
```

### 2. Single Evaluation

**Endpoint:** `POST /api/v1/evaluate`

**Description:** Evaluate a single prompt-response pair against all or selected metrics.

**Request Headers:**

- `Content-Type: application/json`

**Request Body:**

```json
{
  "prompt": "string (required) - The original user query/prompt",
  "response": "string (required) - The AI model's response to evaluate",
  "id": "string (optional) - Unique identifier for tracking",
  "metadata": "object (optional) - Additional metadata",
  "system_instructions": "string (optional) - System-level instructions given to the AI",
  "conversation_history": "string (optional) - Previous conversation context",
  "retrieved_contexts": "string (optional) - Retrieved documents/context (e.g., from RAG)",
  "metrics": ["array of metric names (optional) - Specific metrics to evaluate"]
}
```

**Response:**

```json
{
  "prompt": "What is the capital of France?",
  "response": "The capital of France is Paris.",
  "metrics": {
    "bias_fairness": {
      "score": 1.0,
      "reason": "The response is factual and contains no bias.",
      "threshold": 0.7,
      "passed": true,
      "additional_data": {
        "bias_categories": []
      }
    },
    "hallucination": {
      "score": 1.0,
      "reason": "The response is factually accurate.",
      "threshold": 0.8,
      "passed": true,
      "additional_data": {
        "hallucinations": []
      }
    },
    "toxicity": {
      "score": 1.0,
      "reason": "No toxic content detected.",
      "threshold": 0.9,
      "passed": true,
      "additional_data": {
        "toxic_categories": []
      }
    },
    "relevance": {
      "score": 1.0,
      "reason": "The response directly answers the question.",
      "threshold": 0.7,
      "passed": true,
      "additional_data": {
        "irrelevant_sections": []
      }
    }
  },
  "evaluation_id": "123e4567-e89b-12d3-a456-426614174000",
  "llm_usage": {
    "input_tokens": 450,
    "output_tokens": 200,
    "total_tokens": 650,
    "cost": 0.0045,
    "request_count": 4
  }
}
```

### 3. Batch Evaluation

**Endpoint:** `POST /api/v1/batch-evaluate`

**Description:** Evaluate multiple prompt-response pairs in a single request.

**Request Headers:**

- `Content-Type: application/json` or `Content-Type: application/jsonl`

**Query Parameters:**

- `metrics` (optional): Comma-separated list of specific metrics to evaluate

**Request Body (JSON Array):**

```json
[
  {
    "prompt": "What is AI?",
    "response": "AI stands for Artificial Intelligence...",
    "id": "item-1"
  },
  {
    "prompt": "Explain machine learning",
    "response": "Machine learning is a subset of AI...",
    "id": "item-2",
    "system_instructions": "Explain in simple terms"
  }
]
```

**Request Body (JSONL):**

```jsonl
{"prompt": "What is AI?", "response": "AI stands for Artificial Intelligence...", "id": "item-1"}
{"prompt": "Explain machine learning", "response": "Machine learning is a subset of AI...", "id": "item-2"}
```

**Response:** JSONL formatted results

```jsonl
{"prompt": "What is AI?", "response": "AI stands for...", "id": "item-1", "metrics": {...}}
{"prompt": "Explain machine learning", "response": "Machine learning...", "id": "item-2", "metrics": {...}}
```

### 4. Specific Metric Evaluation

**Endpoint:** `POST /api/v1/metrics/{metric_name}`

**Description:** Evaluate using a specific metric only.

**Path Parameters:**

- `metric_name`: One of `bias_fairness`, `hallucination`, `toxicity`, `relevance`, `explainability`

**Request Body:** Same as single evaluation endpoint

**Response:**

```json
{
  "score": 0.95,
  "reason": "The response is mostly relevant with minor tangential information.",
  "threshold": 0.7,
  "passed": true,
  "additional_data": {
    "irrelevant_sections": ["Brief mention of unrelated topic"]
  },
  "id": "optional-id-if-provided"
}
```

### 5. Dashboard Dataset Management

#### Get Dataset Info

**Endpoint:** `GET /api/v1/dashboard/dataset`

**Description:** Get information about the current dashboard dataset.

**Response:**

```json
{
  "status": "available",
  "record_count": 150,
  "file_size_bytes": 125000,
  "file_size_mb": 0.12,
  "file_path": "/app/data/dashboard/dashboard_data.jsonl",
  "last_modified": "2024-01-15 10:30:00"
}
```

#### Replace Dataset

**Endpoint:** `POST /api/v1/dashboard/dataset/replace`

**Description:** Replace the entire dashboard dataset with new data.

**Request Headers:**

- `Content-Type: application/json` or `Content-Type: application/jsonl`

**Request Body:** Array of evaluation results or JSONL data

**Response:**

```json
{
  "message": "Successfully replaced dashboard dataset with 200 records",
  "record_count": 200,
  "file_path": "/app/data/dashboard/dashboard_data.jsonl"
}
```

#### Append to Dataset

**Endpoint:** `POST /api/v1/dashboard/dataset/append`

**Description:** Append new data to the existing dashboard dataset.

**Request Headers:**

- `Content-Type: application/json` or `Content-Type: application/jsonl`

**Request Body:** Array of evaluation results or JSONL data

**Response:**

```json
{
  "message": "Successfully appended 50 records to dashboard dataset",
  "records_appended": 50,
  "file_path": "/app/data/dashboard/dashboard_data.jsonl"
}
```

## Optional Context Fields

The API supports additional context fields that help provide more accurate evaluations:

### System Instructions

Provide the system-level instructions or persona that was given to the AI model:

```json
{
  "system_instructions": "You are a helpful medical assistant. Always provide accurate medical information and recommend consulting healthcare professionals."
}
```

### Conversation History

Include previous conversation turns for context-aware evaluation:

```json
{
  "conversation_history": "User: What are the symptoms of flu?\nAssistant: Common flu symptoms include fever, cough, and body aches.\nUser: How long does it typically last?"
}
```

### Retrieved Contexts

For RAG (Retrieval-Augmented Generation) systems, include the retrieved documents:

```json
{
  "retrieved_contexts": "Document 1: The flu typically lasts 5-7 days...\nDocument 2: Most people recover from flu within a week..."
}
```

## Error Responses

### 400 Bad Request

```json
{
  "error": "Invalid request data: 'prompt' is a required property"
}
```

### 404 Not Found

```json
{
  "error": "Metric not found: invalid_metric"
}
```

### 500 Internal Server Error

```json
{
  "error": "Internal server error"
}
```

## Example Usage

### Basic Evaluation

```bash
curl -X POST http://localhost:9500/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "response": "The capital of France is Paris."
  }'
```

### Evaluation with Specific Metrics

```bash
curl -X POST http://localhost:9500/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me about climate change",
    "response": "Climate change refers to long-term shifts in temperatures...",
    "metrics": ["bias_fairness", "hallucination"]
  }'
```

### Evaluation with Full Context

```bash
curl -X POST http://localhost:9500/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Based on the document, what is the main cause?",
    "response": "According to the document, the main cause is human activity.",
    "system_instructions": "You are an environmental science expert.",
    "conversation_history": "User: What document are we discussing?\nAssistant: We are discussing the IPCC climate report.",
    "retrieved_contexts": "Document: The IPCC report states that human activities are the dominant cause of observed warming since the mid-20th century.",
    "id": "eval-001",
    "metadata": {
      "source": "climate_qa_system",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  }'
```

### Batch Evaluation with Metrics Filter

```bash
curl -X POST "http://localhost:9500/api/v1/batch-evaluate?metrics=toxicity,relevance" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "prompt": "What is Python?",
      "response": "Python is a high-level programming language.",
      "id": "batch-1"
    },
    {
      "prompt": "Explain quantum computing",
      "response": "Quantum computing uses quantum mechanical phenomena...",
      "id": "batch-2"
    }
  ]'
```

### Single Metric Evaluation

```bash
curl -X POST http://localhost:9500/api/v1/metrics/hallucination \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What year did World War II end?",
    "response": "World War II ended in 1945."
  }'
```

### JSONL Batch Evaluation

```bash
curl -X POST http://localhost:9500/api/v1/batch-evaluate \
  -H "Content-Type: application/jsonl" \
  -d '{"prompt": "What is AI?", "response": "AI is..."}
{"prompt": "What is ML?", "response": "ML is..."}'
```

### Dashboard Dataset Management

```bash
# Get dataset info
curl -X GET http://localhost:9500/api/v1/dashboard/dataset

# Replace dataset
curl -X POST http://localhost:9500/api/v1/dashboard/dataset/replace \
  -H "Content-Type: application/json" \
  -d '[
    {
      "prompt": "Test prompt",
      "response": "Test response",
      "metrics": {...}
    }
  ]'

# Append to dataset
curl -X POST http://localhost:9500/api/v1/dashboard/dataset/append \
  -H "Content-Type: application/jsonl" \
  -d '{"prompt": "New prompt", "response": "New response", "metrics": {...}}'
```

## Rate Limits and Concurrency

- Maximum concurrent LLM requests: 50 (configurable)
- Request timeout: 600 seconds
- Batch processing: Up to 10 concurrent workers
- Metric evaluation: Up to 5 concurrent workers per request

## Dashboard Access

The Streamlit dashboard is available at:

```
http://localhost:9501
```

The dashboard provides:

- Visual analytics of evaluation metrics
- Temporal analysis and trends
- Failure mode analysis
- Detailed explainability views
- Pareto analysis for identifying critical issues