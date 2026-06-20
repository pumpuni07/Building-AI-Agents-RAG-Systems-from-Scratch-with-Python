# Building AI Agents from Scratch with Python
## IBM Skills Network — AI Engineering Labs

**Author:** Jack Pumpuni Frimpong-Manso  
**GitHub:** [@pumpuni07](https://github.com/pumpuni07)  
**LinkedIn:** [Jack Frimpong-Manso](https://www.linkedin.com/in/jack-pumpuni-frimpong-manso/)

---

## Repository Overview

This repository contains fully implemented, well-documented Python labs from the
IBM AI Engineering track, covering:

| File | Lab | Key Concepts |
|------|-----|--------------|
| `lab1_ai_agents_chatbot.py` | Building AI Agents from Scratch | Query routing, agent design, keyword matching, interactive loops |
| `lab2_rag_huggingface_gpt2.py` | RAG with Hugging Face | DPR retrieval, GPT-2 generation, parameter tuning |
| `lab3_rag_pytorch_song_safety.py` | RAG with PyTorch | Cosine vs dot-product similarity, embedding-based content classification |
| `lab4_langchain_prompt_engineering.py` | In-Context Engineering | Prompt templates, LangChain chains, few-shot prompting, role/tone control |

---

## Lab Descriptions

### Lab 1 — Building AI Agents from Scratch (`lab1_ai_agents_chatbot.py`)

A restaurant chatbot for **The Daily Dish** that routes user queries to
specialised agents:

- **QueryProcessor**: Lowercases, cleans punctuation, and expands synonyms
  (e.g. *"food"* → *"menu"*, *"temperature"* → *"weather"*).
- **Router**: Classifies each query as `"weather"` or `"restaurant"`.
- **WeatherAgent**: Fetches live weather from OpenWeatherMap (falls back to
  simulation if no API key is set).
- **DailyDishAgent**: Answers menu, FAQ, and greeting queries from a knowledge
  base.
- **Interactive loop**: Full terminal chat with exit conditions.

**Run:**
```bash
python lab1_ai_agents_chatbot.py          # Chat mode
python lab1_ai_agents_chatbot.py --test   # Run unit tests
```

**Optional env var for live weather:**
```
OPENWEATHERMAP_API_KEY=your_free_key_here
```

---

### Lab 2 — RAG with Hugging Face + GPT-2 (`lab2_rag_huggingface_gpt2.py`)

Demonstrates the full RAG pipeline using Hugging Face models:

1. **DPR** (Dense Passage Retrieval) encodes a passage corpus and user queries
   into dense vectors.
2. Retrieval is performed using **dot product** (default) or **cosine
   similarity** (exercise solution).
3. Retrieved contexts are prepended to a **GPT-2** prompt for generation.
4. **Parameter tuning exercise** compares three generation configurations
   (`max_length`, `min_length`, `length_penalty`, `num_beams`).

**Key observation:** RAG-augmented generation is significantly more accurate and
detailed than direct GPT-2 generation without context.

**Run:**
```bash
pip install transformers torch faiss-cpu
python lab2_rag_huggingface_gpt2.py
```

---

### Lab 3 — RAG with PyTorch: Song Safety (`lab3_rag_pytorch_song_safety.py`)

A content moderation system for a social media platform that checks whether
songs are appropriate for children:

1. A knowledge base of pre-answered content-appropriateness Q&A pairs is
   embedded using **sentence-transformers/all-MiniLM-L6-v2**.
2. New song snippets are embedded and matched to the most relevant Q&A pair.
3. **Exercise solution**: `RAG_QA()` uses **cosine similarity** instead of dot
   product — explained in full with mathematical derivation and justification.
4. Evaluation compares accuracy of both retrieval methods.

**Run:**
```bash
pip install torch sentence-transformers
python lab3_rag_pytorch_song_safety.py
```

---

### Lab 4 — In-Context Engineering with LangChain (`lab4_langchain_prompt_engineering.py`)

Controls LLM behaviour entirely through prompt structure:

- **Role + tone templates**: `"You are a {role}. Respond in a {tone} tone."`
- **Few-shot prompting**: Provides worked examples to enforce output format.
- **LangChain chains**: Combines `PromptTemplate | LLM | StrOutputParser`.
- **Exercise 1**: Compares four LLM parameter configurations (temperature,
  max_tokens) and analyses their effect on output quality.
- **Conversation memory**: Manual multi-turn history management with
  `SystemMessage`, `HumanMessage`, `AIMessage`.
- **Interactive Game Master chat loop** (from lab): Type `"Who are you?"` to
  verify role injection.

**Run:**
```bash
pip install langchain langchain-openai openai python-dotenv
export OPENAI_API_KEY=your_key_here
python lab4_langchain_prompt_engineering.py          # Demo mode
python lab4_langchain_prompt_engineering.py --chat   # Interactive chat
```

*No API key? The script falls back to demo mode showing formatted prompts.*

---

## Setup

```bash
# Clone the repo
git clone https://github.com/pumpuni07/ai-agents-labs.git
cd ai-agents-labs

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
# Lab 1
requests
python-dotenv

# Lab 2
transformers
torch
faiss-cpu

# Lab 3
sentence-transformers
torch

# Lab 4
langchain
langchain-openai
langchain-community
openai
python-dotenv
```

---

## Concepts Covered

| Concept | Labs |
|---------|------|
| Agent architecture & routing | Lab 1 |
| Text preprocessing & synonym expansion | Lab 1 |
| Dense Passage Retrieval (DPR) | Lab 2 |
| GPT-2 text generation & beam search | Lab 2 |
| Cosine similarity vs dot product | Labs 2, 3 |
| Embedding-based semantic search | Labs 2, 3 |
| Content classification with RAG | Lab 3 |
| Prompt templates & in-context engineering | Lab 4 |
| Few-shot prompting | Lab 4 |
| LangChain chains & output parsers | Lab 4 |
| LLM parameter tuning | Labs 2, 4 |
| Multi-turn conversation management | Lab 4 |

---

## References

- IBM Skills Network — AI Engineering Professional Certificate
- Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers
- LangChain Documentation: https://python.langchain.com/docs
- Sentence Transformers: https://www.sbert.net
- OpenWeatherMap API: https://openweathermap.org/api

---

*© 2024 Jack Pumpuni Frimpong-Manso. Code extended from IBM Skills Network lab materials.*
