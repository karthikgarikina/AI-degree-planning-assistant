# 🎓 AI Degree Planning Assistant

An autonomous AI agent that reads a student transcript, understands academic progress, and generates a structured degree plan using **LangGraph**, **LLMs (Ollama)**, and **Pydantic validation**.

---

## 🚀 Overview

This project demonstrates a **production-style agentic AI system** that can:

- 📄 Parse transcript data (PDF / TXT)
- 🧠 Reason using an LLM (via Ollama)
- 🔧 Use tools to fetch course data and validate decisions
- 📅 Plan future semesters intelligently
- ⚠️ Detect schedule conflicts & prerequisite issues
- 📝 Generate structured logs and final output

---

## 🧠 How It Works (High-Level Flow)

```
Transcript → Parser → Agent (LangGraph Loop) → Tool Calls → Validation → Final Degree Plan
```

---

## 📂 Project Structure

```
.
├── main.py                  # CLI entry point
├── agent.py                 # LangGraph agent logic
├── models.py                # Pydantic models & validators
├── tools.py                 # Course search & validation tools
├── pdf_parser.py            # Transcript parsing logic
├── config.py                # Environment config
│
├── data/
│   ├── course_catalog.json
│   ├── program_requirements.json
│   ├── sample_transcript.pdf
│
├── output/
│   ├── agent.log
│   └── degree_plan.json
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## ⚙️ Setup Instructions

### Clone & Setup

```
git clone https://github.com/karthikgarikina/AI-degree-planning-assistant
cd AI-degree-planning-assistant
cp .env.example .env
```

## 1️⃣ 🐳Docker Setup (Recommended)

### Build & Run

```
docker-compose up --build          # this will go with default goal and transcript.
```

### Run Custom Command

```
docker compose build
docker compose run --rm app   --transcript ./data/sample_transcript.pdf   --goal "Plan my next semester"
```


---

### 2️⃣ Install Dependencies (Local)  (another way)

```
pip install -r requirements.txt
```

---

### Run Locally

```
python main.py   --transcript ./data/sample_transcript.txt   --goal "Plan my next two semesters to finish the AI minor"
```

---
## 📥 Output

After execution:

### 📄 Degree Plan
```
output/degree_plan.json
```

### 📊 Logs (Structured JSON)
```
output/agent.log
```

---

## 🧪 Example Input

**Transcript (TXT):**
```
CS101 Intro Programming
CS201 Data Structures
CS301 Algorithms
```

---

## 📈 Example Output

```
Fall 2026:
- Machine Learning
- AI Ethics

Spring 2027:
- Deep Learning
```

---

## 🧩 Key Features

- ✅ LangGraph ReAct loop (agent ↔ tools)
- ✅ Tool-based reasoning (search, prerequisites, validation)
- ✅ Pydantic validation (credit + schedule conflict detection)
- ✅ Structured logging (JSON logs for debugging)
- ✅ Dockerized environment
- ✅ Works with Ollama (local LLM)

---

## ⚠️ Agent Safeguards

- 🔁 Step limit (`AGENT_MAX_STEPS`)
- ❌ Prevents infinite loops
- ⚠️ Handles tool errors gracefully
- 🔄 Fallback mode if LLM unavailable

---

## 🎥 Demo Video

```
https://www.youtube.com/watch?v=5SZTjTo7M2s
```

---

## 🧠 Design Highlights

- Modular architecture (agent / tools / models)
- Clear separation of concerns
- Easily extensible for real APIs
- Production-style logging & validation

---

## 🏁 Final Notes

This project focuses on **agent architecture, tool usage, and system design** rather than perfect academic planning.

---
