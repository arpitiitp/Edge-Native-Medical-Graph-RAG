# Medical Edge-Native Agentic RAG: Architecture Explained

This document provides a highly intuitive, step-by-step breakdown of the system architecture. It is designed to be easily understood by professors, stakeholders, and developers alike.

---

## 1. The Core Problem (Why did we build this?)
Hospitals have massive amounts of unstructured data (patient records, admin policies). Doctors want to use modern AI to query this data (e.g., *"What is John Doe's medication history?"*). 

However, **hospitals cannot send patient data to cloud AI services like OpenAI due to strict privacy laws (HIPAA/GDPR).** 

**The Solution:** Build a system that runs 100% locally on "Edge Hardware" (a local hospital server or laptop). All data and all AI models remain securely on-premises.

## 2. The Constraint (The Edge AI Limitation)
Running AI locally means we cannot use massive, highly intelligent models like GPT-4 (which require massive supercomputers). We must use **Small Language Models (SLMs)** like `Llama 3.2 (3B parameters)`.

While SLMs are fast and private, they are prone to *hallucinations* (making things up) when asked to perform highly complex logic, such as extracting deeply nested Graph relationships (Semantic Triples).

Our architecture is specifically engineered to **overcome the limitations of local SLMs** while still delivering highly accurate retrieval.

---

## 3. Step-by-Step Architecture Flow

### Phase 1: Data Ingestion (Building the Brain)
When a new document (e.g., a Patient Report) is added to the system, it goes through a dual-processing pipeline to ensure no information is lost.

1. **Chunking:** The document is broken down into smaller, readable paragraphs called "Chunks".
2. **Vector Pipeline (Semantic Search):**
   - Each chunk is converted into a list of numbers (a Vector Embedding) using the `nomic-embed-text` model.
   - These vectors are saved in **PostgreSQL (`pgvector`)**.
   - *Purpose:* This allows the AI to later search for "concepts" or "meanings" rather than exact keywords.
3. **Knowledge Graph Pipeline (Relational Search):**
   - We ask the local `Llama 3.2` model to read the chunk and simply extract basic nouns (Entities): `Persons`, `Conditions`, `Medications`, and `Facilities`.
   - These entities are saved into **Neo4j**.
   - *The Edge-Native Optimization (Star Schema):* Instead of forcing the weak SLM to guess the complex relationship verb between John Doe and a Medication (which causes crashes), we connect both entities directly to the Chunk they were found in. 
   - *Example:* `[John Doe]` --> `[Chunk 1]` <-- `[Aspirin]`. 
   - *Purpose:* This allows lightning-fast exact-entity lookups without hallucination risks.

---

### Phase 2: User Query & Agent Routing (The AI Brain)
When a doctor types a question into the Chat CLI (e.g., *"What medications is John Doe taking?"*):

4. **The Pydantic AI Agent Intercepts:** The Agent (powered by `Llama 3.2`) receives the question. It acts as a smart router.
5. **Tool Selection:** The Agent looks at its available tools and decides how to search for the answer:
   - **Scenario A (General Knowledge):** If the question is *"What are the hospital rules for visitors?"*, the Agent uses the **Vector Search Tool** to find paragraphs with similar semantic meaning in Postgres.
   - **Scenario B (Specific Patient/Entity):** If the question is *"What meds does John Doe take?"*, the Agent uses the **Knowledge Graph Tool**.
   - *How the Graph Tool Works:* It instantly queries Neo4j for the exact node `[Person: John Doe]`. It follows the edge to the `[Chunk]` that mentions him, which also happens to be the exact chunk that mentions his medications and conditions.

---

### Phase 3: Generation (The Final Answer)
6. **Synthesizing the Answer:** The databases return the exact, isolated text chunks to the Agent.
7. **Final Reading:** The Agent reads these specific chunks. Because it is only reading a highly relevant, small piece of text (instead of a 50-page document), the local `Llama 3.2` model can easily and accurately understand the context.
8. **Output:** The LLM generates the final human-readable response and prints it to the user's screen securely: *"John Doe is taking Lisinopril for Hypertension."*

---

## Summary of Innovation for Presentation
If you need to summarize the core research value to your professor, use this statement:

> *"This architecture implements a true Edge-Native Hybrid RAG. Standard Graph RAG relies on cloud-based LLMs to generate fragile Semantic Triples. Our system pioneers an 'Extractive Star Schema' that safely outsources simple entity extraction to local Small Language Models (SLMs), using Neo4j index-adjacency to fetch contexts, and then dynamically combining it with PostgreSQL Semantic Vector Search via a Pydantic AI routing agent. This ensures 100% data privacy and zero hallucination on constrained edge hardware."*
