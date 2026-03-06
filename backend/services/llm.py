import httpx
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3" # Phi-3 Mini (3.8B)

async def generate_response(query: str, context: str) -> str:
    """
    Sends the user query and the hybrid context to the local Ollama Phi-3 model.
    Enforces strict grounding on the provided context.
    """
    
    system_prompt = (
        "You are an expert medical AI assistant. Your purpose is to answer the user's query "
        "STRICTLY using the context provided below. The context is derived from medical documents "
        "and a Knowledge Graph. \n\n"
        "RULES:\n"
        "1. If the answer is not contained within the provided context, you MUST state 'I don't have enough information to answer that based on the provided documents.'\n"
        "2. Do not invent, hallucinate, or bring in outside medical knowledge.\n"
        "3. Use the KNOWLEDGE GRAPH context to connect related symptoms or side-effects.\n"
        "4. Be concise, professional, and directly address the user's question.\n"
    )
    
    full_prompt = f"CONTEXT:\n{context}\n\nUSER QUERY:\n{query}"
    
    payload = {
        "model": MODEL_NAME,
        "system": system_prompt,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.0, # Complete determinism to prevent hallucination
            "seed": 42
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "Error: No response generated.")
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return f"Error: Could not connect to the local inference engine. Details: {str(e)}"
