"""
Pydantic AI Agents for the Medical RAG System.
"""

import os
import logging
from typing import Dict, Any, List

# Check if pydantic_ai is available
try:
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models.openai import OpenAIModel
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("pydantic_ai is not installed. Please pip install pydantic-ai")
    Agent = None
    RunContext = None

from agent.tools import search_medical_guidelines, search_knowledge_graph
from agent.models import AgentDependencies

logger = logging.getLogger(__name__)


from agent.providers import get_llm_model

def create_medical_agent() -> Agent:
    """
    Create the core medical routing agent.
    
    This agent will have access to the Vector Search (PostgreSQL) and 
    Knowledge Graph Search (Neo4j) tools.
    """
    
    # We use our standard configured LLM model pointing to Ollama
    ollama_model = get_llm_model()
    
    system_prompt = """
    You are an advanced Edge-Native Medical RAG Assistant running securely within a hospital network.
    Your primary goal is to securely route and answer questions based solely on the provided hospital context.
    
    You have TWO distinct search tools at your disposal:
    1. VECTOR SEARCH (`search_medical_documents`): Use this when the user asks about general guidelines, policies, or semantic meaning from long text reports (e.g., "What is the hospital's policy on Covid?", "Show me reports about patients with chest pain").
    2. GRAPH SEARCH (`search_knowledge_graph`): Use this when the user asks about specific entities, timelines, or precise relationships (e.g., "What conditions does Patient John Doe have?", "Who treated John Doe?", "What medications are related to Hypertension?").
    
    INSTRUCTIONS:
    - If the user asks a casual/conversational query (e.g., "Hello", "How are you", "Who are you"), DO NOT CALL A FUNCTION. You MUST respond with normal, plain conversational text answering the user directly. 
    - NEVER output raw JSON or null function calls (e.g., `{"name": "function <nil>"}`). Just write a normal text reply.
    - Ask for clarification if the medical query is too vague.
    - For medical queries, ALWAYS search the context using the provided tools before answering. 
    - You may use both tools if needed to construct a complete answer.
    - If the user asks for a patient timeline or relationships, lean heavily on the GRAPH SEARCH.
    - Do NOT hallucinate medical data. If the answer is not in the search results, state clearly that you do not have that information in the secure local database.
    """
    
    # Initialize the Agent
    agent = Agent(
        ollama_model,
        system_prompt=system_prompt,
        deps_type=AgentDependencies
    )
    
    # Register the tools with the agent
    @agent.tool
    async def search_vector_db(ctx: RunContext[AgentDependencies], query: str, limit: int = 5) -> str:
        """Search the Postgres pgvector database for medical guidelines and long-text notes."""
        results = await search_medical_guidelines(query, limit)
        return str(results)
        
    @agent.tool
    async def search_graph_db(ctx: RunContext[AgentDependencies], entity_name: str, limit: int = 10) -> str:
        """Search the Neo4j Knowledge Graph for specific entities (Persons, Conditions, Medications, Facilities)."""
        results = await search_knowledge_graph(entity_name, limit)
        return str(results)
        
    return agent


# Instantiate a singleton agent for the app to import
medical_agent = create_medical_agent() if Agent else None

