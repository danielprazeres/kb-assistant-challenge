"""AI Agent for answering Matrix script questions using retrieved context."""

import logging
from typing import List, Tuple
from pydantic_ai import Agent
from langchain_core.documents import Document

from .config import settings

logger = logging.getLogger(__name__)


class MatrixAgent:
    """AI Agent for answering questions about The Matrix script."""

    def __init__(self):
        """Initialize the AI agent with OpenAI model."""
        import os
        # Ensure the API key is available as environment variable
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        self.agent = Agent(
            f"openai:{settings.llm_model}", system_prompt=self._get_system_prompt()
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the AI agent."""
        return """You are an AI assistant that answers questions about The Matrix movie script. 
        
        IMPORTANT RULES:
        1. Answer based on the provided context from the script
        2. ALWAYS include source citations at the end of your response in the format: "**Sources: Context 1, Context 3**" (referencing the context numbers from the provided material)
        3. If the context is limited, provide what information you can and indicate what aspects you cannot address
        4. Do NOT use external knowledge about The Matrix beyond what's in the provided context
        5. Be accurate and specific in your answers, citing specific dialogue or scenes when possible
        
        ADVANCED CAPABILITIES:
        - For COUNTING questions: Carefully count occurrences in the provided context
        - For CHARACTER ANALYSIS: Synthesize information from multiple contexts and infer personality traits from character actions, dialogue, and behavior
        - For RELATIONSHIP questions: Look for connections between characters and events across contexts
        - For COMPOUND questions: Address each part systematically
        
        RESPONSE FORMAT:
        1. Provide your answer based on the context
        2. End with source citations: "**Sources: Context X, Context Y**" (list all context numbers you used)
        
        If you cannot fully answer a question due to limited context, say something like:
        "Based on the available context, [provide what you can]. However, the script excerpts don't contain enough detail about [specific missing aspect]. **Sources: Context X**"
        
        Answer questions thoroughly based on the script context provided."""

    def _format_context(self, documents: List[Tuple[Document, float]]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant context found."

        context_parts = []
        for i, (doc, score) in enumerate(documents, 1):
            context_parts.append(f"Context {i} (relevance: {score:.3f}):")
            context_parts.append(f"Text: {doc.page_content}")

            # Add metadata if available
            if doc.metadata:
                metadata_str = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
                context_parts.append(f"Metadata: {metadata_str}")

            context_parts.append("")  # Empty line for separation

        return "\n".join(context_parts)

    async def answer_question(
        self, question: str, context_documents: List[Tuple[Document, float]]
    ) -> str:
        """Answer a question using retrieved context."""
        # Format the context (even if empty, let the LLM decide)
        context = self._format_context(context_documents)

        # Create the prompt
        prompt = f"""Based on the following context from The Matrix script, please answer this question:

Question: {question}

Context:
{context}

Answer:"""

        try:
            result = await self.agent.run(prompt)

            # Handle different return types from pydantic-ai
            if hasattr(result, "output"):
                return result.output
            elif isinstance(result, str):
                return result
            else:
                return str(result)

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}"

    async def answer_with_retriever(self, question: str, retriever) -> str:
        """Answer a question by first retrieving relevant context."""
        try:
            # Use enhanced retrieval for better coverage
            context_documents = retriever.search(question, k=100)
            
            # For certain question types, use additional search terms
            if self._needs_enhanced_retrieval(question):
                context_documents = await self._get_enhanced_context(question, retriever)

            # Generate answer using context
            answer = await self.answer_question(question, context_documents)
            return answer

        except Exception as e:
            logger.error(f"Failed to answer question with retriever: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _needs_enhanced_retrieval(self, question: str) -> bool:
        """Determine if a question would benefit from enhanced retrieval strategies."""
        question_lower = question.lower()
        
        # Questions that typically need broader context
        enhanced_keywords = [
            'how many', 'count', 'describe', 'personality', 'character',
            'offer', 'exchange', 'deal', 'crew', 'members', 'human fields',
            'purpose', 'created', 'agents want', 'capture', 'real name'
        ]
        
        return any(keyword in question_lower for keyword in enhanced_keywords)
    
    async def _get_enhanced_context(self, question: str, retriever) -> List[Tuple[Document, float]]:
        """Get enhanced context using multiple search strategies."""
        question_lower = question.lower()
        
        # Start with the main question
        context_documents = retriever.search(question, k=200)
        
        # Add related search terms based on question content
        additional_terms = self._extract_search_terms(question_lower)
        
        for term in additional_terms:
            term_docs = retriever.search(term, k=50)
            context_documents.extend(term_docs)
        
        # Deduplicate and return
        return self._deduplicate_documents(context_documents)
    
    def _extract_search_terms(self, question_lower: str) -> List[str]:
        """Extract additional search terms based on question content."""
        terms = []
        
        # Character names
        characters = ['Neo', 'Morpheus', 'Trinity', 'Cypher', 'Agent Smith', 'Tank', 'Dozer']
        for char in characters:
            if char.lower() in question_lower:
                terms.extend([char, f'{char} says', f'{char} wants'])
        
        # Specific concepts
        if 'crew' in question_lower or 'members' in question_lower:
            terms.extend(['Nebuchadnezzar', 'crew', 'team', 'ship'])
            
        if 'human fields' in question_lower or 'purpose' in question_lower:
            terms.extend(['fields', 'grown', 'machines', 'battery', 'power'])
            
        if 'agents want' in question_lower or 'capture' in question_lower:
            terms.extend(['Zion', 'access codes', 'mainframe', 'codes'])
            
        if 'offer' in question_lower or 'exchange' in question_lower:
            terms.extend(['deal', 'betrayal', 'rich', 'actor', 'Matrix'])
            
        return terms
    
    
    def _deduplicate_documents(self, docs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Remove duplicate documents while preserving order and scores."""
        seen_content = set()
        unique_docs = []
        
        for doc, score in docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append((doc, score))
                
        return unique_docs
