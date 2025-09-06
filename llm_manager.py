import logging
from typing import Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMManager:
    """Manages different LLM providers with fallback mechanisms"""
    
    def __init__(self):
        self.google_llm = None
        self.openai_llm = None
        self.ollama_llm = None
        self.current_llm = None
        self.initialize_llms()
    
    def initialize_llms(self):
        """Initialize available LLM instances"""
        try:
            # Initialize Google Gemini
            if Config.GOOGLE_API_KEY:
                self.google_llm = ChatGoogleGenerativeAI(
                    google_api_key=Config.GOOGLE_API_KEY,
                    model=Config.DEFAULT_MODEL,
                    temperature=Config.TEMPERATURE,
                    max_tokens=Config.MAX_TOKENS
                )
                logger.info(f"Initialized Google Gemini LLM with model: {Config.DEFAULT_MODEL}")
            
            # Initialize OpenAI
            if Config.OPENAI_API_KEY:
                self.openai_llm = ChatOpenAI(
                    openai_api_key=Config.OPENAI_API_KEY,
                    model=Config.FALLBACK_MODEL,
                    temperature=Config.TEMPERATURE,
                    max_tokens=Config.MAX_TOKENS
                )
                logger.info(f"Initialized OpenAI LLM with model: {Config.FALLBACK_MODEL}")
            
            # Initialize Ollama (local)
            try:
                self.ollama_llm = ChatOllama(
                    model=Config.OLLAMA_MODEL,
                    base_url=Config.OLLAMA_BASE_URL,
                    temperature=Config.TEMPERATURE
                )
                logger.info(f"Initialized Ollama LLM with model: {Config.OLLAMA_MODEL}")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama LLM: {e}")
                self.ollama_llm = None
            
            # Set current LLM
            self.current_llm = self.google_llm or self.openai_llm or self.ollama_llm
            
        except Exception as e:
            logger.error(f"Error initializing LLMs: {e}")
            # Try to fallback to any available LLM
            self.current_llm = self.openai_llm or self.ollama_llm
    
    def get_llm(self, provider: str = "auto") -> Optional[Any]:
        """Get LLM instance for specified provider"""
        if provider == "google" and self.google_llm:
            return self.google_llm
        elif provider == "openai" and self.openai_llm:
            return self.openai_llm
        elif provider == "ollama" and self.ollama_llm:
            return self.ollama_llm
        elif provider == "auto":
            return self.current_llm
        return None
    
    def generate_response(self, prompt: str, provider: str = "auto") -> Dict[str, Any]:
        """Generate response using specified or best available LLM"""
        try:
            llm = self.get_llm(provider)
            if not llm:
                return {"success": False, "error": "No LLM available"}
            
            # Try the specified provider first
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                return {
                    "success": True,
                    "response": response.content,
                    "provider": provider if provider != "auto" else "auto-selected"
                }
            except Exception as e:
                logger.warning(f"Primary LLM failed, trying fallbacks: {e}")
                
                # Try fallback providers
                fallback_providers = []
                if provider != "google" and self.google_llm:
                    fallback_providers.append(("google", self.google_llm))
                if provider != "openai" and self.openai_llm:
                    fallback_providers.append(("openai", self.openai_llm))
                if provider != "ollama" and self.ollama_llm:
                    fallback_providers.append(("ollama", self.ollama_llm))
                
                for fallback_name, fallback_llm in fallback_providers:
                    try:
                        response = fallback_llm.invoke([HumanMessage(content=prompt)])
                        logger.info(f"Successfully used fallback LLM: {fallback_name}")
                        return {
                            "success": True,
                            "response": response.content,
                            "provider": f"fallback-{fallback_name}"
                        }
                    except Exception as fallback_e:
                        logger.warning(f"Fallback LLM {fallback_name} failed: {fallback_e}")
                        continue
                
                return {"success": False, "error": "All LLM providers failed"}
                
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_medical_response(self, query: str, context: str) -> Dict[str, Any]:
        """Generate medical-specific response with context"""
        medical_prompt = f"""You are a medical AI assistant. Please provide a comprehensive, accurate, and evidence-based response to the following medical query.

Query: {query}

Context from medical documents:
{context}

Please provide a detailed response that:
1. Directly addresses the query
2. References specific information from the provided context
3. Maintains medical accuracy and professionalism
4. Includes relevant source citations where possible

Response:"""
        
        return self.generate_response(medical_prompt)
    
    def evaluate_response_quality(self, query: str, response: str, context: str) -> Dict[str, Any]:
        """Evaluate the quality of a generated response"""
        evaluation_prompt = f"""Evaluate the quality of this medical AI response:

Query: {query}

Response: {response}

Context used: {context}

Rate the response on:
1. Relevance to the query (0-10)
2. Accuracy of medical information (0-10)
3. Use of provided context (0-10)
4. Overall helpfulness (0-10)

Provide a brief assessment and overall score (0-10):"""
        
        result = self.generate_response(evaluation_prompt)
        if result["success"]:
            return {
                "success": True,
                "evaluation": result["response"]
            }
        return result
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available LLM models"""
        models = {}
        
        if self.google_llm:
            models["google"] = {
                "model": Config.DEFAULT_MODEL,
                "status": "available"
            }
        
        if self.openai_llm:
            models["openai"] = {
                "model": Config.FALLBACK_MODEL,
                "status": "available"
            }
        
        if self.ollama_llm:
            models["ollama"] = {
                "model": Config.OLLAMA_MODEL,
                "status": "available",
                "base_url": Config.OLLAMA_BASE_URL
            }
        
        return models
