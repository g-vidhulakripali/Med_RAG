import logging
from typing import Dict, Any, List, Optional
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from config import Config
from vector_store import VectorStore
from llm_manager import LLMManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(BaseModel):
    """State object for the RAG agent workflow"""
    query: str = Field(description="The user's query")
    retrieved_documents: List[Document] = Field(default_factory=list, description="Retrieved relevant documents")
    context: str = Field(default="", description="Processed context from documents")
    response: str = Field(default="", description="Generated response")
    confidence: float = Field(default=0.0, description="Confidence score for the response")
    sources: List[str] = Field(default_factory=list, description="Source documents used")
    evaluation: Dict[str, Any] = Field(default_factory=dict, description="Response quality evaluation")
    error: str = Field(default="", description="Any error messages")
    step: str = Field(default="start", description="Current step in the workflow")
    improvement_count: int = Field(default=0, description="Number of times the response has been improved")

class RAGAgent:
    """Agentic RAG system using LangGraph for autonomous decision-making"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm_manager = LLMManager()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for agentic behavior"""
        
        # Define the workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("analyze", self._analyze_query)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("evaluate", self._evaluate_response)
        workflow.add_node("improve", self._improve_response)
        
        # Define the flow
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "analyze")
        workflow.add_edge("analyze", "generate")
        workflow.add_edge("generate", "evaluate")
        
        # Conditional edge for improvement
        workflow.add_conditional_edges(
            "evaluate",
            self._should_improve,
            {
                "improve": "improve",
                "end": END
            }
        )
        workflow.add_edge("improve", "evaluate")
        
        return workflow.compile()
    
    def _retrieve_documents(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents for the query"""
        try:
            state.step = "retrieve"
            logger.info(f"Retrieving documents for query: {state.query}")
            
            # Perform similarity search
            documents = self.vector_store.similarity_search(state.query)
            
            if not documents:
                state.error = "No relevant documents found"
                return state
            
            state.retrieved_documents = documents
            state.sources = [doc.metadata.get('source', 'unknown') for doc in documents]
            
            # Create context from retrieved documents
            context_parts = []
            for doc in documents:
                source = doc.metadata.get('source', 'unknown')
                chunk_id = doc.metadata.get('chunk_id', 'unknown')
                context_parts.append(f"Source: {source} (Chunk {chunk_id})\n{doc.page_content}\n")
            
            state.context = "\n".join(context_parts)
            logger.info(f"Retrieved {len(documents)} documents")
            
        except Exception as e:
            state.error = f"Error in document retrieval: {str(e)}"
            logger.error(f"Document retrieval error: {e}")
        
        return state
    
    def _analyze_query(self, state: AgentState) -> AgentState:
        """Analyze the query to determine the best approach"""
        try:
            state.step = "analyze"
            logger.info("Analyzing query complexity and requirements")
            
            # Use LLM to analyze query complexity
            analysis_prompt = f"""Analyze this medical query and determine:
1. Query type (factual, analytical, comparative, etc.)
2. Required information depth
3. Potential challenges or ambiguities
4. Recommended approach for response generation

Query: {state.query}

Context available: {len(state.retrieved_documents)} document chunks

Provide a brief analysis:"""
            
            analysis_result = self.llm_manager.generate_response(analysis_prompt)
            
            if analysis_result["success"]:
                logger.info("Query analysis completed")
                # Store analysis in state for potential use in response generation
                state.evaluation["query_analysis"] = analysis_result["response"]
            else:
                logger.warning("Query analysis failed, proceeding with default approach")
                
        except Exception as e:
            state.error = f"Error in query analysis: {str(e)}"
            logger.error(f"Query analysis error: {e}")
        
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate initial response using retrieved context"""
        try:
            state.step = "generate"
            logger.info("Generating response using LLM")
            
            if not state.context:
                state.error = "No context available for response generation"
                return state
            
            # Generate medical response
            response_result = self.llm_manager.generate_medical_response(
                query=state.query,
                context=state.context
            )
            
            if response_result["success"]:
                state.response = response_result["response"]
                state.confidence = 0.8  # Initial confidence
                logger.info("Response generated successfully")
            else:
                state.error = f"Response generation failed: {response_result.get('error', 'Unknown error')}"
                logger.error(f"Response generation failed: {response_result}")
                
        except Exception as e:
            state.error = f"Error in response generation: {str(e)}"
            logger.error(f"Response generation error: {e}")
        
        return state
    
    def _evaluate_response(self, state: AgentState) -> AgentState:
        """Evaluate the quality and relevance of the generated response"""
        try:
            state.step = "evaluate"
            logger.info("Evaluating response quality")
            
            if not state.response:
                state.error = "No response to evaluate"
                return state
            
            # Evaluate response quality
            evaluation_result = self.llm_manager.evaluate_response_quality(
                query=state.query,
                response=state.response,
                context=state.context
            )
            
            if evaluation_result["success"]:
                state.evaluation["quality_assessment"] = evaluation_result["evaluation"]
                
                # Extract confidence score from evaluation
                # This is a simplified approach - in practice you might want more sophisticated scoring
                if "confidence" in evaluation_result["evaluation"].lower():
                    state.confidence = 0.9
                elif "good" in evaluation_result["evaluation"].lower():
                    state.confidence = 0.8
                elif "adequate" in evaluation_result["evaluation"].lower():
                    state.confidence = 0.6
                else:
                    state.confidence = 0.5
                
                logger.info(f"Response evaluation completed. Confidence: {state.confidence}")
            else:
                logger.warning("Response evaluation failed")
                state.confidence = 0.5  # Default confidence
                
        except Exception as e:
            state.error = f"Error in response evaluation: {str(e)}"
            logger.error(f"Response evaluation error: {e}")
            state.confidence = 0.5
        
        return state
    
    def _should_improve(self, state: AgentState) -> str:
        """Determine if response should be improved based on evaluation"""
        # Limit improvements to prevent infinite recursion
        if state.improvement_count >= 3:
            logger.info(f"Maximum improvement attempts ({state.improvement_count}) reached, ending workflow")
            return "end"
        
        if state.confidence < Config.CONFIDENCE_THRESHOLD:
            state.improvement_count += 1
            logger.info(f"Low confidence ({state.confidence}), improving response (attempt {state.improvement_count}/3)")
            return "improve"
        else:
            logger.info(f"Confidence ({state.confidence}) meets threshold, ending workflow")
            return "end"
    
    def _improve_response(self, state: AgentState) -> AgentState:
        """Improve the response based on evaluation feedback"""
        try:
            state.step = "improve"
            logger.info("Improving response based on evaluation")
            
            improvement_prompt = f"""The previous response to this query received a low confidence score.
Please improve the response by addressing any identified issues.

Original Query: {state.query}

Previous Response: {state.response}

Evaluation Feedback: {state.evaluation.get('quality_assessment', 'No specific feedback available')}

Context: {state.context}

Please provide an improved, more accurate, and comprehensive response:"""
            
            improvement_result = self.llm_manager.generate_response(improvement_prompt)
            
            if improvement_result["success"]:
                state.response = improvement_result["response"]
                state.confidence = min(0.95, state.confidence + 0.1)  # Increase confidence
                logger.info("Response improved successfully")
            else:
                logger.warning("Response improvement failed")
                
        except Exception as e:
            state.error = f"Error in response improvement: {str(e)}"
            logger.error(f"Response improvement error: {e}")
        
        return state

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query through the complete RAG workflow"""
        try:
            logger.info(f"Processing query: {query}")

            # Initialize state
            initial_state = AgentState(query=query)

            # Run the workflow
            final_state = self.workflow.invoke(initial_state)

            # If final_state is a dict (LangGraph style)
            if isinstance(final_state, dict):
                response = final_state.get("response") or final_state.get("answer") or final_state.get("output")
                confidence = final_state.get("confidence", 0.0)
                sources = final_state.get("sources", [])
                evaluation = final_state.get("evaluation", {})
                workflow_steps = final_state.get("step", "N/A")
                error = final_state.get("error")
            else:
                # If final_state is a custom object with attributes
                response = getattr(final_state, "response", None)
                confidence = getattr(final_state, "confidence", 0.0)
                sources = getattr(final_state, "sources", [])
                evaluation = getattr(final_state, "evaluation", {})
                workflow_steps = getattr(final_state, "step", "N/A")
                error = getattr(final_state, "error", None)

            result = {
                "success": True,
                "query": query,
                "response": response,
                "confidence": confidence,
                "sources": sources,
                "evaluation": evaluation,
                "workflow_steps": workflow_steps,
                "error": error,
            }

            logger.info(f"Query processing completed with confidence: {confidence}")
            return result

        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "response": None,
                "confidence": 0.0,
                "sources": [],
                "evaluation": {},
                "workflow_steps": "N/A"
            }

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health"""
        try:
            vector_stats = self.vector_store.get_collection_stats()
            llm_models = self.llm_manager.get_available_models()
            
            return {
                "vector_store": vector_stats,
                "llm_models": llm_models,
                "config": {
                    "chunk_size": Config.CHUNK_SIZE,
                    "chunk_overlap": Config.CHUNK_OVERLAP,
                    "top_k_retrieval": Config.TOP_K_RETRIEVAL,
                    "confidence_threshold": Config.CONFIDENCE_THRESHOLD
                }
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
