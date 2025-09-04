"""
Core chatbot implementation that manages token usage and conversation history.
"""
from typing import Dict, List, Optional, Union, Any, AsyncGenerator
import logging
from functools import wraps
import asyncio
import uuid

from langchain_core.messages import (
    AIMessage, 
    BaseMessage, 
    HumanMessage, 
    SystemMessage
)
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.exceptions import LangChainException
from langchain_core.documents import Document

from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import ConnectionPool, AsyncConnectionPool

from ..models.state import ChatbotState
from ..utils.rag_utils import get_context_source, format_source
from ..utils.db_utils import setup_db_checkpoint_table

# Import prompt strings from prompts.py
from .prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_CITATION_INSTRUCTIONS,
    DEFAULT_MAX_TOKENS_MESSAGE,
    DEFAULT_QUERY_PROCESSING_PROMPT,
    CONTEXT_SECTION_TEMPLATE,
    CONTEXT_REFERENCE_TEMPLATE,
    CITATION_INSTRUCTIONS_SECTION_TEMPLATE,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatbotError(Exception):
    """Base exception class for chatbot errors."""
    pass


class TokenLimitError(ChatbotError):
    """Raised when token limits are exceeded."""
    pass


class ModelError(ChatbotError):
    """Raised when there's an error with the model."""
    pass


def handle_exceptions(func):
    """Decorator to handle exceptions in chatbot methods."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except LangChainException as e:
            # Handle LangChain specific errors
            logger.error(f"LangChain error: {str(e)}")
            raise ModelError(f"Error with language model: {str(e)}") from e
        except Exception as e:
            # Handle other exceptions
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise ChatbotError(f"Unexpected error: {str(e)}") from e
    return wrapper


def handle_async_exceptions(func):
    """Decorator to handle exceptions in async chatbot methods."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except LangChainException as e:
            # Handle LangChain specific errors
            logger.error(f"LangChain error: {str(e)}")
            raise ModelError(f"Error with language model: {str(e)}") from e
        except Exception as e:
            # Handle other exceptions
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise ChatbotError(f"Unexpected error: {str(e)}") from e
    return wrapper


class ConfigurableChatbot:
    """A configurable chatbot that manages token usage and conversation history."""
    
    def __init__(
        self,
        model: BaseChatModel,
        retriever: Optional[BaseRetriever] = None,
        is_async: bool = True,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        citation_instructions: str = DEFAULT_CITATION_INSTRUCTIONS,
        max_history_tokens: int = 10000,
        keep_indices: Optional[List[int]] = None,
        max_tokens_behavior: str = "trim",  # Options: "trim" or "warn"
        max_tokens_message: str = DEFAULT_MAX_TOKENS_MESSAGE,
        query_processing_prompt: str = DEFAULT_QUERY_PROCESSING_PROMPT,
        checkpointer: Optional[PostgresSaver | AsyncPostgresSaver | str] = None
    ):
        """
        Initialize the chatbot.
        
        Args:
            model: The LLM to use for chat
            retriever: Optional retriever for RAG functionality (can be sync or async)
            system_prompt: The system prompt to use
            citation_instructions: Instructions for citing retrieved context
            max_history_tokens: Maximum number of tokens to include from history
            keep_indices: Message indices to always keep (e.g., [0] to keep system message)
            max_tokens_behavior: How to handle reaching max tokens: "trim" (default, remove old messages) or "warn" (add warning message)
            max_tokens_message: Message to display when max tokens are reached (only used with max_tokens_behavior="warn")
            query_processing_prompt: The prompt to use when processing the first user message for better retrieval
            checkpointer: Either a BaseCheckpointSaver instance, or a PostgreSQL connection string
        """
        self.model = model
        self.retriever = retriever
        self.is_async = is_async
        self.base_system_prompt = system_prompt
        self.citation_instructions = citation_instructions
        self.max_history_tokens = max_history_tokens
        self.keep_indices = keep_indices or [0]  # By default, keep the system message
        self.max_tokens_behavior = max_tokens_behavior
        self.max_tokens_message = max_tokens_message
        self.query_processing_prompt = query_processing_prompt
        # Track threads that have reached their token limit
        self._blocked_threads = set()
        # Store original messages for debugging
        self._original_messages = {}
        # Flag to check if retriever is async
        self._is_async_retriever = hasattr(retriever, 'aretrieve') if retriever else False
        
        # Initialize checkpointer
        if isinstance(checkpointer, str):
            # Validate that the string is a PostgreSQL connection string
            if not checkpointer.startswith("postgresql://"):
                raise ValueError("Invalid checkpointer string: must be a valid PostgreSQL connection string starting with 'postgresql://'")
            
            self._pg_connection_string = checkpointer
            # Setup the database tables
            setup_db_checkpoint_table(self._pg_connection_string)
            
            # For async usage we need AsyncConnectionPool
            if hasattr(self, "is_async") and self.is_async:
                # Create AsyncPostgresSaver with a lazy pool initialization
                self._async_pool = None  # Will be initialized on first use
                
                class LazyAsyncPostgresSaver(AsyncPostgresSaver):
                    def __init__(self, conn_string, owner):
                        self.conn_string = conn_string
                        self.owner = owner
                        self.pool = None
                        self._initialized = False
                        
                    async def _ensure_initialized(self):
                        if not self._initialized:
                            self.pool = AsyncConnectionPool(self.conn_string, min_size=1, max_size=5)
                            await self.pool.open()
                            # Call parent init after pool is ready
                            super().__init__(conn=self.pool)
                            await self.setup()
                            self._initialized = True
                            logger.info("AsyncPostgresSaver initialized on first use")
                    
                    async def aget(self, *args, **kwargs):
                        await self._ensure_initialized()
                        return await super().aget(*args, **kwargs)
                        
                    async def aput(self, *args, **kwargs):
                        await self._ensure_initialized()
                        return await super().aput(*args, **kwargs)
                
                # Create our lazy saver
                self.checkpointer = LazyAsyncPostgresSaver(self._pg_connection_string, self)
                logger.info("Created LazyAsyncPostgresSaver - initialization deferred until first use")
            
            # Use synchronous mode
            else:
                # Create a regular ConnectionPool
                pool = ConnectionPool(self._pg_connection_string, min_size=1, max_size=5)
                self.checkpointer = PostgresSaver(conn=pool)
                
                # Initialize the database tables
                try:
                    self.checkpointer.setup()
                    logger.info("PostgresSaver setup completed")
                except Exception as e:
                    logger.error(f"Error during PostgresSaver setup: {str(e)}")
        elif checkpointer is not None:
            self.checkpointer = checkpointer
        else:
            self.checkpointer = InMemorySaver()

        # Set up the workflow
        try:
            self._setup_workflow()
        except Exception as e:
            logger.error(f"Failed to set up workflow: {str(e)}")
            raise ChatbotError(f"Failed to initialize chatbot workflow: {str(e)}") from e
        
    def _setup_workflow(self):
        """Set up the LangGraph workflow."""
        workflow = StateGraph(state_schema=ChatbotState)
        
        # Process user query to optimize for retrieval
        def process_initial_query(state: ChatbotState):
            if not state.get("is_first_message", False) or not state.get("messages"):
                return {"processed_query": None}
                
            # Get the user's query from the last message
            user_query = state["messages"][-1].content
            logger.info(f"Processing initial query for retrieval: {user_query}")
            
            try:
                # Store original message for debugging
                if hasattr(self.app, "_current_config") and hasattr(self.app._current_config, "get"):
                    config = getattr(self.app, "_current_config", {})
                    if isinstance(config, dict) and "configurable" in config:
                        thread_id = config["configurable"].get("thread_id")
                
                self._original_messages[thread_id] = user_query
                
                # Process the query using the LLM
                processed_query = self._process_query_with_llm(user_query)
                logger.info(f"Processed query: {processed_query}")
                
                # Replace the original message with the processed one
                state["messages"][-1] = HumanMessage(content=processed_query)
                
                return {"processed_query": processed_query}
            except Exception as e:
                logger.error(f"Error processing initial query: {str(e)}")
                # If processing fails, use the original query
                return {"processed_query": None}
                
        # Define retrieval node to fetch context on first message
        def retrieve_context(state: ChatbotState):
            # Skip if we don't have a retriever
            if not self.retriever:
                return {"retrieved_context": None}
            
            # If not the first message, preserve any existing retrieved_context
            if not state.get("is_first_message", False):
                # Return any existing retrieved_context without change
                existing_context = state.get("retrieved_context")
                return {"retrieved_context": existing_context}
            
            # Get the user's query from the last message
            query = state["messages"][-1].content
            logger.info(f"Retrieving context for query: {query}")
            
            try:
                # Retrieve relevant documents - check if we're using async retriever
                if self._is_async_retriever:
                    # Create a new event loop if needed
                    try:
                        loop = asyncio.get_running_loop()
                        if loop.is_running():
                            # Use run_coroutine_threadsafe if we're in a running loop
                            future = asyncio.run_coroutine_threadsafe(
                                self.retriever.ainvoke(query), loop
                            )
                            retrieved_docs = future.result()
                        else:
                            # We have a loop but it's not running
                            retrieved_docs = loop.run_until_complete(self.retriever.ainvoke(query))
                    except RuntimeError:
                        # No event loop, create one
                        retrieved_docs = asyncio.run(self.retriever.ainvoke(query))
                else:
                    # Synchronous retrieval
                    retrieved_docs = self.retriever.invoke(query)
                
                logger.info(f"Retrieved {len(retrieved_docs)} documents")

                # Format the context source
                for doc in retrieved_docs:
                    doc.metadata["source"] = format_source(doc.metadata.get("source", "Unknown"))
                
                return {
                    "retrieved_context": retrieved_docs,
                    "is_new_context": True
                }
            except Exception as e:
                logger.error(f"Error retrieving context: {str(e)}")
                return {"retrieved_context": []}
        
        # Define the function that processes messages and calls the model
        def call_model(state: ChatbotState):
            try:
                token_count = 0
                token_array = []                
                # Create template with system prompt and messages
                if state.get("is_first_message", True):
                    state["messages"].insert(0, SystemMessage(content=state["system_prompt"]))
                    system_message_tokens = self._count_tokens(state["messages"][0])
                    token_count += system_message_tokens
                    token_array.append(system_message_tokens)

                # If this is the first message, we need to add the retrieved context
                if state.get("is_first_message", True):
                    context = state.get("retrieved_context")
                    if context:
                        last_user_message = self._create_user_message_with_context(
                            state["messages"][-1].content, context
                        )
                        # Update that this is not longer the first message
                        state["is_first_message"] = False
                        # Replace the last user message
                        state["messages"][-1] = HumanMessage(content=last_user_message)

                # If context was retrieved, return it in the output
                turn_context = None
                if state.get("retrieved_context", False):
                    turn_context = state.get("retrieved_context")

                # Get trimmed messages respecting token limits and keep_indices
                trimmed_messages = self._trim_messages(state["messages"], state["token_array"])
                if trimmed_messages is not None:
                    state["messages"] = trimmed_messages
                    state["token_array"] = [self._count_tokens(msg) for msg in trimmed_messages[:-1]]  # Don't count the last user message, we do it below
                    state["token_count"] = sum(state["token_array"])
                

                response = self.model.invoke(state["messages"])
                
                # Token stats update
                last_user_message_tokens = self._count_tokens(state["messages"][-1])
                response_tokens = response.usage_metadata.get("output_tokens", 0)
                token_count += last_user_message_tokens + response_tokens
                token_array.extend([last_user_message_tokens, response_tokens])
                
                return {
                    "messages": [response],
                    "retrieved_context": turn_context,
                    "token_count": token_count,
                    "token_array": token_array,
                }
            except Exception as e:
                logger.error(f"Error in call_model: {str(e)}")
                # Create an error message as a fallback
                error_message = AIMessage(content=f"I encountered an error processing your request. Please try again later.")
                return {
                    "messages": [error_message],
                }
        
        # Add the nodes to the workflow
        workflow.add_edge(START, "process_query")
        workflow.add_node("process_query", process_initial_query)
        workflow.add_edge("process_query", "retrieve")
        workflow.add_node("retrieve", retrieve_context)
        workflow.add_edge("retrieve", "model")
        workflow.add_node("model", call_model)
        workflow.add_edge("model", END)
        
        # Set up memory persistence
        self.app = workflow.compile(checkpointer=self.checkpointer)
    
    def _process_query_with_llm(self, query: str) -> str:
        """
        Process the initial user query using the LLM to optimize it for retrieval.
        
        Args:
            query: The original user query
            
        Returns:
            The processed query optimized for retrieval
        """
        try:
            # Create a prompt to instruct the model how to process the query
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.query_processing_prompt),
                ("human", query)
            ])
            
            # Invoke the model
            formatted_prompt = prompt.invoke({})
            response = self.model.invoke(formatted_prompt)
            
            # Return the processed query
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error processing query with LLM: {str(e)}")
            # In case of error, return the original query
            return query

    async def _async_process_query_with_llm(self, query: str) -> str:
        """
        Async version of _process_query_with_llm.
        Process the initial user query using the LLM to optimize it for retrieval.
        
        Args:
            query: The original user query
            
        Returns:
            The processed query optimized for retrieval
        """
        try:
            # Create a prompt to instruct the model how to process the query
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.query_processing_prompt),
                ("human", query)
            ])
            
            # Invoke the model asynchronously
            formatted_prompt = await prompt.ainvoke({})
            response = await self.model.ainvoke(formatted_prompt)
            
            # Return the processed query
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error processing query with LLM asynchronously: {str(e)}")
            # In case of error, return the original query
            return query
    
    def _create_user_message_with_context(self, base_prompt: str, context: List[Document]) -> str:
        """
        Create a user message that includes retrieved context and citation instructions.
        
        Args:
            base_prompt: The base user message
            context: List of retrieved documents
            
        Returns:
            User message with context and citation instructions
        """
        if not context or len(context) == 0:
            return base_prompt
        
        # Use templates from prompts.py
        context_str = CONTEXT_SECTION_TEMPLATE
        for i, doc in enumerate(context, 1):
            source = get_context_source(doc.metadata)
            id = doc.metadata.get("id", 'Unknown').split('/')[0]

            context_str += CONTEXT_REFERENCE_TEMPLATE.format(
                index=i,
                source=source,
                id=id,
                content=doc.page_content
            )
        
        # Combine base prompt, context, and citation instructions using templates
        full_prompt = (
            f"{base_prompt}"
            f"{context_str}"
            f"{CITATION_INSTRUCTIONS_SECTION_TEMPLATE.format(instructions=self.citation_instructions)}"
        )
        return full_prompt
            
    def _count_tokens(self, message: BaseMessage) -> int:
        """Count tokens in a message."""
        try:
            if isinstance(message, (SystemMessage, HumanMessage, AIMessage)):
                return self.model.get_num_tokens(message.content)
            # Fallback for other message types
            else:
                return self.model.get_num_tokens(str(message.content))
        except Exception as e:
            logger.warning(f"Error counting tokens: {str(e)}")
            # Return an estimated count based on text length as fallback
            return len(message.content) // 4  # Rough estimate (4 chars ~= 1 token)
    
    def _trim_messages(self, messages: List[BaseMessage], token_counts: List[int]) -> Optional[List[BaseMessage]]:
        """
        Trim messages to fit within token limits while keeping specified indices.
        If max_tokens_behavior is set to "warn", will return a single warning message 
        indicating the conversation limit has been reached.
        
        Args:
            messages: List of messages to trim
            token_counts: List mapping message indices to token counts
            
        Returns:
            Trimmed list of messages or a warning message
        """
        if not messages:
            return []
            
        # Calculate total tokens in all messages
        total_tokens = sum(token_counts)
        
        # Check if we're over the limit
        if total_tokens <= self.max_history_tokens:
            return
            
        # If we're using "warn" behavior, return just a warning message
        if self.max_tokens_behavior == "warn":
            # Get the thread_id from the current context config if possible
            if hasattr(self.app, "_current_config") and hasattr(self.app._current_config, "get"):
                config = getattr(self.app, "_current_config", {})
                if isinstance(config, dict) and "configurable" in config:
                    thread_id = config["configurable"].get("thread_id")
            
            # Mark this thread as blocked
            self._blocked_threads.add(thread_id)
            
            warning_message = AIMessage(content=self.max_tokens_message)
            return [warning_message]
            
        # Otherwise, use the trim behavior (default)
        # Always keep the messages at specified indices
        keep_messages = {i: messages[i] for i in self.keep_indices if i < len(messages)}
        keep_tokens = sum(token_counts[i] for i in self.keep_indices if i < len(token_counts))
        
        # How many tokens we have available for other messages
        available_tokens = max(0, self.max_history_tokens - keep_tokens)
        
        # Start with most recent messages and work backward
        result = []
        current_tokens = 0
        
        # Process messages in reverse order (newest first)
        for i in range(len(messages) - 1, -1, -1):
            # Skip if this index is in keep_messages (we'll add them later)
            if i in keep_messages:
                continue
                
            message = messages[i]
            tokens = token_counts[i]
            
            # If we can fit this message, add it
            if current_tokens + tokens <= available_tokens:
                result.insert(0, message)  # Add to front to preserve order
                current_tokens += tokens
            else:
                # Can't fit more messages
                break
        
        # Add the kept messages at their original positions
        for idx, msg in sorted(keep_messages.items()):
            # Insert only if the index is within the bounds of our result
            if idx <= len(result):
                result.insert(idx, msg)
            else:
                # If the index is beyond our result length, just append
                result.append(msg)
        
        return result
    
    @handle_exceptions
    def chat(
        self, 
        message: str, 
        thread_id: Optional[str] = None, 
        stream: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[List[BaseMessage], Any]:
        """
        Send a message to the chatbot and get a response.
        
        Args:
            message: The user message
            thread_id: The conversation thread ID
            stream: Whether to stream the response
            model_kwargs: Dictionary of kwargs to pass to the underlying model (temperature, etc.)
            **kwargs: Additional kwargs to pass to the LangGraph runtime
            
        Returns:
            The model response if stream=False, otherwise a generator
        """
        # Check if the thread is blocked
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        elif thread_id in self._blocked_threads:
            error_message = AIMessage(content="This conversation has reached its token limit and is blocked.")
            return {"response": [error_message]}
        
        # Create the human message
        human_message = HumanMessage(content=message)
        
        # Set up config with thread_id
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id}
        }
        
        # Add model kwargs if provided
        if model_kwargs:
            config["model_kwargs"] = model_kwargs
        
        # Add additional kwargs if provided
        if kwargs:
            config.update(kwargs)
        
        # Determine if this is the first message in a thread
        is_first_message = False
        thread_id_key = {"configurable": {"thread_id": thread_id}}
        if not hasattr(self.app.checkpointer, "get") or not self.app.checkpointer.get(thread_id_key):
            is_first_message = True
            logger.info(f"First message in thread {thread_id}, will retrieve context")
        
        # Set up the initial state or update existing state
        input_state = {
            "messages": [human_message],
            "system_prompt": self.base_system_prompt,
            "token_count": 0,
            "token_array": [],
            "is_new_context": False,
            "is_first_message": is_first_message,
            "original_query": message if is_first_message else None
        }
        
        # Stream or invoke
        try:
            if stream:
                return self.app.stream(
                    input_state,
                    config,
                    stream_mode="messages"
                )
            else:
                result = self.app.invoke(input_state, config)
                return {
                    "response": result["messages"],
                    "retrieved_context": result.get("retrieved_context"),
                }
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            # Return a fallback message in case of error
            error_message = AIMessage(content="I encountered an error. Please try again later.")
            return {"response": [error_message]}
    
    @handle_async_exceptions
    async def achat(
        self, 
        message: str, 
        thread_id: Optional[str] = None, 
        stream: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[List[BaseMessage], AsyncGenerator[BaseMessage, None]]:
        """
        Async version of chat method.
        Send a message to the chatbot and get a response asynchronously.
        
        Args:
            message: The user message
            thread_id: The conversation thread ID
            stream: Whether to stream the response
            model_kwargs: Dictionary of kwargs to pass to the underlying model
            **kwargs: Additional kwargs to pass to the LangGraph runtime
            
        Returns:
            The model response if stream=False, otherwise an async generator
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        elif thread_id in self._blocked_threads:
            reason = self._blocked_threads[thread_id]
            error_message = AIMessage(content=f"This thread is blocked. Reason: {reason}")
            if stream:
                async def error_generator():
                    yield error_message
                return error_generator()
            else:
                return {"response": [error_message]}
        
        # Save the original message for debugging
        self._original_messages[thread_id] = message
        
        # Process the message with the query processor if this is the first message
        is_first_message = True
        existing_history = await self.async_get_chat_history(thread_id)
        if existing_history:
            is_first_message = False
        
        # If this is the first message, we might want to process it with the query processor
        if is_first_message and self.query_processing_prompt:
            try:
                processed_message = await self._async_process_query_with_llm(message)
                if processed_message and processed_message != message:
                    logger.info(f"Processed query: {message} -> {processed_message}")
                    message = processed_message
            except Exception as e:
                logger.warning(f"Error processing query: {str(e)}")
                # Continue with the original message
        
        # Create the human message
        human_message = HumanMessage(content=message)
        
        # Create the config
        config = {"configurable": {"thread_id": thread_id, "model_kwargs": model_kwargs or {}}}
        
        # Set up the initial state or update existing state
        input_state = {
            "messages": [human_message],
            "system_prompt": self.base_system_prompt,
            "token_count": 0,
            "token_array": [],
            "is_first_message": is_first_message,
            "original_query": message if is_first_message else None
        }
        
        # Stream or invoke
        try:
            if stream:
                return self.app.astream(
                    input_state,
                    config,
                    stream_mode="messages"
                )
            else:
                result = await self.app.ainvoke(input_state, config)
                return {
                    "response": result["messages"],
                    "retrieved_context": result.get("retrieved_context"),
                }
        except Exception as e:
            logger.error(f"Error in async_chat: {str(e)}")
            # Return a fallback message in case of error
            error_message = AIMessage(content="I encountered an error. Please try again later.")
            return {"response": [error_message]}

    async def async_get_chat_history(self, thread_id: str) -> List[BaseMessage]:
        """
        Retrieve the full chat history for a thread asynchronously.
        
        Args:
            thread_id: The conversation thread ID
            
        Returns:
            List of messages in the conversation history
        """
        try:
            # Get the current state from the checkpointer
            thread_id_key = {"configurable": {"thread_id": thread_id}}
            state = await self.app.checkpointer.aget(thread_id_key)
            
            # Extract messages from state
            if state and "messages" in state.get("channel_values", {}):
                messages = state["channel_values"]["messages"]
                return messages
            return []
        except Exception as e:
            logger.error(f"Error retrieving chat history for thread {thread_id}: {str(e)}")
            return []

    async def async_get_retrieved_context(self, thread_id: str) -> Optional[List[Document]]:
        """
        Async version of get_retrieved_context.
        Get the retrieved context for a chat thread asynchronously.
        
        Args:
            thread_id: The conversation thread ID
            
        Returns:
            List of retrieved documents if available, None otherwise
        """
        try:
            # Get the current state from the checkpointer
            thread_id_key = {"configurable": {"thread_id": thread_id}}
            state = await self.app.checkpointer.aget(thread_id_key)
            
            # Extract retrieved context from state
            if state and "retrieved_context" in state.get("channel_values", {}):
                retrieved_context = state["channel_values"]["retrieved_context"]
                return retrieved_context
            return None
        except Exception as e:
            logger.error(f"Error retrieving context for thread {thread_id}: {str(e)}")
            return None

    def reset_chat(self, thread_id: str) -> str:
        """
        Reset the chat history for a given thread.
        
        Args:
            thread_id: The conversation thread ID
            
        Returns:
            A new thread ID
        """
        # Create a new thread_id
        new_id = f"{thread_id}_new_{id(thread_id)}"
        logger.info(f"Reset chat: {thread_id} → {new_id}")
        return new_id

    def get_chat_history(self, thread_id) -> List[BaseMessage]:
        """
        Retrieve the full chat history for a thread.
        
        Args:
            thread_id: The conversation thread ID
            
        Returns:
            List of messages in the conversation history
        """
        # For AsyncPostgresSaver, the user needs to use the async version
        if isinstance(self.checkpointer, AsyncPostgresSaver) and self.is_async:
            logger.warning("For async checkpointer, use async_get_chat_history instead of get_chat_history")
            return []
        
        try:
            # Get the current state from the checkpointer
            thread_id_key = {"configurable": {"thread_id": thread_id}}
            if hasattr(self.app.checkpointer, "get") and callable(self.app.checkpointer.get):
                state = self.app.checkpointer.get(thread_id_key)
                if state and "messages" in state.get('channel_values', {}):
                    return state["channel_values"]["messages"]
            
            logger.info(f"No chat history found for thread {thread_id}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving chat history: {str(e)}")
            return []
        
    def get_retrieved_context(self, thread_id) -> Optional[List[Document]]:
        """
        Get the retrieved context for a chat thread.
        This is useful for debugging and frontend display.
        
        Args:
            thread_id: The conversation thread ID
            
        Returns:
            List of retrieved documents if available, None otherwise
        """
        # For AsyncPostgresSaver, the user needs to use the async version
        if isinstance(self.checkpointer, AsyncPostgresSaver) and self.is_async:
            logger.warning("For async checkpointer, use async_get_retrieved_context instead of get_retrieved_context")
            return None
        
        try:
            # Get the current state from the checkpointer
            thread_id_key = {"configurable": {"thread_id": thread_id}}
            if hasattr(self.app.checkpointer, "get") and callable(self.app.checkpointer.get):
                state = self.app.checkpointer.get(thread_id_key)
                if state and "retrieved_context" in state.get('channel_values', {}):
                    return state["channel_values"]["retrieved_context"]
            
            logger.info(f"No retrieved context found for thread {thread_id}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return None

    def get_original_message(self, thread_id) -> Optional[str]:
        """
        Get the original user message for debugging purposes.
        
        Args:
            thread_id: The conversation thread ID
            
        Returns:
            The original user message if available, None otherwise
        """
        return self._original_messages.get(thread_id)
    
    def _delete_thread(self, thread_id: str) -> None:
        """
        Delete the thread from the database.
        """
        self.app.checkpointer.delete_thread(thread_id)

    async def _adelete_thread(self, thread_id: str) -> None:
        await self.app.checkpointer.delete_thread(thread_id)

if __name__ == "__main__":
    import os
    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Check for API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set the OPENAI_API_KEY environment variable.")
        print("Example: export OPENAI_API_KEY='your-api-key'")
        import sys
        sys.exit(1)
        
    # Initialize the model
    model = ChatOpenAI(model="gpt-4.1")

    # Create the chatbot
    chatbot = ConfigurableChatbot(
        model=model,
        system_prompt="You are a helpful assistant that answers questions concisely.",
        max_history_tokens=1000,
        keep_indices=[0, 1],  # Always keep the system message (index 0),
    )
    
    # Example conversation
    thread_id = "example_thread"
    
    # First message
    print("\nUser: Hello, who are you?")
    response = chatbot.chat("Hello, who are you?", thread_id=thread_id)
    print(f"Assistant: {response[-1].content}")
    
    # Second message
    print("\nUser: What can you help me with?")
    response = chatbot.chat("What can you help me with?", thread_id=thread_id)
    print(f"Assistant: {response[-1].content}")
    
    # Example with streaming
    print("\nStreaming example:")
    print("User: Tell me a short joke")
    
    for chunk in chatbot.chat("Tell me a short joke", thread_id=thread_id, stream=True):
        if hasattr(chunk, "content"):
            print(chunk.content, end="")
    print()
