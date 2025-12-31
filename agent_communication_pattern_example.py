"""
Example: Adding a Specialized Agent to LangGraph

This demonstrates the recommended patterns for agent-to-agent communication:
1. Shared State Pattern - All agents read/write to the same State
2. Message-based communication - Agents communicate via messages
3. Custom state fields - For structured data exchange
"""

from typing import Annotated, List, Any, Optional, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# ============================================================================
# PATTERN 1: Extend State with Custom Fields for Structured Data
# ============================================================================

class State(TypedDict):
    messages: Annotated[List[Any], add_messages]  # Message-based communication
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool
    
    # NEW: Custom field for specialized agent output
    specialized_result: Optional[Dict[str, Any]]  # Structured data from specialized agent
    task_for_specialist: Optional[str]  # Task delegated to specialized agent


# ============================================================================
# PATTERN 2: Worker Agent - Delegates to Specialized Agent
# ============================================================================

def worker(self, state: State) -> Dict[str, Any]:
    """
    Worker agent that can delegate tasks to a specialized agent.
    It communicates by:
    1. Adding messages to state["messages"]
    2. Setting state["task_for_specialist"] to trigger specialized agent
    """
    system_message = f"""You are a helpful assistant.
    
    You can delegate specialized tasks to a specialized agent by:
    - Setting the task_for_specialist field in your response
    - The specialized agent will process it and return results
    
    Success criteria: {state["success_criteria"]}
    """
    
    # Check if specialized agent has results
    if state.get("specialized_result"):
        system_message += f"""
        
    The specialized agent has completed its task. Results:
    {state["specialized_result"]}
    
    Use these results to complete your task.
        """
        # Clear the result so it's not reused
        specialized_result = state["specialized_result"]
    else:
        specialized_result = None
    
    messages = [SystemMessage(content=system_message)] + state["messages"]
    response = self.worker_llm.invoke(messages)
    
    # Worker decides if it needs specialized help
    # In practice, you might use structured output or parse the response
    update = {"messages": [response]}
    
    # Example: If worker mentions "analyze" or "research", delegate
    if "analyze" in response.content.lower() or "research" in response.content.lower():
        update["task_for_specialist"] = response.content
    
    return update


# ============================================================================
# PATTERN 3: Specialized Agent - Processes Task and Returns Results
# ============================================================================

def specialized_agent(self, state: State) -> Dict[str, Any]:
    """
    Specialized agent that:
    1. Reads input from state["task_for_specialist"] or state["messages"]
    2. Processes the task
    3. Returns results via state["specialized_result"] AND messages
    """
    # PATTERN: Read input from state
    task = state.get("task_for_specialist")
    if not task:
        # Fallback: Extract from last worker message
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            task = last_message.content
    
    system_message = f"""You are a specialized analysis agent.
    
    Your task: {task}
    
    Process this task and return structured results.
    """
    
    messages = [SystemMessage(content=system_message)] + state["messages"]
    response = self.specialized_llm.invoke(messages)
    
    # PATTERN: Return results in multiple ways:
    # 1. As a message (for conversation flow)
    # 2. As structured data (for programmatic access)
    return {
        "messages": [AIMessage(content=f"Specialized analysis complete: {response.content}")],
        "specialized_result": {
            "task": task,
            "analysis": response.content,
            "status": "complete"
        },
        "task_for_specialist": None  # Clear the task
    }


# ============================================================================
# PATTERN 4: Router Functions - Control Flow Between Agents
# ============================================================================

def worker_router(self, state: State) -> str:
    """
    Router decides: should worker delegate to specialized agent?
    """
    # Check if worker has delegated a task
    if state.get("task_for_specialist"):
        return "specialized_agent"
    
    # Check if specialized agent has results ready
    if state.get("specialized_result"):
        return "worker"  # Worker should process results
    
    # Check for tool calls
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "evaluator"


def specialized_router(self, state: State) -> str:
    """
    Router after specialized agent: return to worker
    """
    return "worker"


# ============================================================================
# PATTERN 5: Building the Graph with Multiple Agents
# ============================================================================

async def build_graph(self):
    graph_builder = StateGraph(State)
    
    # Add all agent nodes
    graph_builder.add_node("worker", self.worker)
    graph_builder.add_node("specialized_agent", self.specialized_agent)
    graph_builder.add_node("tools", ToolNode(tools=self.tools))
    graph_builder.add_node("evaluator", self.evaluator)
    
    # Add edges
    graph_builder.add_edge(START, "worker")
    
    # Conditional routing from worker
    graph_builder.add_conditional_edges(
        "worker",
        self.worker_router,
        {
            "specialized_agent": "specialized_agent",
            "tools": "tools",
            "evaluator": "evaluator",
            "worker": "worker"  # Loop back if results available
        }
    )
    
    # Specialized agent always returns to worker
    graph_builder.add_conditional_edges(
        "specialized_agent",
        self.specialized_router,
        {"worker": "worker"}
    )
    
    # Tools return to worker
    graph_builder.add_edge("tools", "worker")
    
    # Evaluator routes based on evaluation
    graph_builder.add_conditional_edges(
        "evaluator",
        self.route_based_on_evaluation,
        {"worker": "worker", "END": END}
    )
    
    self.graph = graph_builder.compile(checkpointer=self.checkpointer)


# ============================================================================
# ALTERNATIVE PATTERN: Pure Message-Based Communication
# ============================================================================

def specialized_agent_message_only(self, state: State) -> Dict[str, Any]:
    """
    Alternative: Pure message-based communication (no custom state fields)
    The specialized agent just adds messages, worker reads them.
    """
    # Read the last message from worker
    last_message = state["messages"][-1]
    
    system_message = """You are a specialized agent. Process the task in the last message."""
    
    messages = [SystemMessage(content=system_message)] + state["messages"]
    response = self.specialized_llm.invoke(messages)
    
    # Return only messages - worker will read from messages
    return {
        "messages": [
            AIMessage(content=f"[Specialized Agent] Task received and processed."),
            AIMessage(content=response.content)  # Results as message
        ]
    }


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
"""
1. SHARED STATE: All agents read/write to the same State object
   - This is the primary communication mechanism

2. MESSAGES: Use state["messages"] for conversational flow
   - Agents append messages, next agent reads them
   - Messages are automatically merged with add_messages reducer

3. CUSTOM FIELDS: Add fields to State for structured data
   - Useful for flags, intermediate results, task delegation
   - Example: task_for_specialist, specialized_result

4. ROUTERS: Use conditional edges to control flow
   - Check state fields to decide next agent
   - Can route based on messages, flags, or any state field

5. CLEARING FIELDS: Clear delegation flags after processing
   - Prevents infinite loops
   - Example: Set task_for_specialist = None after processing

6. READING INPUT: Specialized agent can read from:
   - Custom state fields (task_for_specialist)
   - Messages (state["messages"][-1])
   - Any other state field

7. RETURNING OUTPUT: Specialized agent can return via:
   - Messages (for conversation)
   - Custom state fields (for structured data)
   - Both (recommended for flexibility)
"""

