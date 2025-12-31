"""
Minimal Example: Adding a Specialized Agent to Your Existing Sidekick

This shows the minimal changes needed to add a specialized agent that:
- Takes input from the worker agent
- Processes it
- Returns results the worker can use
"""

# ============================================================================
# STEP 1: Extend State (add to your existing State class)
# ============================================================================

# In your State TypedDict, add:
# specialized_task: Optional[str]  # Task delegated to specialized agent
# specialized_result: Optional[str]  # Result from specialized agent


# ============================================================================
# STEP 2: Modify Worker to Delegate (update your worker method)
# ============================================================================

def worker_with_delegation(self, state: State) -> Dict[str, Any]:
    """Modified worker that can delegate to specialized agent"""
    system_message = f"""You are a helpful assistant that can use tools to complete tasks.
    You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
    
    This is the success criteria:
    {state["success_criteria"]}
    """
    
    # Check if specialized agent has results
    if state.get("specialized_result"):
        system_message += f"""
        
    The specialized agent has provided these results:
    {state["specialized_result"]}
    
    Use these results to complete your task.
        """
    
    # ... rest of your existing worker code ...
    
    messages = [SystemMessage(content=system_message)] + state["messages"]
    response = self.worker_llm_with_tools.invoke(messages)
    
    # Example: If worker needs specialized help, set the flag
    # In practice, you might use structured output or parse response
    update = {"messages": [response]}
    
    # Simple heuristic: if response mentions needing analysis/research
    if "need to analyze" in response.content.lower():
        update["specialized_task"] = "Please analyze: " + response.content
    
    return update


# ============================================================================
# STEP 3: Create Specialized Agent Method
# ============================================================================

def specialized_agent(self, state: State) -> Dict[str, Any]:
    """
    Specialized agent that processes tasks delegated by worker.
    
    Pattern:
    1. Read input from state["specialized_task"] or state["messages"]
    2. Process the task
    3. Return results via state["specialized_result"] AND messages
    """
    # Read the task
    task = state.get("specialized_task")
    if not task:
        # Fallback: get from last worker message
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            task = last_message.content
    
    system_message = f"""You are a specialized analysis agent.
    
    Your task: {task}
    
    Process this task thoroughly and provide detailed results.
    """
    
    # Get relevant context from messages
    messages = [SystemMessage(content=system_message)] + state["messages"]
    
    # Use a specialized LLM (could be same or different model)
    specialized_llm = ChatOpenAI(model="gpt-4o-mini")
    response = specialized_llm.invoke(messages)
    
    # Return results in two ways:
    # 1. As a message (for conversation flow)
    # 2. As structured data (for worker to use programmatically)
    return {
        "messages": [AIMessage(content=f"[Specialized Analysis] {response.content}")],
        "specialized_result": response.content,  # Worker can read this
        "specialized_task": None  # Clear the task flag
    }


# ============================================================================
# STEP 4: Update Router to Handle Specialized Agent
# ============================================================================

def worker_router_with_specialist(self, state: State) -> str:
    """Updated router that handles specialized agent routing"""
    last_message = state["messages"][-1]
    
    # If worker delegated a task, go to specialized agent
    if state.get("specialized_task"):
        return "specialized_agent"
    
    # If specialized agent has results, return to worker
    if state.get("specialized_result") and not state.get("specialized_task"):
        return "worker"
    
    # Check for tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise go to evaluator
    return "evaluator"


# ============================================================================
# STEP 5: Update Graph Building
# ============================================================================

async def build_graph_with_specialist(self):
    graph_builder = StateGraph(State)
    
    # Add nodes (including new specialized agent)
    graph_builder.add_node("worker", self.worker)
    graph_builder.add_node("specialized_agent", self.specialized_agent)  # NEW
    graph_builder.add_node("tools", ToolNode(tools=self.tools))
    graph_builder.add_node("evaluator", self.evaluator)
    
    # Add edges
    graph_builder.add_edge(START, "worker")
    
    # Updated conditional routing from worker
    graph_builder.add_conditional_edges(
        "worker",
        self.worker_router,  # Updated router
        {
            "specialized_agent": "specialized_agent",  # NEW route
            "tools": "tools",
            "evaluator": "evaluator",
            "worker": "worker"  # Loop back if results available
        }
    )
    
    # Specialized agent always returns to worker
    graph_builder.add_edge("specialized_agent", "worker")
    
    # Tools return to worker
    graph_builder.add_edge("tools", "worker")
    
    # Evaluator routes based on evaluation
    graph_builder.add_conditional_edges(
        "evaluator",
        self.route_based_on_evaluation,
        {"worker": "worker", "END": END}
    )
    
    # ... rest of your graph compilation ...


# ============================================================================
# SUMMARY: The Pattern
# ============================================================================
"""
COMMUNICATION PATTERN:

1. WORKER → SPECIALIZED AGENT:
   - Worker sets: state["specialized_task"] = "task description"
   - Router sees flag and routes to specialized_agent

2. SPECIALIZED AGENT PROCESSES:
   - Reads: state["specialized_task"] or state["messages"][-1]
   - Processes task
   - Returns: {
       "specialized_result": "results",
       "specialized_task": None  # Clear flag
     }

3. SPECIALIZED AGENT → WORKER:
   - Router sees specialized_result exists
   - Routes back to worker
   - Worker reads state["specialized_result"] and uses it

KEY POINTS:
- All communication via shared State object
- Use custom fields for structured data (specialized_task, specialized_result)
- Use messages for conversational flow
- Clear delegation flags after processing
- Router controls the flow based on state
"""

