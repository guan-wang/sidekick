from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
#from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from typing import List, Any, Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum
from sidekick_tools import playwright_tools, other_tools
import uuid
import asyncio
from datetime import datetime

load_dotenv(override=True)


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool
    tutor_specialist_needed: bool
    tutor_specialist_output: Optional[Any]  # Store TutorSpecialistOutput structured output for worker to use


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if more input is needed from the user, or clarifications, or the assistant is stuck"
    )

class LanguageItemType(str, Enum):
    """Type of language learning item"""
    VOCAB = "vocab"
    GRAMMAR = "grammar"
    SENTENCE_PATTERN = "sentence_pattern"


class LanguageItem(BaseModel):
    """A language learning item (vocab, grammar point, or sentence pattern)"""
    type: LanguageItemType = Field(description="Type of language item: vocab, grammar, or sentence_pattern")
    korean: str = Field(description="The Korean original text of the language item")
    english: str = Field(description="English translation or explanation of the language item")
    context: Optional[str] = Field(
        default=None,
        description="Optional context or example sentence showing how this item is used"
    )


class Article(BaseModel):
    """A language learning article with Korean text and language items"""
    korean_text: str = Field(description="The Korean text paragraph from the article")
    english_translation: str = Field(description="English translation of the Korean text paragraph")
    language_items: List[LanguageItem] = Field(
        description="List of language learning items (vocab, grammar points, sentence patterns) extracted from the text"
    )
    date: Optional[str] = Field(
        default=None,
        description="Publication date or date associated with the article (format: YYYY-MM-DD)"
    )
    link: Optional[str] = Field(
        default=None,
        description="URL or link to the original article or source"
    )
    title: Optional[str] = Field(
        default=None,
        description="Title of the article"
    )
    source: Optional[str] = Field(
        default=None,
        description="Source or publication name where the article came from"
    )
    topic: Optional[str] = Field(
        default=None,
        description="Topic or category of the article"
    )


class TutorSpecialistOutput(BaseModel):
    """Structured output from the language tutor specialist containing formatted articles"""
    articles: List[Article] = Field(
        description="List of articles, each containing Korean text, English translation, and language learning items"
    )


class Sidekick:
    def __init__(self):
        self.worker_llm_with_tools = None
        self.evaluator_llm_with_output = None
        self.korean_tutor_specialist_llm_with_output = None
        self.tools = None
        self.llm_with_tools = None
        self.graph = None
        self.sidekick_id = str(uuid.uuid4())
        #self.memory = MemorySaver()
        self.browser = None
        self.playwright = None
        self.db_conn = None
        self.checkpointer = None

    async def setup(self):
        self.tools, self.browser, self.playwright = await playwright_tools()
        self.tools += await other_tools()
        worker_llm = ChatOpenAI(model="gpt-4o-mini")
        self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)
        evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
        self.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)
        korean_tutor_llm = ChatOpenAI(model="gpt-4o-mini")
        self.korean_tutor_specialist_llm_with_output = korean_tutor_llm.with_structured_output(TutorSpecialistOutput)
        await self.build_graph()

    def _is_korean_learning_request(self, state: State) -> bool:
        """
        Check if the user request is about Korean learning.
        Looks for Korean learning keywords in both English and Korean in user messages and success criteria.
        
        Returns:
            bool: True if this is a Korean learning request, False otherwise
        """
        korean_learning_keywords = ["korean", "learn korean", "korean news", "korean article", "korean language", 
                                    "한국", "한국어", "한국 뉴스", "한국 기사", "한국어 학습"]
        user_request = ""
        user_request_original = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_request += msg.content.lower() + " "
                user_request_original += msg.content + " "
        
        success_criteria_lower = state["success_criteria"].lower()
        success_criteria_original = state["success_criteria"]
        
        has_english_keyword = any(keyword in user_request or keyword in success_criteria_lower 
                                 for keyword in korean_learning_keywords[:5])
        has_korean_chars = any('\uac00' <= char <= '\ud7a3' for char in user_request_original + success_criteria_original)
        has_korean_keyword = any(keyword in user_request_original or keyword in success_criteria_original 
                                for keyword in korean_learning_keywords[5:])
        is_korean_learning = has_english_keyword or has_korean_chars or has_korean_keyword
        
        if is_korean_learning:
            print(f"[DEBUG] Korean learning detected! has_english_keyword={has_english_keyword}, "
                  f"has_korean_chars={has_korean_chars}, has_korean_keyword={has_korean_keyword}")
        
        return is_korean_learning

    def worker(self, state: State) -> Dict[str, Any]:
        # Check if this is a Korean learning request to customize instructions
        is_korean_learning = self._is_korean_learning_request(state)
        
        system_message = f"""You are a helpful assistant that can use tools to complete tasks.
    You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
    You have many tools to help you, including tools to browse the internet, navigating and retrieving web pages.
    Use the Google Places API tool to find places of interest in the user's given location.
    You have a tool to run python code, but note that you would need to include a print() statement if you wanted to receive output.
    You also have a new specialist agent that can transform Korean articles into a simpler, more understandable format for A2 level learners.
    When the Korean tutor specialist has processed articles, the structured output will be available in state["tutor_specialist_output"].
    Use the MongoDB tool to store and retrieve data from the database for custom data storage as per user request. For Korean learning, only save the structured output from the Korean tutor specialist.
    In order to save all the language learning items into the database, use exactly the format from state["tutor_specialist_output"] which contains the TutorSpecialistOutput structure with articles and language items.
    The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    This is the success criteria:
    {state["success_criteria"]}
    You should reply either with a question for the user about this assignment, or with your final response.
    """
        
        # Add Korean-specific instructions if this is a Korean learning request
        if is_korean_learning:
            system_message += """
    CRITICAL KOREAN CONTENT REQUIREMENTS:
    - When searching for Korean articles, news, or content, you MUST search using Korean language queries.
    - Use Korean search terms in your search queries (e.g., "한국 뉴스", "한국 기사", not "Korean news").
    - Navigate to Korean language websites (e.g., naver.com, daum.net, Korean news sites).
    - Retrieve content that is written in Korean language ONLY - do not use English translations or English-language articles.
    - If you find Korean content, the specialist agent will automatically process it for language learning.
    - Your search queries should be in Korean when looking for Korean content.
    """
    # user provided clarifications. Continue task
        if state.get("user_input_needed"):
            system_message += f"""
    Previously, you asked clarifying questions to the user about the request. The user has now replied with clarifications. 
    If the clarifications are adequate, you can then start to perform the task with these clarifications. 
    Otherwise you should ask further questions - be very specific. 
    """
        else:
            system_message += f"""
    DO NOT perform the task or provide any answer if the request is ambiguous. Ask three clarifying questions to the user unless if the request is crystal clear. 
    If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer.
    """
        if state.get("feedback_on_work"):
            system_message += f"""
    Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
    Here is the feedback on why this was rejected:
    {state["feedback_on_work"]}
    With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user."""

        # Add in the system message

        found_system_message = False
        messages = state["messages"]
        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                found_system_message = True

        if not found_system_message:
            messages = [SystemMessage(content=system_message)] + messages

        # Invoke the LLM with tools
        response = self.worker_llm_with_tools.invoke(messages)

        # Only trigger specialist if:
        # 1. It's a Korean learning request
        # 2. Korean text exists in tool results (ToolMessages) OR in worker response
        # 3. Tools have actually been executed (ToolMessage exists in message history)
        # 4. The specialist hasn't already processed content (to avoid infinite loop)
        if is_korean_learning:
            # Check if tools have been executed (ToolMessage exists in message history)
            # This ensures we only trigger specialist after tools have actually run and returned results
            tools_have_executed = any(isinstance(msg, ToolMessage) for msg in state["messages"])
            
            # Check for Korean text in ToolMessages (where tool results are stored)
            # Tools extract text from web pages, and Korean content will be in ToolMessage results
            has_korean_text_in_tools = False
            if tools_have_executed:
                for msg in state["messages"]:
                    if isinstance(msg, ToolMessage):
                        tool_content = msg.content if hasattr(msg, 'content') else str(msg)
                        if any('\uac00' <= char <= '\ud7a3' for char in tool_content):
                            has_korean_text_in_tools = True
                            break
            
            # Also check if the response contains Korean text (worker might have summarized it)
            response_content = response.content if hasattr(response, 'content') else str(response)
            has_korean_text_in_response = any('\uac00' <= char <= '\ud7a3' for char in response_content)
            
            # Korean text exists if it's in either tool results or worker response
            has_korean_text = has_korean_text_in_tools or has_korean_text_in_response
            
            # Check if specialist has already processed content - if so, don't trigger again
            specialist_already_processed = state.get("tutor_specialist_output") is not None
            
            # Check if the response has pending tool calls
            # If it does, we should process those first before triggering the specialist
            has_pending_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls
            
            # Trigger specialist if:
            # - Korean text exists in tool results or worker response (tools have executed and returned Korean content)
            # - Tools have actually been executed (ToolMessage exists)
            # - Specialist hasn't already processed content
            # - Response has NO pending tool calls (LLM is done with tools, ready for specialist)
            if has_korean_text and tools_have_executed and not specialist_already_processed and not has_pending_tool_calls:
                return {
                    "messages": [response],
                    "tutor_specialist_needed": True,
                }

        # Return updated state
        return {
            "messages": [response],
        }

    def worker_router(self, state: State) -> str:
        last_message = state["messages"][-1]

        # Check if Korean tutor specialist is needed
        if state.get("tutor_specialist_needed"):
            return "korean_tutor_specialist"

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        else:
            return "evaluator"
    

    def format_conversation(self, messages: List[Any]) -> str:
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                text = message.content or "[Tools use]"
                conversation += f"Assistant: {text}\n"
        return conversation

    def korean_tutor_specialist(self, state: State) -> Dict[str, Any]:
        """
        Korean tutor specialist agent that:
        1. Takes Korean article content from messages
        2. Simplifies text to A2 level
        3. Extracts key language items (vocab, grammar, sentence patterns)
        4. Formats output using TutorSpecialistOutput
        """
        # Extract Korean article content from messages
        # Look for Korean text in the messages (could be from user or worker)
        korean_articles_text = ""
        
        # Search through messages to find Korean content
        for message in reversed(state["messages"]):
            if isinstance(message, (HumanMessage, AIMessage)):
                content = message.content if hasattr(message, 'content') else str(message)
                # Check if message contains Korean characters
                if any('\uac00' <= char <= '\ud7a3' for char in content):
                    korean_articles_text = content
                    break
        
        # If no Korean text found, try to extract from the last message
        if not korean_articles_text:
            last_message = state["messages"][-1]
            if hasattr(last_message, 'content'):
                korean_articles_text = last_message.content
        
        system_message = f"""You are a Korean language tutor specialist. Your task is to:
1. Take Korean news articles and simplify them to A2 (elementary) learner level
2. Extract key language learning items: vocabulary words, grammar points, and sentence patterns
3. Format everything according to the TutorSpecialistOutput structure

INSTRUCTIONS:
- Simplify the Korean text to A2 level by:
  * Using simpler vocabulary where possible
  * Breaking complex sentences into shorter, clearer ones
  * Maintaining the core meaning and information
  * Keeping it natural and readable

- Extract language items:
  * VOCAB: Important vocabulary words that learners should learn
  * GRAMMAR: Grammar points or patterns used in the text
  * SENTENCE_PATTERN: Useful sentence structures or expressions

- For each language item, provide:
  * Korean original text
  * English translation/explanation
  * Optional context showing usage

- Extract metadata if available:
  * Date, link, title, source, topic from the original articles

Current date and time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        user_message = f"""Process the following Korean article(s) and format them for A2-level learners:

{korean_articles_text}

Please:
1. Simplify the Korean text to A2 level
2. Provide English translations
3. Extract and categorize all important language learning items
4. Include any available metadata (date, link, title, source, topic)

Format your response according to the TutorSpecialistOutput structure with articles containing simplified Korean text, translations, and language items.
"""

        tutor_messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]

        # Get structured output
        tutor_result = self.korean_tutor_specialist_llm_with_output.invoke(tutor_messages)
        
        # Format the output as a message for the worker
        articles_summary = f"[Korean Tutor Specialist] Processed {len(tutor_result.articles)} article(s).\n\n"
        for i, article in enumerate(tutor_result.articles, 1):
            articles_summary += f"Article {i}:\n"
            articles_summary += f"Korean (A2 level): {article.korean_text}\n"
            articles_summary += f"English: {article.english_translation}\n"
            articles_summary += f"Language items: {len(article.language_items)} items extracted\n"
            if article.title:
                articles_summary += f"Title: {article.title}\n"
            articles_summary += "\n"
        
        # Return updated state
        # Store the structured output in state for worker to use
        return {
            "messages": [AIMessage(content=articles_summary)],
            "tutor_specialist_needed": False,  # Clear the flag
            "tutor_specialist_output": tutor_result,  # Store structured output for database saving
        }

    def evaluator(self, state: State) -> State:
        # Handle case where last message might not have content (e.g., tool calls)
        last_message = state["messages"][-1]
        last_response = last_message.content if hasattr(last_message, 'content') and last_message.content else "[No text content - tool calls or empty message]"

        system_message = """You are an evaluator that determines if a task has been completed successfully by an Assistant.
    Assess the Assistant's last response based on the given criteria. Respond with your feedback, and with your decision on whether the success criteria has been met,
    and whether more input is needed from the user."""

        user_message = f"""You are evaluating a conversation between the User and Assistant. You decide what action to take based on the last response from the Assistant.

    The entire conversation with the assistant, with the user's original request and all replies, is:
    {self.format_conversation(state["messages"])}

    The success criteria for this assignment is:
    {state["success_criteria"]}

    And the final response from the Assistant that you are evaluating is:
    {last_response}

    Respond with your feedback, and decide if the success criteria is met by this response.
    Also, decide if more user input is required, either because the assistant has a question, needs clarification, or seems to be stuck and unable to answer without help.

    The Assistant has access to a tool to write files. If the Assistant says they have written a file, then you can assume they have done so.
    Overall you should give the Assistant the benefit of the doubt if they say they've done something. But you should reject if you feel that more work should go into this.

    """ 
        if state.get("feedback_on_work"):
            user_message += f"Also, note that in a prior attempt from the Assistant, you provided this feedback: {state['feedback_on_work']}\n"
            user_message += "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is required."

        evaluator_messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]

        eval_result = self.evaluator_llm_with_output.invoke(evaluator_messages)
        new_state = {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Evaluator Feedback on this answer: {eval_result.feedback}",
                }
            ],
            "feedback_on_work": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed,
        }
        return new_state

    def route_based_on_evaluation(self, state: State) -> str:
        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END"
        else:
            return "worker"

    async def build_graph(self):
        # Set up Graph Builder with State
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("korean_tutor_specialist", self.korean_tutor_specialist)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator)

        # Add edges
        graph_builder.add_conditional_edges(
            "worker", 
            self.worker_router, 
            {
                "korean_tutor_specialist": "korean_tutor_specialist",
                "tools": "tools", 
                "evaluator": "evaluator"
            }
        )
        graph_builder.add_edge("korean_tutor_specialist", "worker")  # Specialist returns to worker
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges(
            "evaluator", self.route_based_on_evaluation, {"worker": "worker", "END": END}
        )
        graph_builder.add_edge(START, "worker")

        # Compile the graph
        db_path = "memory_new.db"
        # Initialize AsyncSqliteSaver with aiosqlite connection
        # We'll create the connection and pass it to AsyncSqliteSaver
        # Note: aiosqlite connections need to be created in an async context
        self.db_conn = await aiosqlite.connect(db_path)
        self.checkpointer = AsyncSqliteSaver(self.db_conn)
        self.graph = graph_builder.compile(checkpointer=self.checkpointer)

    async def run_superstep(self, message, success_criteria, history):
        config = {"configurable": {"thread_id": self.sidekick_id}}

        state = {
            "messages": message,
            "success_criteria": success_criteria or "The answer should be clear and accurate",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "tutor_specialist_needed": False,
            "tutor_specialist_output": None,
        }
        result = await self.graph.ainvoke(state, config=config)
        user = {"role": "user", "content": message}
        reply = {"role": "assistant", "content": result["messages"][-2].content}
        feedback = {"role": "assistant", "content": result["messages"][-1].content}
        return history + [user, reply, feedback]

    def cleanup(self):
        if self.browser:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.browser.close())
                if self.playwright:
                    loop.create_task(self.playwright.stop())
            except RuntimeError:
                # If no loop is running, do a direct run
                asyncio.run(self.browser.close())
                if self.playwright:
                    asyncio.run(self.playwright.stop())
        if self.db_conn:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.db_conn.close())
            except RuntimeError:
                asyncio.run(self.db_conn.close())
