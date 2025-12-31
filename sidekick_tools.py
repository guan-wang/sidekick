from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from dotenv import load_dotenv
import os
import requests
from langchain.agents import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_google_community.places_api import GooglePlacesTool
from pymongo import MongoClient
import json
#from langchain_openai import OpenAI
#from langchain_core.agents import initializeAgentExecutorWithOptions

load_dotenv(override=True)
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"
serper = GoogleSerperAPIWrapper()

# MongoDB configuration
mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
mongodb_db_name = os.getenv("MONGODB_DB_NAME", "sidekick_data")
mongodb_collection_name = os.getenv("MONGODB_COLLECTION_NAME", "user_data")

# Initialize MongoDB connection (lazy)
_mongo_client = None
_mongo_collection = None

def get_mongo_collection():
    """Get or create MongoDB collection
    
    NOTE: MongoDB must be running before using these tools!
    - For local MongoDB: Install and start MongoDB service
    - For MongoDB Atlas: Set MONGODB_URI in .env file (requires pymongo[srv])
    """
    global _mongo_client, _mongo_collection
    if _mongo_client is None:
        try:
            # For Atlas connections, add recommended connection parameters if not present
            if "mongodb+srv://" in mongodb_uri and "retryWrites" not in mongodb_uri:
                # Add recommended parameters for Atlas if not already present
                separator = "&" if "?" in mongodb_uri else "?"
                uri_with_params = f"{mongodb_uri}{separator}retryWrites=true&w=majority"
            else:
                uri_with_params = mongodb_uri
            
            _mongo_client = MongoClient(uri_with_params, serverSelectionTimeoutMS=5000)
            # Test connection
            _mongo_client.admin.command('ping')
            db = _mongo_client[mongodb_db_name]
            _mongo_collection = db[mongodb_collection_name]
        except Exception as e:
            raise ConnectionError(
                f"MongoDB connection failed at {mongodb_uri.split('@')[-1] if '@' in mongodb_uri else mongodb_uri}. "
                f"\n\nTo use MongoDB tools:\n"
                f"1. For Atlas: Ensure MONGODB_URI is correct and you have pymongo[srv] installed\n"
                f"2. For local: Install and start MongoDB service\n"
                f"\nError: {str(e)}"
            )
    return _mongo_collection

async def playwright_tools():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools(), browser, playwright


def push(text: str):
    """Send a push notification to the user"""
    requests.post(pushover_url, data = {"token": pushover_token, "user": pushover_user, "message": text})
    return "success"


def get_file_tools():
    toolkit = FileManagementToolkit(root_dir="sandbox")
    return toolkit.get_tools()


async def other_tools():
    push_tool = Tool(name="send_push_notification", func=push, description="Use this tool when you want to send a push notification")
    file_tools = get_file_tools()

    tool_search =Tool(
        name="search",
        func=serper.run,
        description="Use this tool when you want to get the results of an online web search"
    )

    wikipedia = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)

    python_repl = PythonREPLTool()
    google_places = GooglePlacesTool()

    # MongoDB tools using pymongo
    def store_user_data(data_json: str) -> str:
        """Store user custom data in MongoDB. 
        
        REQUIRES: MongoDB must be running (local or Atlas).
        Input should be a JSON string with the data to store.
        Example: '{\"key\": \"preferences\", \"value\": \"dark theme\", \"user_id\": \"user123\"}'
        """
        try:
            data = json.loads(data_json)
            collection = get_mongo_collection()
            collection.insert_one(data)
            return f"Successfully stored data in MongoDB"
        except ConnectionError as e:
            return f"MongoDB connection error: {str(e)}"
        except json.JSONDecodeError as e:
            return f"Invalid JSON input: {str(e)}"
        except Exception as e:
            return f"Error storing data: {str(e)}"
    
    def retrieve_user_data(query_json: str) -> str:
        """Retrieve user custom data from MongoDB.
        
        REQUIRES: MongoDB must be running (local or Atlas).
        Input should be a JSON string with query criteria.
        Example: '{\"user_id\": \"user123\"}' or '{\"key\": \"preferences\"}'
        """
        try:
            query = json.loads(query_json)
            collection = get_mongo_collection()
            results = list(collection.find(query))
            # Remove MongoDB _id field for cleaner output
            for doc in results:
                doc.pop("_id", None)
            return json.dumps(results, indent=2, default=str)
        except ConnectionError as e:
            return f"MongoDB connection error: {str(e)}"
        except json.JSONDecodeError as e:
            return f"Invalid JSON query: {str(e)}"
        except Exception as e:
            return f"Error retrieving data: {str(e)}"
    
    mongo_store_tool = Tool(
        name="store_user_data",
        func=store_user_data,
        description="Store user custom data in MongoDB. Input must be a JSON string with the data to store. Example: '{\"key\": \"preferences\", \"value\": \"dark theme\", \"user_id\": \"user123\"}'"
    )
    
    mongo_retrieve_tool = Tool(
        name="retrieve_user_data",
        func=retrieve_user_data,
        description="Retrieve user custom data from MongoDB. Input must be a JSON string with query criteria. Example: '{\"user_id\": \"user123\"}' or '{\"key\": \"preferences\"}'"
    )

    return file_tools + [
        push_tool, 
        tool_search, 
        python_repl, 
        wiki_tool, 
        google_places,
        mongo_store_tool,
        mongo_retrieve_tool
    ]

