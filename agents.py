from typing import Type
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from linkup import LinkupClient
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

import os

load_dotenv()

api_key = os.getenv("LINKUP_API_KEY")

class LinkUpSearchInput(BaseModel):
    """Schema for LinkUp Search tool validated using Pydantic"""
    query: str = Field(description="Perform search operation of user query")
    # default paramter can change in both depth and output_type
    depth: str = Field(default="standard", description="How deep the search is standard/deep")
    output_type: str = Field(default="searchResults", description="Output type: 'searchResults', 'sourcedAnswer', or 'structured'")

class LinkUpSearchTook(BaseTool):
    name: str = "LinkUp Search"
    description: str = "Search the web for the information and return accurate results"
    args_schema: Type[BaseModel] = LinkUpSearchInput

    def __init__(self):
        super().__init__()

    def _run(self, query: str, depth: str = "standard", output_type: str = "searchResults")-> str:
        """Execute Linkup Search and return results"""
        try:
            #Initialize Linkup client with its API key
            linkup_client = LinkupClient(api_key=api_key)

            #Perform Search
            search_response = linkup_client.search(
                query=query,
                depth=depth,
                output_type=output_type
            )
            return str(search_response)
        except Exception as e:
            return f"Error While Search using Linkup: {str(e)}"


def get_llm_client():
    "Initialize and return the llm"
    return LLM(
        model="ollama/deepseek-r1:7b",
        base_url="http://localhost:6060"
    )

def research_agent(ok):
    return "ok"

def analysis_agent(ok):
    return "ok"

def content_writer_agent(ok):
    return "ok"