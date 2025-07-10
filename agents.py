from typing import Type
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from linkup import LinkupClient
from crewai import Agent, Task, Crew, Process
from langchain_core.tools import BaseTool
from langchain_community.llms import ollama

import os

load_dotenv()

api_key = os.getenv("LINKUP_API_KEY")

def get_llm_client():
    "Initialize and return the llm"
    return ollama(
        model="llama2",
        base_url="http://localhost:11434"
    )
class LinkUpSearchInput(BaseModel):
    """Schema for LinkUp Search tool validated using Pydantic"""
    query: str = Field(description="Perform search operation of user query")
    # default paramter can change in both depth and output_type
    depth: str = Field(default="standard", description="How deep the search is standard/deep")
    output_type: str = Field(default="searchResults", description="Output type: 'searchResults', 'sourcedAnswer', or 'structured'")

class LinkUpSearchTool(BaseTool):
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


def multiagent(query: str):
    """Create a multiagent that includes all sub agent and its tasks"""
    #Initialize tool
    linkup_search_tool = LinkUpSearchTool()

    #Get the llm client
    client = get_llm_client()

    web_search_agent = Agent(
        role="Web Search",
        goal="Search the most suitable and relevant information from the web along with documents or source links(urls)",
        backstory="You are an expert at formulating user query or search query and retrieving relevant information.Passes the result to 'Research Analyst' only.",
        allow_delegation=True,
        verbose=True,
        tools=[linkup_search_tool],
        llm=client
    )

    research_analyst_agent = Agent(
        role="Research Analyst",
        goal="Analyze and Synthesize the raw information into a structured insights along with source links(urls) as its citations.",
        backstory="An expert at analysing information, identifying the patterns and extracting the key insights.If required, can delegate the task of fact checking/verification to 'Web Searcher' only. Passes the final results to the 'Technical Writer' only.",
        verbose=True,
        allow_delegation=True,
        llm=client
    )

    technical_writer_agent = Agent(
        role="Technical Writer",
        goal="Create well-structured, clear and comprehensive responses in markdown format with citations/source links(urls).",
        backstory="An expert at communicating complex information in an accessible and easy way.",
        verbose=True,
        allow_delegation=False,
        llm=client
    )

    #Define Tasks for each of the above Agent
    search_task = Task(
        description=f"Search for comprehensive information about: {query}",
        agent=web_search_agent,
        expected_output="Detailed raw search results including source links(urls).",
        tools=linkup_search_tool
    )

    analysis_task = Task(
        description=f"Analyze the raw search results that we got from search task, identify key information, verify facts and prepare a structured analysis.",
        agent=research_analyst_agent,
        expected_output="A structured analysis of the information with verified facts and key insights along with source links(urls)",
        context=[search_task]
    )

    writer_task = Task(
        description="Create a comprehensive, well-organized response based on the research analysis output done by the analysis task.",
        agent=technical_writer_agent,
        expected_output="A clear, comprehensive response that directly answers the query with proper citations/source links (urls).",
        context=[analysis_task]
    )

    #Define the final Crew that integrates Agent and its Task
    crew = Crew(
        agents=[web_search_agent, research_analyst_agent, technical_writer_agent],
        tasks=[search_task, analysis_task, writer_task],
        verbose=True,
        process=Process.sequential
    )

    return crew

def run_research(query: str):
    """Run the whole research process and return the correct results"""
    try:
        crew = multiagent(query)
        result = crew.kickoff()
        return result.raw
    except Exception as e:
        return f"Error: {str(e)}"