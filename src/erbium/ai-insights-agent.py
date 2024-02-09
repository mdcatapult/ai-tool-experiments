from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from config.config import OPENAI_API_KEY
import os

# import the OpenAI API key from the os environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# instantiate the Ollama language model , you can use different models
ollama_llm = Ollama(model="llama2")

# using the DuckDuckGoSearchRun tool
search_tool = DuckDuckGoSearchRun()


# Agent roles are defined and instantiated here
class AgentRoles:
    def __init__(self):
        self.researcher = Agent(
            role="Researcher",
            goal="Research new AI insights",
            backstory="You are a researcher at a university. You are working on a new AI algorithm that will help people with their daily lives.",
            verbose=True,
            allow_delegation=False,
            tools=[search_tool],
            llm=ollama_llm,
        )

        self.writer = Agent(
            role="Writer",
            goal="Write a paper about the new AI algorithm",
            backstory="You are a writer at a university. You are writing a paper about a new AI algorithm that will help people with their daily lives.",
            verbose=True,
            allow_delegation=False,
            llm=ollama_llm,
        )


# Tasks are defined here
class Tasks:
    def __init__(self):
        self.task1 = Task(
            description="Investigate the latest medical research",
            agent=AgentRoles().researcher,
        )

        self.task2 = Task(
            description="Investigate the latest AI research", agent=AgentRoles().writer
        )


# Instructor
#  Crew is used to define the agents and tasks that will be used in the simulation
class CrewAI:
    def __init__(self):
        self.crew = Crew(
            name="AI Research Crew",
            agents=[AgentRoles().researcher, AgentRoles().writer],
            tasks=[Tasks().task1, Tasks().task2],
            verbose=2,
            process=Process.sequential,
        )

    def start_simulation(self):
        return self.crew.kickoff()


if __name__ == "__main__":
    crew = CrewAI()
    result = crew.start_simulation()
    print(result)
