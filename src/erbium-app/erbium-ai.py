from crewai import Agent, Task, Crew, Process
import os
from open_ai_config.openai_config import OPENAI_API_KEY

# import the OpenAI API key from the os environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Agent roles are defined and instantiated here
class AgentRoles:
    def __init__(self):
        self.researcher = Agent(
            role="Researcher",
            goal="Research new AI insights",
            backstory="You are a researcher at a university. You are working on a new AI algorithm that will help people with their daily lives.",
            verbose=True,
            allow_delegation=False,
        )

        self.writer = Agent(
            role="Writer",
            goal="Write a paper about the new AI algorithm",
            backstory="You are a writer at a university. You are writing a paper about a new AI algorithm that will help people with their daily lives.",
            verbose=True,
            allow_delegation=False,
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


class CrewAI:
    def __init__(self):
        self.crew = Crew(
            name="AI Research Crew",
            agents=[AgentRoles().researcher, AgentRoles().writer],
            tasks=[Tasks().task1, Tasks().task2],
            verbose=2,
            process=Process.sequential,
        )

    def kickoff(self):
        return self.crew.kickoff()


if __name__ == "__main__":
    crew = CrewAI()
    result = crew.kickoff()
    print(result)
