from crewai import Agent, Task, Crew, Process
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API")

researcher = Agent(
    role="Researcher",
    goal="Research new AI insights",
    backstory="You are a researcher at a university. You are working on a new AI algorithm that will help people with their daily lives.",
    verbose=True,
    allow_delegation=False,
)


writer = Agent(
    role="Writer",
    goal="Write a paper about the new AI algorithm",
    backstory="You are a writer at a university. You are writing a paper about a new AI algorithm that will help people with their daily lives.",
    verbose=True,
    allow_delegation=False,
)


task1 = Task(description="Investigate the latest medical research", agent=researcher)
task2 = Task(description="Investigate the latest AI research", agent=writer)

crew = Crew(
    name="AI Research Crew",
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2,
    process=Process.sequential,
)

result = crew.kickoff()

print(result)
