#  Copyright 2024 Medicines Discovery Catapult
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os

from crewai import Agent, Task, Crew
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI

from src.config.config import OPENAI_API_KEY

# import the OpenAI API key from the os environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# instantiate the Ollama language model , you can use different models
# llm = Ollama(model="llama2", verbose=True, temperature=0.6)
llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
# using the DuckDuckGoSearchRun tool
search_tool = DuckDuckGoSearchRun()


# Agent roles are defined and instantiated here
class AgentRoles:
    def __init__(self):
        self.researcher = Agent(
            role="Senior Drug Discovery scientist",
            goal="Uncover cutting edge advancements in medicines discovery technology and drugs",
            backstory="""You work at a major pharmaceutical drug discovery company.
            Your expertise lies in identifying how existing and new technologies
            can be applied in the field of medicines discovery.""",
            verbose=True,
            allow_delegation=False,
            tools=[search_tool],
            llm=llm,
        )

        self.writer = Agent(
            role="Writer",
            goal="Write compelling and succinct summaries of the latest advancements in medicines discovery.",
            backstory="""You are a world famous writer with a passion for science and technology.
            You have been asked to write a blog post on the latest advancements in medicines discovery. 
            You translate complex scientific concepts into easy to understand language.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )


# Tasks are defined here
class Tasks:
    def __init__(self):
        self.task1 = Task(
            description="""Conduct a comprehensive review of the latest medical research over the last 2 months. Identify
            breakthrough technologies and new drugs that are being developed.""",
            agent=AgentRoles().researcher
        )

        self.task2 = Task(
            description="""Using any insights provided, develop an engaging blog post style summary that
            highlights current advancements in the field of medicines discovery. Avoid words that are too complex and provide layman
            descriptions of any new technologies involved. Your final answer should contain at least 1 paragraph per technology.""",
            agent=AgentRoles().writer
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
            # process=Process.sequential,
        )

    def start_simulation(self):
        return self.crew.kickoff()


if __name__ == "__main__":
    crew = CrewAI()
    result = crew.start_simulation()
    print(result)
