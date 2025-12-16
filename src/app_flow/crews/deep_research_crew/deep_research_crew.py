from crewai import Agent, Crew, Process, Task 
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool

from typing import List

from app_flow.llm import asimov_llm, embedding_model

@CrewBase
class DeepResearchCrew():
    """DeepResearchCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            llm=asimov_llm,
            verbose=True,
            tools=[SerperDevTool()],
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
            llm=asimov_llm,
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the DeepResearchCrew crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            embedder=embedding_model
        )

if __name__ == "__main__":
    crew_instance = DeepResearchCrew()
    deep_research_crew = crew_instance.crew()
    deep_research_crew.kickoff(
        inputs={
            "research_task": {
                "topic": "The impact of climate change on global agriculture",
                "depth": "in-depth"
            },
            "reporting_task": {
                "format": "detailed report",
                "audience": "policy makers"
            }
        }
    )
