from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, after_kickoff, agent, before_kickoff, crew, task
from crewai_tools import SerperDevTool
from pydantic import BaseModel

from crewai_trainig_step_by_step.tools.serper_scraper_tool import SerperScrapeTool
from eval import EvalListenerSetup

eval_listener = EvalListenerSetup()


class SummarizationOutput(BaseModel):
    topic: str
    summary: str
    key_points: List[str]


@CrewBase
class CrewaiTrainigStepByStep:
    """CrewaiTrainigStepByStep crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @before_kickoff
    def prepare_inputs(self, inputs):
        # Preprocess or modify inputs, that will be accessible in the 'inputs'
        # parameter of the crew. In our case, we have two inputs: topic and current_year.
        # Let's modify slightly the topic, by overriding it with an hardcoded option.
        # This method is often used to fetch external data such as s3 or any
        # other cloud bucket.

        inputs["topic"] = "Agentic AI Stacks: Why CrewAI is the best option."

        return inputs

    @after_kickoff
    def log_results(self, result):
        # This method is often used to save the results to a file or any other storage.
        # More in general, it is used to perform any action after the crew has finished executing.
        # In this case, we are logging the results to the console.
        print("Crew execution completed with result:", result.json)
        return result.json

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],  # type: ignore[index]
            verbose=True,
            tools=[SerperScrapeTool(), SerperDevTool()],
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["reporting_analyst"],  # type: ignore[index]
            verbose=True,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config["reporting_task"],  # type: ignore[index]
            output_file="report.md",
        )

    @task
    def summarization_task(self) -> Task:
        return Task(
            config=self.tasks_config["summarization_task"],  # type: ignore[index]
            output_json=SummarizationOutput,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CrewaiTrainigStepByStep crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
