#!/usr/bin/env python
from crewai import LLM, Agent
from crewai.flow import Flow, listen, or_, start, router
from crewai.flow.persistence import persist

from app_flow.agents.report_analysist_agent import build_report_analysis
from app_flow.agents.answer_provider_agent import generate_answer
from app_flow.agents.context_analyist_agent import check_for_research
from app_flow.agents.researcher_agent import run_research
# from app_flow.crews.deep_research_crew.deep_research_crew import DeepResearchCrew
from app_flow.models import DeepResearchState, Message

from app_flow.llm import asimov_llm

@persist()
class DeepResearchFlow(Flow[DeepResearchState]):
    llm: LLM = asimov_llm

    @start()
    def get_user_message(self):
        print(f"State: {self.state}")
        self.state.history.append(Message(content=self.state.user_message, role="user"))
        response = check_for_research(self.state)
        print(f"Context Analyist Response: {response}")

        self.state.research_needed = f"{response}" == 'YES'

    @router(get_user_message)
    def check_for_deep_research(self):
        
        print(f"Check for deep research: {self.state.research_needed}")
        if self.state.research_needed:
            return 'DO_DEEP_RESEARCH'  # or 'SKIP_DEEP_RESEARCH' based on some condition

        return 'SKIP_DEEP_RESEARCH'

    @router('DO_DEEP_RESEARCH')
    def clarify(self):
        clarifier_agent = Agent(
            role="Clarifier Specialist",
            goal="""
                    Your task is to ask the user follow-up questions to clarify their request for better research results
                    """,
            backstory="""
                    You are a highly skilled clarifier specializing in asking precise and relevant follow-up questions.
                    You excel at understanding user needs and ensuring that research is well-targeted.
                    """,
            tools=[],
            reasoning=True,
            llm=asimov_llm,
            verbose=True,
        )

        prompt = """
        Come up with possible follow up questions to desambiguate the user's
        request for better research results.
        Only return the follow up questions if necessary. If no follow up questions are necessary,
        respond with "No follow up questions needed".
        """

        prompt += f"""
        User Message: "{self.state.user_message}"
        history: "{self.state.history}"
        """
        response = clarifier_agent.kickoff(prompt)

        if f"{response}" == "No follow up questions needed":
            return "TRIGGER_RESEARCH"

        follow_up_questions = [q.strip() for q in response.raw.split('\n') if q.strip()]
        self.state.followup_questions = follow_up_questions

        return 'FOLLOW_UP_QUESTIONS'

    @listen('TRIGGER_RESEARCH')
    def deep_research(self):
        result = run_research(self.state)
        self.state.research_data = result.raw

        print(f"result: {result}")

    @listen(deep_research)
    def generate_report(self):
        report = build_report_analysis(self.state)

        self.state.report = report.raw

    @listen(or_('FOLLOW_UP_QUESTIONS', 'SKIP_DEEP_RESEARCH', generate_report))
    def answer(self):
        response = generate_answer(self.state)

        print(f"Answer Generator Response: {response}")
        print(f"State: {self.state}")

        if self.state.followup_questions:
            for q in self.state.followup_questions:
                self.state.history.append(Message(content=q, role="assistant"))

            return self.state.followup_questions
        else:
          self.state.history.append(Message(content=response, role="assistant"))

        return response
        

def kickoff():
    poem_flow = DeepResearchFlow()
    answer = poem_flow.kickoff(inputs={
        "id": "d4bb078a-d979-4ba0-a990-2e6b7b3d54a4",
        "user_message": """
         - in germany; yes 2025;
         - Cultural aspects;
         - personal experiences;
         - econimic impact related;
        """
    })

    print(f"Final Answer: {answer}")


def plot():
    poem_flow = DeepResearchFlow()
    poem_flow.plot()
