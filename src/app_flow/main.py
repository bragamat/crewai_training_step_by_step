#!/usr/bin/env python
from crewai import LLM
from crewai.flow import Flow, listen, or_, start, router
from crewai.flow.persistence import persist

from app_flow.agents.answer_provider_agent import generate_answer
from app_flow.agents.context_analyist_agent import check_for_research
from app_flow.models import DeepResearchState, Message

from app_flow.llm import asimov_llm

@persist()
class DeepResearchFlow(Flow[DeepResearchState]):
    llm: LLM = asimov_llm

    @start()
    def get_user_message(self):
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

    @listen('DO_DEEP_RESEARCH')
    def clarify(self):
        print('Running clarify step...')

    @listen(clarify)
    def deep_research(self):
        pass

    @listen(deep_research)
    def generate_report(self):
        pass

    @listen(or_('SKIP_DEEP_RESEARCH', generate_report))
    def answer(self):
        response = generate_answer(self.state)

        print(f"Answer Generator Response: {response}")
        print(f"State: {self.state}")
        self.state.history.append(Message(content=response, role="assistant"))

        return response
        

def kickoff():
    poem_flow = DeepResearchFlow()
    poem_flow.kickoff(inputs={
        "id": "d4bb078a-d979-4ba0-a990-2e6b7b3d54a4",
        "user_message": "Quem e serjao dos foguetes?",
        "history": [],
        "research_needed": False,
        "research_data": "",
        "with_plot": True
    })


def plot():
    poem_flow = DeepResearchFlow()
    poem_flow.plot()
