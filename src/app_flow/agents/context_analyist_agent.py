from crewai import Agent
from app_flow.llm import asimov_llm

context_analyist = Agent(
            role="Context Analyist",
            goal="""
            Analyze the user's message to determine if deep research is
            needed
            """,
            backstory="""
            You are an expert in analyzing user requests and determining the
            depth of research required.
            With your vast experience, you can quickly assess whether a  user's
            request is straightforward or requires in-depth investigation.
            """,
            tools=[],
            llm=asimov_llm,
            verbose=True,
        )

def check_for_research(state):
    prompt = f"""
    Analyze the following user message and determine if deep research is
    required to provide a comprehensive response.
    User Message: "{state.user_message}"
    History: {[mess for mess in state.history]}
    Respond with 'YES' if deep research is needed, otherwise respond with 'NO'.
    """

    response = context_analyist.kickoff(prompt)
    return response


