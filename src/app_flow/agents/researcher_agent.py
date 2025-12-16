from crewai.agent import Agent
from app_flow.llm import asimov_llm

researcher = Agent(
    role="Researcher Specialist",
    goal="""
            Analyze the user message and perform web research
            """,
    backstory="""
            You are a highly skilled researcher specializing in gathering
            information from various sources.
            You excel at conducting in-depth research and providing
            well-structured and defined topics.
            """,
    tools=[],
    llm=asimov_llm,
    verbose=True,

)


def run_research(state):
    prompt =  """
    You are a researcher Agent
    Your task is to analyze the user message and perform well structure and 
    very well defined topics search.
    """

    prompt += f"""
    User Message: "{state.user_message}"
    """
    response = researcher.kickoff(prompt)

    return response


