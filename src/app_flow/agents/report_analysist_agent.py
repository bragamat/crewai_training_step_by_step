
from crewai.agent import Agent
from app_flow.llm import asimov_llm

report_analysist = Agent(
    role="Report Analysis Agent",
    goal="""
            Analyze the research data provided and generate a comprehensive report
            based on the findings.
            """,
    backstory="""
            You are an expert in analyzing research data and compiling detailed
            reports.
            You utilize your analytical skills and knowledge to provide
            well-structured and informative reports.
            """,
    tools=[],
    llm=asimov_llm,
    verbose=True,

)


def build_report_analysis(state):
    prompt =  """You are a report analysis agent.
    Your task is to analyze the research data provided and generate
    a comprehensive report based on the findings and the user message.
    """

    prompt += f"""
    User Message: "{state.user_message}"
    Research Data: "{state.research_data}"
    """
    response = report_analysist.kickoff(prompt)

    return response


