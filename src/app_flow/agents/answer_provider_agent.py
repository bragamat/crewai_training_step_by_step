from crewai import Agent
from app_flow.llm import asimov_llm


answer_agent = Agent(
    role="Answer Generator",
    goal="""
            Generate a comprehensive answer based on the user's message and
            any research conducted.
            """,
    backstory="""
            You are an expert in generating detailed and accurate answers to
            user queries.
            You utilize your extensive knowledge and research skills to provide
            well-informed responses.
            """,
    tools=[],
    llm=asimov_llm,
    verbose=True,
)

def generate_answer(state) -> str:
        prompt = f"""
        Based on the user's message and any research conducted, generate a 
        comprehensive answer.
        User Message: "{state.user_message}"

        research date: "{state.research_data}" 
        history: "{state.history}"
        """

        response = answer_agent.kickoff(prompt)

        return f"{response}"

