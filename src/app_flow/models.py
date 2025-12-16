
from typing import List, Literal
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: Literal["user", "assistant"] = Field(default="user", description="Role of the message sender")
    content: str = Field(..., description="Content of the message")

class DeepResearchState(BaseModel):
    history: List[Message] = []
    user_message: str = ""
    research_needed: bool = False
    research_data: str = ""
    report: str = ""
    followup_questions: List[str] = []
