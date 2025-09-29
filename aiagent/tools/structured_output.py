from openai import OpenAI
from pydantic import BaseModel


class StructuredOutput(BaseModel):
    name: str
    date: str
    participants: list[str]