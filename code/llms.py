import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv

load_dotenv()


available_models = [
    "gpt-4o-mini",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "llama-3.1-8b-instant",
    "phi3:mini"
]


def get_llm(model: str):
    if model not in available_models:
        raise ValueError(f"Invalid model. Available models: {available_models.keys()}")

    if model == "gpt-4o-mini":
        return ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif model in ["gemini-1.5-flash", "gemini-1.5-pro"]:
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

    elif model == "llama-3.1-8b-instant":
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    elif model == "phi3:mini":
        return ChatOllama(
            model="phi3:mini",
            temperature=0.0,
        )
