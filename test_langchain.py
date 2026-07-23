import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(
    model="anthropic/claude-sonnet-5",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

prompt = ChatPromptTemplate.from_template(
    "Rezumă acest text în maxim {length} cuvinte: {content}"
)

chain = prompt | model | StrOutputParser()

result = chain.invoke(
    {
        "content": "BMW a prezentat noua generație a modelului electric iX3, cu autonomie extinsă și tehnologie de încărcare rapidă.",
        "length": 10,
    }
)
print(result)
