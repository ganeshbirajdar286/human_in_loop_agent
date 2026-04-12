from dotenv import load_dotenv
load_dotenv()
from langchain_tavily import TavilySearch
from langchain_mistralai import  ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

search_tool = TavilySearch(max_results=5)
llm = ChatMistralAI(model="mistral-small-2506")

prompt =ChatPromptTemplate.from_template(
    """
you are helpfull assistant
summarize the following news into clear bullet points{news}
    """
)

chain =prompt|llm|StrOutputParser()
news_result =search_tool.invoke("latest ai news of 2026")

result=chain.invoke({"news":news_result})

print(result)