from dotenv import load_dotenv
import os
import requests
from rich import print
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain.agents import create_agent
from tavily import TavilyClient
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, wrap_tool_call
from langchain_core.messages import  AIMessage,SystemMessage,HumanMessage,ToolMessage

load_dotenv()

@tool
def get_wheather(city:str)->str: 
    """ get current weather of a city"""
    api_key=os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    response =requests.get(url)
    data=response.json()
    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"
    temp=data["main"]["temp"]
    desc=data["weather"][0]["description"]

    return f"wheather in {city}:{desc},{temp}°c"

#tavily new tool

tavily=TavilyClient( api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city:str)->str:
    """get latest news aboout the city"""
    response = tavily.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])
    if not results:
        return f"No news found for {city}"
    
    news_list = []
    
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        
        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )
    
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


llm=ChatMistralAI(model="mistral-small-2506")

#middleware for human in the loop approval before tool call
@wrap_tool_call
def  human_approval(request,handler):
    """Ask for human approval before calling a tool"""
    tool_name=request.tool_call["name"]
    confirm=input(f"Agent to call tool {tool_name} . Approve? (y/n): ")
    if confirm.lower()!="y":
        return ToolMessage(content="tool call denied by user",tool_call_id=request.tool_call["id"])
    return  handler(request)


agent=create_agent(
      llm,
      tools=[get_wheather,get_news],
      system_prompt="you are a helpful assistant that provides weather and news information about cities in india",
      middleware=[human_approval]
  )

print("City Agent | type exit to quit")

while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break 
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    print("bot : ", result['messages'][-1].content )

