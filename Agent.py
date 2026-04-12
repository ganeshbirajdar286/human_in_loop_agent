from dotenv import load_dotenv
import os
import requests
from rich import print
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import  AIMessage,SystemMessage,HumanMessage,ToolMessage
from tavily import TavilyClient

load_dotenv()



@tool
def get_wheather(city:str)->str: 
    """ get current weather of a city"""
    api_key=os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    response =requests.get(url)
    data=response.json()

    print("debug",data)

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

tools={
    "get_news":get_news,
    "get_wheather":get_wheather,
}

llm_with_tools=llm.bind_tools([get_news,get_wheather])

#Agent loop---very important 
messages=[]

print("City intelligence  system ")
print("type Exit to quit")

while True:
    user_input=input("You:")
    if  user_input.lower()=="exit":
        break;
    messages.append(HumanMessage(content=user_input))
    while True:
        result=llm_with_tools.invoke(messages)
        messages.append(result)


      #if tool is required
        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name=tool_call['name']
                
                #human in the loop
                confirm=input(f"Agent wants to call tool: {tool_name} . Do you want to proceed? (yes/no): ")

                if confirm.lower() == "no":
                    print("Tool call cancelled by user.")
                    break;

                #execute the tool
                tool_result=tools[tool_name].invoke(tool_call)

                messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call['id']
                    ))
            continue
        else:
            print(f"Agent: {result.content}")
            break;


