# This is a Human-in-the-loop not agent


from dotenv import load_dotenv
from langchain_mistralai import  ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import  tool
from langchain_core.messages import HumanMessage
from rich import print

load_dotenv()

#1 creating a tool
@tool
def get_text_length(text:str)->int:
    """ Return the number of character in a given text  """
    return len(text)


tools={
    "get_text_length":get_text_length
}

llm=ChatMistralAI(
   model="mistral-small-2506"
)
#2 tool binding
llm_with_tool=llm.bind_tools([get_text_length])
messages=[]
prompt=input("YOU:")
query=HumanMessage(prompt)
messages.append(query)

result=llm_with_tool.invoke(messages)
messages.append(result)



if result.tool_calls:
    tool_name=result.tool_calls[0]["name"]
    tool_message=tools[tool_name].invoke(result.tool_calls[0])
    messages.append(tool_message)

result=llm_with_tool.invoke(messages)
print(result.content)
    
