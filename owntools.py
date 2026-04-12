# tools are also runnerable.so we can use invoke

from langchain.tools import tool

@tool
def get_greeting(name:str)->str:
    """Generate a greeting message for a user""" # this give  descriptions about the function
    return f"hello {name} welcome to ai school"

result =get_greeting.invoke({"name":"Ganesh"})
print(result)

print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)