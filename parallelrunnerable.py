from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda


# this  is use when the both chatpromptrtemplate value are same 

# 1. Prompt Template

short_prompt = ChatPromptTemplate.from_template(
"Explain {topic} in 1-2 lines"
)
detailed_prompt = ChatPromptTemplate.from_template(
"Explain {topic} in detail"

)

# 2. Model
model = ChatMistralAI(model="mistral-small-2506")

# 3. Output Parser
parser = StrOutputParser()


# Input
topic = "Machine Learning"

chain = RunnableParallel({
"short" :short_prompt | model | parser ,
"detailed" :detailed_prompt |model |parser
})

result = chain. invoke({"topic" :"Machine Learming"})

print(result ['short'])
print(result ['detailed' ])



# different value in chatprompttemplate 

short_prompt = ChatPromptTemplate.from_template(
"Explain {topic} in 1-2 lines"
)
detailed_prompt = ChatPromptTemplate.from_template(
"Explain {topic} in detail"
)

chain = RunnableParallel({
"short" :RunnableLambda(lambda x:x["short"])|short_prompt | model | parser ,
"detailed" :RunnableLambda(lambda x:x['detailed'])|detailed_prompt |model |parser
})

# lambda is use to remove the topic form the  dictornary
result=chain.invoke({
    "short":{"topic":"manchine learning"},
    "detailed":{"topic":"deep learning"}
})
