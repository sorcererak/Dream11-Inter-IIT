from langchain_google_genai import ChatGoogleGenerativeAI
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
import streamlit as st
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langgraph.prebuilt import ToolNode
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from app.chatbot.tools import web_search_tool,user_manual_tool
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from dotenv import load_dotenv

load_dotenv()

template = """You are a chatbot designed to give helpful, descriptive and to the point answers to user queries.Respond to the user in a friendly and helpful manner, and use the information provided in the chat history to generate your responses.
JUST ANSWER THE QUESTION DO NOT INCLUDE ANYTHING ELSE

RETURN A SHORT PARAGRAH ANSWERING THE QUESTION, IT SHOULD BE IN SIMPLE TEXT NO MD

Following are the Guidelines:
1. Do not provide any medical, legal, or financial advice.
2. If the context does not provide the necessary information , respond according to your knowledge.


Chat History:
{chat_history}


YOUR TASK IS TO ANSWER THE BELOW QUESTION
{human_input}

Chatbot:"""

Guidelines = """
Following guidelines must be strictly followed:
All sports queries will be cricket related.
ALWAYS CALL THE web_search_tool .
Any query involving the word dream should be sent to the user_manual_tool.
FOR ANY SPORTS RELATED QUERY USE web_search_tool.DO NOT HALUCINATE ANYTHING
0. FOR NON-SPORTS RELATED QUERIES call the user_manual_tool.
1. For sports-related queries, use web_search_tool to find relevant information, use the response by the tools to generate a helpful and personalized response.
2. Pay attention to the user's query and provide a response that is relevant and helpful, but do not provide response to inappropriate or offensive queries.



Question to answer:
"""
parser = JsonOutputToolsParser(return_id=True)
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input","context"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")
tools = [web_search_tool,user_manual_tool]

#! https://ai.google.dev/gemini-api/docs/api-key --> Use this link to get google api


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_llm_with_tools = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=GEMINI_API_KEY).bind_tools(tools=tools)

gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=GEMINI_API_KEY)


# llm = OpenAI()
llm_chain = LLMChain(
    llm= gemini_llm,
    prompt=prompt,
    verbose=False,
    memory=memory,
)

tool_node = ToolNode(tools=tools)

def tool_selector(query):
    # Invoke the tools
    tools_called = gemini_llm_with_tools.invoke(Guidelines+query)
    # tools_called = "web_search_tool"
    # Parse the tool responses
    print("\n\nTools Called:",tools_called)
    parsed = parser.invoke(tools_called)
    print("\n\nParsed:",parsed)
    # Get the tool responses
    tool_responses = [
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": r["type"], "args": r["args"], "id": r["id"]}
                        ],
                    )
                ]
            }
        )
        for r in parsed
    ]
    
    print(tool_responses)
    # Generate the final response using llm_chain
    response = llm_chain.invoke(query + "\n\n" + "Context:" + str(tool_responses))
    # print(response)
    return response['text']



# while True:
#     query = input("Human Input:")
#     final_response = tool_selector(query)
#     print("\n\n\n\n\n\n\n",final_response)

