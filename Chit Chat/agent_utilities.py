from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentType, AgentType, load_tools, initialize_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
import matplotlib.pyplot as plt
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
import os
import streamlit as st
os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
os.environ['SERPAPI_API_KEY'] = st.secrets["SERPAPI_API_KEY"]
llm = ChatGoogleGenerativeAI(model='gemini-pro',api_key=st.secrets["GOOGLE_API_KEY"])

serch_tools = load_tools(['llm-math','serpapi'],llm=llm)


def create_agent(df):
    python_repl = PythonREPL()
    repl_tool = Tool(
        name="python_repl",
        globals={'df':df},
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. You can Modify the dataframe here with help of python commands.",
        func=python_repl.run,
    )

    llm.bind_tools([repl_tool]+serch_tools)

    pandas_df_agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            allow_dangerous_code=True,
            return_intermediate_steps=True,
            extra_tools=serch_tools,
    )
    
    return pandas_df_agent

def get_response(query,df):
    agent = create_agent(df)
    response = agent.invoke(query+" System Prompt -  1. Always update/reflect the changes done in orginal dataframe.\n")
    return response

def get_commands(response):
    commands = []
    for res in response['intermediate_steps']:
        if res[0].tool == 'python_repl' or res[0].tool == 'python_repl_ast':
            command = res[0].tool_input
            commands.append(command)
    
    return commands
    