import pandas as pd
import os
import streamlit as st
from agent_utilities import get_response, get_commands
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage,AIMessage,SystemMessage
import google.generativeai as genai

os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
os.environ['SERPAPI_API_KEY'] = st.secrets["SERPAPI_API_KEY"]

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
st.set_page_config(layout='wide')

def fine_tune_output(question,reponse):
    model=genai.GenerativeModel("gemini-pro")
    chat = model.start_chat()
    send_question = f"""
        Human Query = {question}
        Response by agent = {reponse}
        Task = 
        If the Generated reponse is correct then, Just give the correct reponse with some formatting and Nothing else.
        Else
        Generate a message saying relavent info to query in funnier way and nothing else.
    """
    response=chat.send_message(send_question,stream=True)
    generated_text = ""
    for chunk in response:
        generated_text += chunk.text
    
    return generated_text

def get_stream_lit_commands(commands):
    llm = ChatGoogleGenerativeAI(model='gemini-pro',api_key=st.secrets["GOOGLE_API_KEY"])
    reply = llm([
        HumanMessage("\n".join(commands)+"\n For the above code generate the stream_lit code to plot the graph in ONE LINE.\n Give me only python code(without any formatting in plain text) and nothing else.\n Output format - \n PYTHON CODE ")
    ])
    return reply.content

def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

st.header('Welcome to Chit Chat Frame')
file = st.file_uploader("Upload the csv file",type='csv')


if file:
    df = pd.read_csv(file)
    prompt = st.chat_input("Say something")
    with st.chat_message("user"):
            st.dataframe(df)
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Generating response"):
                try:
                    response = get_response(prompt,df)
                    commands = get_commands(response)
                    if "PLOT" not in str(prompt).upper() or "GRAPH" not in str(prompt).upper():
                        fine_ouput = fine_tune_output(str(prompt),str(response['output']))
                        st.write_stream(response_generator(fine_ouput))
                except Exception as e:
                    st.write("Model is currently Overloaded.. Try after some time")
                    print(e)
                try:
                    if "PLOT" in str(prompt).upper() or "GRAPH" in str(prompt).upper():
                        print("Executed")
                        stream_lit_commands = get_stream_lit_commands(commands)
                        print(stream_lit_commands)
                        exec(stream_lit_commands)
                except Exception as e:
                    st.write("Model Overloaded.. Try after some time")
                    try:
                        stream_lit_commands = get_stream_lit_commands(commands)
                        print(stream_lit_commands)
                        exec(stream_lit_commands)
                    except Exception as e:
                        try:
                            stream_lit_commands = get_stream_lit_commands(commands)
                            print(stream_lit_commands)
                            exec(stream_lit_commands)
                        except Exception as e:
                            pass
                    pass