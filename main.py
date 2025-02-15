from dotenv import load_dotenv
from langchain import hub
from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain import Tool

load_dotenv()

if __name__ == "__main__":
    instructions = """
    You are an agent designed to write and execute python code to answer the questions.
    You have access to a python REPL, which you can use to execute your code.
    If you get an error, debug your code and try again.
    Only use the output your code to answer the questions.
    You might know the answer without running any code, but you should still run the code to get answer.
    If it doesn't seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")

    tools = [PythonREPLTool()]
    agent = create_react_agent(
        prompt=base_prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )

    python_agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    python_agent_executor.invoke(
        input={
            "input": """
            Generate and save in the current directory 15 QR codes
            that point to www.Github.com or https://gist.github.com,
            assuming you have the qrcode package installed already.
            """,
            "instructions": instructions,  # Add the instructions variable here
        }
    )
    ### Comment this agent if you need run Router Agent.

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        path="Car_details.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    csv_agent.invoke(input={"input": "How many car are there Hyundai name"})
    ### Comment this agent if you need run Router Agent.

    ############# Router Agent ###################

    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor.invoke,
            description=""" Useful  when you need to transform natural language to python and execute the python code,
            returning the result of your code execution
            DOES NOT ACCEPT CODE AS INPUT
            """
        ),
        Tool(
            name="CSV agent",
            func=csv_agent.invoke,
            description=""" Useful  when you need to answer question over Car_details.csv file,
            takes an input the entire question and answer after running pandas calculations.
               """
        )
    ]


    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools
    )

    grand_agent_executor = AgentExecutor(agent=grand_agent,tools=tools,verbose=True)

    print(grand_agent_executor.invoke(input={"input": "How many car are"}))
