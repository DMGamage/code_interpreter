from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool

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
        tools=tools
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # agent_executor.invoke(
    #     input={
    #         "input": """
    #         Generate and save in the current directory 15 QR codes
    #         that point to www.Github.com or https://gist.github.com,
    #         assuming you have the qrcode package installed already.
    #         """
    #     }
    # )

    agent_executor.invoke(
        input={
            "input": """
            Generate and save in the current directory 15 QR codes
            that point to www.Github.com or https://gist.github.com,
            assuming you have the qrcode package installed already.
            """,
            "instructions": instructions  # Add the instructions variable here
        }
    )
