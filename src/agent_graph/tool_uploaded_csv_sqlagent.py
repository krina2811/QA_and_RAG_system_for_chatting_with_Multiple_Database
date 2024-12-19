from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.tools import tool
from agent_graph.load_tools_config import LoadToolsConfig
from langchain_community.agent_toolkits import create_sql_agent
import os
TOOLS_CFG = LoadToolsConfig()


class UploadCsvSQLAgent:
    def __init__(self, sqldb_directory: str, llm: str, llm_temerature: float) -> None:
        """Initializes the ChinookSQLAgent with the LLM and database connection.

        Args:
            sqldb_directory (str): The directory path to the SQLite database file.
            llm (str): The LLM model identifier (e.g., "gpt-3.5-turbo").
            llm_temerature (float): The temperature value for the LLM, determining the randomness of the model's output.
        """
        self.sql_agent_llm = ChatOpenAI(model=llm, temperature=llm_temerature)
        self.db = SQLDatabase.from_uri(f"sqlite:///{sqldb_directory}")
        print(self.db.get_usable_table_names())
        self.agent_executor = create_sql_agent(self.sql_agent_llm, db=self.db, agent_type="openai-tools", verbose=True)


@tool
def query_upload_csv_sqldb(query: str) -> str:
    """Query the upload csv SQL Database. Input should be a search query."""
    agent = UploadCsvSQLAgent(
        sqldb_directory=TOOLS_CFG.upload_csv_directory,
        llm=TOOLS_CFG.upload_csv_sqlagent_llm,
        llm_temerature=TOOLS_CFG.upload_csv_llm_temperature
    )

    response = agent.agent_executor.invoke({"input": query})
    return response["output"]

    