from typing import Dict, List
import os
from langchain.tools import tool, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.utilities.sql_database import SQLDatabase
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
import pandas as pd
load_dotenv()

@tool
def web_search_tool(query: str) -> List[Dict]:
    """
    Perform a web search using Tavily search engine. Fetches detailed responses for unknown entities.
    """
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
    tavily_search = TavilySearchResults(
        max_results=10,
        search_depth="advanced",
        include_answer=False,
        include_raw_content=True,
        include_images=False,
    )

    try:
        search_results = tavily_search.invoke({"query": query})
        return search_results
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]
try:
    # current_dir = os.getcwd()
    current_dir = os.getcwd()+"/app/chatbot"
    file_path = os.path.join(current_dir, "bot_manual.pdf")
    print("File Manual",file_path)
    if os.path.exists(file_path):
        loader = PyPDFLoader(file_path)
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist")
    
    docs = loader.load()
except Exception as e:
    print(f"Error loading PDF: {e}")
    docs = []


try:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
    documents = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_document")
    db = FAISS.from_documents(documents, embedding=embeddings)
    retriever = db.as_retriever()
except Exception as e:
    print(f"Error setting up User Manual tool: {e}")
    documents = None
    
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", convert_system_message_to_human=True)
prompt = ChatPromptTemplate.from_template("""
Based on the context, provide a detailed response to the user's query in an intuitive manner.
Start by retrieving all relevant information for the user's question, ensuring the steps flow logically (e.g., if the query is about "selecting players," include the steps before player selection, like how to select a match).
Ensure the answer is clear, concise, and directly addresses the user's question.
If necessary, refer to relevant sections of the manual to support the answer.
Avoid adding features or functions not mentioned in the manual.
Make the answer user-friendly, ensuring the process is easy to follow and not overwhelming.
If the input is unclear, ask for further clarification.
<context>
{context}
</context>
Question: {input}""")

document_chain=create_stuff_documents_chain(llm,prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain) if documents else None


def user_manual_llm(question):
    if not retrieval_chain:
        return "User manual functionality is currently unavailable."
    try:
        response = retrieval_chain.invoke({"input": question})
        return response.get('answer', "Sorry, I couldn't find an answer.")
    except Exception as e:
        return f"Error processing user manual query: {e}"


@tool
def user_manual_tool(query: str) -> str:
    """
    Dream11 User Manual

    Dream11 is an Indian fantasy sports platform enabling users to create fantasy teams, participate in contests, and track performance through a point-based system. This RAG (Retrieval-Augmented Generation) agent provides support for the following functionalities:

    Key Features:
    1. **General Overview**:
    - Information about the platform's background, goals, and Indian-centric focus.
    - Support for major regional languages, including audio explanations of player selection rationales.

    2. **Navigation and Match Selection**:
    - **Routes**:
        - `/`: Landing page providing an overview of the platform, FAQs, and a "Get Started" CTA.
        - `/home`: Main dashboard for exploring matches and accessing platform features.
        - `/teamSelect`: Match selection page for choosing two teams to compare or simulate.
        - `/custommatch`: Dedicated page for creating custom matches via manual inputs or CSV uploads.
        - `/dreamTeam`: Interactive page for exploring curated teams, analyzing players, and experimenting with strategies.
    - **Match Selection**:
        - Featured and all matches are displayed with filtering options by date.
        - Users can choose scheduled matches or create custom matches when no matches exist.

    3. **Team Creation**:
    - Selection of players for Team A and Team B using a search bar or CSV upload.
    - Automatic generation of a "dream team" of 11 players, with drag-and-drop customization.
    - Player cards provide detailed statistics, career data, and AI-driven inclusion/exclusion rationales.
    - Finalized teams can be saved for participation in contests.

    4. **Custom Match Option**:
    - Users can define match details such as format, date, and team composition.
    - Simulate matches and explore outcomes based on custom inputs.

    5. **Additional Features**:
    - **Player Statistics**: Detailed performance data available on player cards during team creation.
    - **Total Dream Points**: Summarizes team performance based on predefined metrics.
    - **Regional Language Support**: Explanations of player selection available in regional languages via audio formats.
    - **Undo and Save Options**: Undo changes, save progress, or share teams on social media.

    6. **Help and Tutorials**:
    - A step-by-step guide accessible on the homepage to assist with platform navigation and feature utilization.

    FAQs:
    1. **How do I log in to Dream11?**
    - Login is not required. Users only need to specify their preferred regional language.
    2. **How can I select a match?**
    - Browse matches on the dashboard, filter by date, and select a match to view details or generate a team.
    3. **What is the process for creating a team?**
    - Select a match, choose players for two teams, customize the generated dream team, and save the lineup.
    4. **Can I create a custom match?**
    - Yes, custom matches can be created manually or via CSV uploads.
    5. **Where can I view player performance data?**
    - Player cards during team creation display detailed statistics and rationale for their selection.

    This RAG agent provides detailed explanations and guidance on the above features, ensuring users can navigate and utilize Dream11 effectively.
    """

    try:
        # Call the user_manual_llm function with the input query
        response = user_manual_llm(query)
        return response
    except Exception as e:
        # Handle exceptions and return an error message
        return f"An error occurred while fetching the manual details: {e}"

# @tool
# def Search_database(query):


#     load_dotenv()
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

#     os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", convert_system_message_to_human=True)


#     try:
#         current_dir = os.getcwd()+"/app/chatbot"
#         file_path = os.path.join(current_dir, "bot_manual.pdf")
#         print(file_path)
#         if os.path.exists(file_path):
#             loader = PyPDFLoader(file_path)
#         else:
#             raise FileNotFoundError(f"The file {file_path} does not exist")
        
#         docs = loader.load()
#     except Exception as e:
#         print(f"Error loading PDF: {e}")
#         docs = []


#     try:
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
#         documents = text_splitter.split_documents(docs)
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_document")
#         db = FAISS.from_documents(documents, embedding=embeddings)
#         retriever = db.as_retriever()
#     except Exception as e:
#         print(f"Error setting up User Manual tool: {e}")
#         documents = None

#     ## Design ChatPrompt Template
#     prompt = ChatPromptTemplate.from_template("""
#     Based on the context, provide a detailed response to the user's query in an intuitive manner.
#     Start by retrieving all relevant information for the user's question, ensuring the steps flow logically (e.g., if the query is about "selecting players," include the steps before player selection, like how to select a match).
#     Ensure the answer is clear, concise, and directly addresses the user's question.
#     If necessary, refer to relevant sections of the manual to support the answer.
#     Avoid adding features or functions not mentioned in the manual.
#     Make the answer user-friendly, ensuring the process is easy to follow and not overwhelming.
#     If the input is unclear, ask for further clarification.
#     <context>
#     {context}
#     </context>
#     Question: {input}""")


#     document_chain=create_stuff_documents_chain(llm,prompt)

#     retrieval_chain = create_retrieval_chain(retriever, document_chain) if documents else None


#     def user_manual_llm(question):
#         if not retrieval_chain:
#             return "User manual functionality is currently unavailable."
#         try:
#             response = retrieval_chain.invoke({"input": question})
#             return response.get('answer', "Sorry, I couldn't find an answer.")
#         except Exception as e:
#             return f"Error processing user manual query: {e}"



#     try:
#         # Initialize database connection
#         db_user = "postgres"
#         db_password = "Ak%40123"
#         db_host = "db:5432"
#         db_name = "dream11"

#         # Try connecting to the database
#         DATABASE_URL="postgresql://postgres:Ak%40123@db:5432/dream11"
#         database = SQLDatabase.from_uri(DATABASE_URL)
#         # database = SQLDatabase.from_uri(f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}")

#         # Check if the database is connected by running a simple query
#         # if database:
#         #     try:
#         #         # Execute a simple query to confirm the connection is valid
#         #         query_result = database.query("SELECT 1;")
#         #         if query_result:
#         #             print("Database connected successfully.")
#         #         else:
#         #             print("Database connection failed.")
#         #     except Exception as e:
#         #         # print("Database connection failed.")
#         #         print(f"Error executing test query: {e}")
#         #         database = None  # In case of failure, set db to None
#         # else:
#         #     database = None  # If db is None after initialization, it means the connection failed

#         # Setup prompt template for rephrasing
#         answer_prompt = PromptTemplate.from_template(
#             """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
#             Question: {question}
#             SQL Query: {query}
#             SQL Result: {result}
#             Answer: """
#         )

#         # Define the rephrase_answer chain
#         rephrase_answer = answer_prompt | llm | StrOutputParser()

#     except Exception as e:
#         print(f"Error setting up database: {e}")
#         database = None

#     # Add condition to check if db is connected before proceeding
#     if database:
#         print("Database is ready for queries.")
#     else:
#         print("Database connection failed. Functionality is unavailable.")

#     examples = [
#     # Normal queries (Simple retrieval)
#     {
#         "input": "What is the name of the player who scored the highest runs in the IPL 2023?",
#         "query": """
#             SELECT full_name
#             FROM player_stats
#             WHERE start_date BETWEEN '2023-01-01' AND '2023-12-31'
#             ORDER BY highest_runs DESC LIMIT 1;
#         """
#     },
#     {
#         "input": "Give me the teams that participated in the 2023 IPL season.",
#         "query": """
#             SELECT DISTINCT name
#             FROM teams
#             JOIN matches ON teams.name = ANY (matches.teams)
#             WHERE matches.event_name = 'IPL 2023';
#         """
#     },
    
#     # Inter-table queries (Join across tables)
#     {
#         "input": "List the names of all players from 'Mumbai Indians' along with their batting average.",
#         "query": """
#             SELECT players.name, player_stats.batting_average_10
#             FROM players
#             JOIN player_stats ON players.id = player_stats.player_id
#             JOIN teams ON players.team_id = teams.id
#             WHERE teams.name = 'Mumbai Indians';
#         """
#     },
#     {
#         "input": "Show me the highest fantasy score of any player in the IPL 2023 final.",
#         "query": """
#             SELECT player_stats.full_name, MAX(player_stats.fantasy_score_total)
#             FROM player_stats
#             JOIN players ON player_stats.player_id = players.id
#             JOIN matches ON player_stats.match_id = matches.match_id
#             WHERE matches.event_name = 'IPL 2023' AND matches.match_number = 'Final'
#             GROUP BY player_stats.full_name
#             ORDER BY MAX(player_stats.fantasy_score_total) DESC LIMIT 1;
#         """
#     },


#     # Specific queries (Based on a specific condition)
#     {
#         "input": "Which players from 'Delhi Capitals' hit the most sixes in the 2023 season?",
#         "query": """
#             SELECT player_stats.full_name, MAX(player_stats.sixes_scored)
#             FROM player_stats
#             JOIN players ON player_stats.player_id = players.id
#             JOIN teams ON players.team_id = teams.id
#             WHERE teams.name = 'Delhi Capitals' AND player_stats.start_date BETWEEN '2023-01-01' AND '2023-12-31'
#             GROUP BY player_stats.full_name
#             ORDER BY MAX(player_stats.sixes_scored) DESC LIMIT 1;
#         """
#     },
#     {
#         "input": "Give me the total number of runs scored by 'Rajasthan Royals' players in IPL 2023.",
#         "query": """
#             SELECT SUM(player_stats.runs_scored)
#             FROM player_stats
#             JOIN players ON player_stats.player_id = players.id
#             JOIN teams ON players.team_id = teams.id
#             WHERE teams.name = 'Rajasthan Royals' AND player_stats.start_date BETWEEN '2023-01-01' AND '2023-12-31';
#         """
#     },


#     # Common queries (Frequently asked, general information)
#     {
#         "input": "List all the players who played for 'Chennai Super Kings' in the 2023 IPL season.",
#         "query": """
#             SELECT players.name
#             FROM players
#             JOIN teams ON players.team_id = teams.id
#             WHERE teams.name = 'Chennai Super Kings' AND players.id IN (SELECT player_id FROM player_stats WHERE start_date BETWEEN '2023-01-01' AND '2023-12-31');
#         """
#     },
#     {
#         "input": "What are the most common bowling styles in IPL 2023?",
#         "query": """
#             SELECT DISTINCT bowling_style_left_arm_fast, bowling_style_left_arm_spin, bowling_style_right_arm_fast, bowling_style_right_arm_spin
#             FROM player_stats
#             WHERE start_date BETWEEN '2023-01-01' AND '2023-12-31';
#         """
#     },


#     # Combination queries (Using multiple clauses)
#     {
#         "input": "Who were the top 3 highest run-scorers for 'Kolkata Knight Riders' in IPL 2023?",
#         "query": """
#             SELECT player_stats.full_name, SUM(player_stats.runs_scored)
#             FROM player_stats
#             JOIN players ON player_stats.player_id = players.id
#             JOIN teams ON players.team_id = teams.id
#             WHERE teams.name = 'Kolkata Knight Riders' AND player_stats.start_date BETWEEN '2023-01-01' AND '2023-12-31'
#             GROUP BY player_stats.full_name
#             ORDER BY SUM(player_stats.runs_scored) DESC LIMIT 3;
#         """
#     },
#     {
#         "input": "Get the players who took more than 2 wickets in the IPL 2023 final.",
#         "query": """
#             SELECT player_stats.full_name, player_stats.wickets_taken
#             FROM player_stats
#             JOIN players ON player_stats.player_id = players.id
#             JOIN matches ON player_stats.match_id = matches.match_id
#             WHERE matches.event_name = 'IPL 2023' AND matches.match_number = 'Final'
#                 AND player_stats.wickets_taken > 2;
#         """
#     },
#     {
#         "input": "Find all players who played in the '2023 IPL final' and scored more than 30 runs.",
#         "query": """
#             SELECT player_stats.full_name, player_stats.runs_scored
#             FROM player_stats
#             JOIN players ON player_stats.player_id = players.id
#             JOIN matches ON player_stats.match_id = matches.match_id
#             WHERE matches.event_name = 'IPL 2023' AND matches.match_number = 'Final'
#                 AND player_stats.runs_scored > 30;
#         """
#     }
#     ]

#     example_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("human", "{input}\nSQLQuery:"),
#         ("ai", "{query}"),
#     ]
#     )
#     few_shot_prompt = FewShotChatMessagePromptTemplate(
#     example_prompt=example_prompt,
#     examples=examples,
#     input_variables=["input","top_k"],
#     # input_variables=["input"],
#     )

#     final_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run. Unless otherwise specificed.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries."),
#         few_shot_prompt,
#         ("human", "{input}"),
#     ]
#     )



#     def table_llm(question):
#         try:
#             # Generate the SQL query based on the user's question
#             generate_query = create_sql_query_chain(llm, database, final_prompt)
#             print("Generate query",generate_query)
#             try:
#                 print("Question", question)
#                 query_data = generate_query.invoke({"question": question})
#                 print("Query data", query_data)
#                 print("Query End")
#             except Exception as e:
#                 return f"Error during query generation: {str(e)}"

#             # Extract and clean the SQL query
#             try:
#                 query = query_data.strip('`').strip().split('\n', 1)[1].strip()
#                 # query = query_data.strip().replace('```sql', '').replace('```', '').strip()
#                 # Extract the actual query content
#                 # query = query.split('\n', 1)[1].strip() if '\n' in query else query
#                 print("Stipped Query", query)
#             except (IndexError, AttributeError) as e:
#                 return f"Error extracting SQL query: {str(e)}"

#             # Execute the SQL query
#             execute_query = QuerySQLDataBaseTool(db=database)
#             print("Exc", execute_query)
#             try:
#                 result = execute_query.invoke(query)
#                 print("Result --->: ", result)
#             except Exception as e:
#                 return f"Error during query execution: {str(e)}"

#             # Rephrase the answer based on the query and result
#             try:
#                 answer = rephrase_answer.invoke({
#                     "question": question,
#                     "query": query,
#                     "result": result
#                 })
#                 print("Answer", answer)
#             except Exception as e:
#                 return f"Error during answer rephrasing: {str(e)}"

#             return answer

#         except Exception as e:
#             # Catch any unexpected errors and return a fail-safe response
#             return f"An unexpected error occurred: {str(e)}"

#     # ### INTEGRATING THE TOOL

#     from langchain.tools import Tool


#     # Define the User Manual Tool
#     user_manual_tool = Tool(
#     name="UserManual",
#     func=user_manual_llm,
#     description="Provides intuitive answers by referencing the user manual. Ideal for questions about bot features, usage, and manual details."
#     )


#     # Define the Database Query Tool
#     db_query_tool = Tool(
#     name="DBQuery",
#     func=table_llm,
#     description="Executes database queries and provides answers for questions requiring data retrieval from the database."
#     )


#     # Define the Tool Selection Prompt
#     tool_selection_prompt = PromptTemplate.from_template("""
#     You are an intelligent assistant designed to select the most appropriate tool to answer the user's question. 

#     ### Available Tools:
#     1. **UserManual**: Use this tool for queries about:
#     - Bot features, functionality, and usage instructions.
#     - General discussions or non-specific queries.
#     2. **DBQuery**: Use this tool **only** for:
#     - Questions that require detailed data about cricket players.
#     - Queries about cricket matches played on or after **01 August 2024**.

#     ### Instructions:
#     - Analyze the user's question carefully.
#     - Select the tool that best matches the intent and requirements of the question.
#     - Return **only** the tool name: "UserManual" or "DBQuery". Do not provide explanations or additional text.

#     ### User Question:
#     "{question}"
#     ### Selected Tool:
#     """)



#     tool_selection_chain = LLMChain(llm=llm, prompt=tool_selection_prompt)

#     tool_mapping = {
#         "UserManual": user_manual_tool,
#         "DBQuery": db_query_tool,
#     }

#     def tool_selector(question):
#         try:
#             if not question.strip():
#                 return "Please provide a valid question."

#             tool_name = tool_selection_chain.run({"question": question}).strip()
#             tool = tool_mapping.get(tool_name)

#             if tool:
#                 return tool.run(question)
#             else:
#                 return f"Error: No suitable tool found for the question."
#         except Exception as e:
#             return f"Unexpected error occurred: {e}"


#     # Test the function`
#     # print(tool_selector("what is full name of RG sharma"))
#     # print( ("who scored the most number of hundreds in IPL?"))
#     # print(tool_selector("How to create team on this app?"))
    
