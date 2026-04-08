# text2sqlbot
CSV based Text to SQL Assistant Chatbot

## Quick Setup
 
**1. Add your Groq API key**
 
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```
 
**2. Install dependencies**
```bash
pip install -r requirements.txt
```
 
**3. Run the app**
```bash
streamlit run app.py
```

## Text - SQL Approach


To convert the natural text to an SQL query we need an LLM. As LLMs are good at understanding natural language and can return us the valid SQL results. In this task, I am using a **Groq LLaMA 3.3 model** (why? they are for **free tier**, **fast**, **strong SQL generation quality** and also have some best trained local models).


## Schema Handling Approach


First the user will upload the CSVs, and from the uploaded files, need to **detect the relations** between them, if it exists or not. How? again using an LLM. Once, the user uploads the files, I have an info about the uploaded files & the data it has, features/columns and samples/rows. Based on the available info, and using an LLM I can extract the relations between the files. Let's say employee_data file has EmpID and employee_engagement_survey_data has Employee ID which are one and the same. For that, first I load each data to **in-memory SQLite** (one table for file), extract schema for each table and detect relationships using **semantic similarity** between column names.


Given a user's question, need to return only the relevant table schemas which is done by an **Embedding based retrieval**. Embedded each schema document using Sentence Transformer **all-MiniLM-L6-v2**. Computed **Cosine similarity / Dot product** between query embedding and each table's schema embedding and returned out top scored tables.


To provide the context to LLM and for follow-up questions and for reducing the **token context window**, **chat history** / **session state** with summary must be maintained. Store info about which tables are referenced before, filters or conditions that the user mentioned in their earlier queries, any clarifications given by the user and the resulted SQL query and it's results as well.


**Prompt design** for LLM that results SQL query must include rich context like uploaded table's schema, sample rows, detected relations for accurate SQL queries. Also added strict rules to only use the columns and table names that are provided and don't hallucinate or assume or create a new ones. Prompted just to return the SQL query and a short explanation.


Using the llama-3.3-70b-versatile model as LLM, the LLM generates the response (sql and explanation). Run the generated sql on SQLite connection and see if it works. If the generated sql query failed, retry by providing the failed error for a new result (sql and explanation). Also when the query results out 0 records, confirm using an evaluator before returning to the user, whether it is really the right query that results 0 records or an incorrect query.


**Execution flow:**
 
```
Generate SQL
     ↓
Execute on SQLite
     ↓ (on error)
Retry with error feedback → Re-execute
     ↓ (on 0 rows)
Validate: is this truly empty, or a bad query?
     ↓
Return result + explanation to user
```

## Known limitations

- Detecting relations or joins for ambiguously related columns from 2 different CSVs is a challenge

- Transitive relationships are hard to detect like A->B, B->C and C->D, then A->D is hardly detected.

- Very large CSVs will cause memory issues as only an in-memory SQLite is used

