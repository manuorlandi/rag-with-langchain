# LangChain Documentation Chatbot

A chatbot that answers questions about LangChain documentation using RAG (Retrieval-Augmented Generation).

## Features

- Loads and processes LangChain documentation
- Creates vector embeddings for efficient retrieval
- Provides conversational interface for documentation questions
- Maintains chat history

## Setup

1. Clone the repository
2. Install dependencies using Poetry:
   ```
   poetry install
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Place the LangChain documentation CSV file in `data/raw/langchain_docs.csv`

## Running the Application
