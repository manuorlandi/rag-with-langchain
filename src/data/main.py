import os
import streamlit as st
import polars as pl
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_openai import ChatOpenAI


SYSTEM_PROMPT = """
You are a chatbot tasked with responding to questions about the documentation of the LangChain library and project.

You should never answer a question with a question, and you should always respond with the most relevant documentation page.

Given a question, you should respond providing an exhaustive answer from information retrieved from the documentation and add a link with the most relevant documentation page for further information by following the relevant SYSTEM_PROMPT below:\n
{context}
"""


def load_dataset(dataset_name: str = "langchain_docs.csv") -> pl.DataFrame:
    """
    Load the dataset from the csv file

    Args:
        dataset_name (str): The name of the dataset to load

    Returns:
        pl.DataFrame: The loaded dataset containing data gathered by langchain
    """
    data_dir = "./src/data"
    file_path = os.path.join(data_dir, dataset_name)
    df = pl.read_csv(file_path)
    return df


def create_chunks(
    df: pl.DataFrame,
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
) -> list[pl.DataFrame]:
    """
    Create chunks from the dataset

    Args:
        df (pl.DataFrame): The dataset to create chunks from
        chunk_size (int): The size of the chunks to create
        chunk_overlap (int): The overlap between the chunks

    Returns:
        list[pl.DataFrame]: The list of chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    text_chunks = DataFrameLoader(
        df.to_pandas(),
        page_content_column="body"
    ).load_and_split(
        text_splitter=text_splitter
    )

    # add metadata to chucks to simplify retrieval
    for doc in text_chunks:
        title = doc.metadata["title"]
        description = doc.metadata["description"]
        content = doc.page_content
        url = doc.metadata["url"]
        final_content = f"TITLE: {title}\DESCRIPTION:{description}"
        final_content += f"\BODY: {content}\nURL:{url}"
        doc.page_content = final_content

    return text_chunks


def create_or_get_vector_store(chunks: pl.DataFrame) -> FAISS:
    """
    Create or get the vector store

    Args:
        chunks (pl.DataFrame): The chunks to create the vector store from

    Returns:
        FAISS: The vector store
    """

    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    if not os.path.exists("./db"):
        print("Creating vector store")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local("./db")
    else:
        print("Loading vector store")
        vector_store = FAISS.load_local(
            folder_path="./db",
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

    return vector_store


def get_conversation_chain(
    vector_store: FAISS,
    human_message: str,
    system_message: str,
) -> None:
    """
    Create a conversation chain using the vector store and messages

    Args:
        vector_store (FAISS): The vector store containing the embedded documents
        human_message (str): The message template to use for human inputs
        system_message (str): The system message to set the AI assistant's behavior

    Returns:
        ConversationChain: The configured conversation chain for question answering
    """

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages(
                [
                    system_message,
                    human_message,
                ]
            ),
        },
    )
    return conversation_chain


def main():
    load_dotenv()
    df = load_dataset()
    chunks = create_chunks(df, 1000, 0)

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        SYSTEM_PROMPT
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        "{question}"
    )

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = create_or_get_vector_store(chunks)

    if "conversation" not in st.session_state:
        st.session_state.conversation_chain = get_conversation_chain(
            st.session_state.vector_store,
            human_message_prompt,
            system_message_prompt
        )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.set_page_config(
        page_title="Documentation Chatbot",
        page_icon=":books:",
    )

    st.title("Documentation Chatbot")
    st.subheader("Documentation Chatbot")
    st.markdown(
        """
        TEST CHATBOT
        """
    )
    st.image("https://python.langchain.com/svg/langchain_stack_112024_dark.svg") 

    user_question = st.text_input("Cosa vuoi chiedere?")

    if user_question:
        with st.spinner("Sto pensando..."):
            response = st.session_state.conversation_chain.invoke(
                {"question": user_question}
            )
            st.write(response["answer"])


if __name__ == "__main__":
    main()
