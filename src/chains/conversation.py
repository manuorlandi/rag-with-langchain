from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

SYSTEM_PROMPT = """
You are a chatbot tasked with responding to questions about the documentation of the LangChain library and project.

You should never answer a question with a question, and you should always respond with the most relevant documentation page.

Given a question, you should respond providing an exhaustive answer from information retrieved from the documentation and add a link with the most relevant documentation page for further information by following the relevant SYSTEM_PROMPT below:\n
{context}
"""


def create_prompt_templates():
    """
    Create the prompt templates for the conversation

    Returns:
        tuple: The system and human message prompt templates
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        SYSTEM_PROMPT
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        "{question}"
    )
    
    return system_message_prompt, human_message_prompt

def get_conversation_chain(
    vector_store: FAISS,
    human_message: HumanMessagePromptTemplate,
    system_message: SystemMessagePromptTemplate,
    model_name: str = "gpt-3.5-turbo"
) -> ConversationalRetrievalChain:
    """
    Create a conversation chain using the vector store and messages

    Args:
        vector_store (FAISS): The vector store containing the embedded documents
        human_message (HumanMessagePromptTemplate): The message template for human inputs
        system_message (SystemMessagePromptTemplate): The system message template
        model_name (str): The name of the LLM model to use

    Returns:
        ConversationalRetrievalChain: The configured conversation chain for question answering
    """
    llm = ChatOpenAI(model=model_name)
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