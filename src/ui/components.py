import streamlit as st


def display_chat_interface():
    """
    Display the chat interface
    """
    st.title("LangChain Documentation Chatbot")
    st.markdown(
        """
        Ask questions about LangChain documentation and get answers based on the official docs.
        """
    )
    st.image("https://python.langchain.com/svg/langchain_stack_112024_dark.svg")


def display_chat_history(chat_history):
    """
    Display the chat history
    
    Args:
        chat_history (list): The chat history to display
    """
    if not chat_history:
        return
        
    st.subheader("Chat History")
    for q, a in chat_history:
        st.markdown(f"**Question:** {q}")
        st.markdown(f"**Answer:** {a}")
        st.markdown("---")