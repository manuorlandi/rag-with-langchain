import streamlit as st
from dotenv import load_dotenv
from src.data.loader import load_dataset
from src.data.processor import create_chunks
from src.models.vectorstore import create_or_get_vector_store
from src.chains.conversation import get_conversation_chain, create_prompt_templates


def main():
    load_dotenv()

    # Set up page configuration
    st.set_page_config(
        page_title="LangChain Documentation Chatbot",
        page_icon=":books:",
    )

    st.title("LangChain Documentation Chatbot")
    st.markdown(
        """
        Ask questions about LangChain documentation and get answers based on the official docs.
        """
    )
    st.image("https://python.langchain.com/svg/langchain_stack_112024_dark.svg") 

    # Load and process data
    if "data_loaded" not in st.session_state:
        with st.spinner("Loading documentation data..."):
            df = load_dataset()
            chunks = create_chunks(df, 1000, 0)
            system_message_prompt, human_message_prompt = create_prompt_templates()
            
            # Initialize vector store
            if "vector_store" not in st.session_state:
                st.session_state.vector_store = create_or_get_vector_store(chunks)
            
            # Initialize conversation chain
            if "conversation" not in st.session_state:
                st.session_state.conversation_chain = get_conversation_chain(
                    st.session_state.vector_store,
                    human_message_prompt,
                    system_message_prompt
                )
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
                
            st.session_state.data_loaded = True

    # Chat interface
    user_question = st.text_input("What would you like to know about LangChain?")

    if user_question:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation_chain.invoke(
                {"question": user_question}
            )
            st.write(response["answer"])
            
            # Add to chat history
            st.session_state.chat_history.append((user_question, response["answer"]))
    
    # Display chat history
    if st.session_state.get("chat_history"):
        st.subheader("Chat History")
        for q, a in st.session_state.chat_history:
            st.markdown(f"**Question:** {q}")
            st.markdown(f"**Answer:** {a}")
            st.markdown("---")

if __name__ == "__main__":
    main()