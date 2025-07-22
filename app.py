# app.py
import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(__file__))

from rag_utils import load_llm, load_vectorstore, create_qa_chain

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Medical Chatbot", 
    page_icon="🏥", 
    layout="wide"
)

# --- Header ---
st.title("🏥 Medical Chatbot")
st.markdown("Ask questions about medical conditions, symptoms, and treatments based on medical data.")

# --- Disclaimer ---
st.warning("⚠️ **Medical Disclaimer:** This chatbot provides general information only. Always consult healthcare professionals for medical advice.")

# --- Initialize Session State for QA Chain and LLM ---
if "qa_chain" not in st.session_state:
    with st.spinner("🔄 Initializing chatbot..."):
        try:
            llm = load_llm()
            vectorstore = load_vectorstore()
            st.session_state.qa_chain = create_qa_chain(llm, vectorstore)
            st.success("✅ Chatbot ready!")
        except Exception as e:
            st.error(f"❌ Failed to initialize: {e}")
            st.exception(e)
            st.info("Please check your .env file and Pinecone configuration.")
            st.stop()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your medical question here..."):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                response = st.session_state.qa_chain.invoke({"query": prompt})
                answer = response.get("result", "I'm not sure. Please consult a medical professional.")
                source_documents = response.get("source_documents", [])

                # Display answer
                st.markdown(answer)
                
                # Show sources if available
                if source_documents:
                    st.divider()
                    st.subheader("📚 Sources")
                    unique_sources = set()
                    for doc in source_documents:
                        source_name = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'N/A')
                        source_info = f"{source_name} (Page: {page})"
                        if source_info not in unique_sources:
                            st.write(f"• {source_info}")
                            unique_sources.add(source_info)

                # Save full response
                full_response = answer
                if source_documents:
                    full_response += "\n\n**📚 Sources:**\n"
                    unique_sources = set()
                    for doc in source_documents:
                        source_name = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'N/A')
                        source_info = f"• {source_name} (Page: {page})"
                        if source_info not in unique_sources:
                            full_response += f"{source_info}\n"
                            unique_sources.add(source_info)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"⚠️ Error: {e}")
                st.exception(e)
                error_msg = "Sorry, I encountered an error. Please try again."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})