import streamlit as st
from rag_chain import build_chain

st.set_page_config(page_title="MediAssist", page_icon="🩺", layout="centered")
st.title("🩺 MediAssist")
st.caption("RAG-powered Medical Symptom Checker · LangChain · ChromaDB · Llama-3.1")
st.warning("⚠️ For informational purposes only. Not a substitute for professional medical advice.")

if "chain" not in st.session_state:
    with st.spinner("Loading MediAssist..."):
        st.session_state.chain = build_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("About")
    st.markdown("""
**MediAssist** analyzes your symptoms against a knowledge base of **141 diseases**.

**Try asking:**
- *"I have high fever, joint pain and rash"*
- *"Severe headache and stiff neck"*
- *"Fatigue, weight gain, cold intolerance"*
    """)
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain = build_chain()
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Describe your symptoms..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = st.session_state.chain.invoke(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")