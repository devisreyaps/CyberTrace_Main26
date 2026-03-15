import streamlit as st
import requests
import uuid
import io
import base64  # ✅ Keep this


st.set_page_config(page_title="CyberTrace", layout="wide")
st.title("💻 CYBERTRACE - RAG + Malware Scanner")

# Sidebar Tabs
tab1, tab2 = st.tabs(["🔍 Cybersecurity Search", "🦠 File Scanner"])

# ========== TAB 1: RAG CHAT ==========
with tab1:
    # Chat history
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "current_chat_id" not in st.session_state:
        chat_id = str(uuid.uuid4())
        st.session_state.current_chat_id = chat_id
        st.session_state.chats[chat_id] = []

    # Sidebar chat management
    with st.sidebar:
        if st.button("➕ New Chat"):
            new_chat_id = str(uuid.uuid4())
            st.session_state.chats[new_chat_id] = []
            st.session_state.current_chat_id = new_chat_id
            st.rerun()
        
        st.markdown("---")
        st.title("💬 Chats")
        for chat_id in st.session_state.chats.keys():
            chat_messages = st.session_state.chats[chat_id]
            title = chat_messages[0]["content"][:30] if chat_messages else "New Chat"
            if st.button(title, key=chat_id):
                st.session_state.current_chat_id = chat_id
                st.rerun()

    # Current chat
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    for message in current_chat:
        st.chat_message(message["role"]).write(message["content"])

    # Chat input
    user_input = st.chat_input("Ask cybersecurity question...")
    if user_input:
        current_chat.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        try:
            res = requests.post("http://localhost:8000/query", json={"query": user_input, "top_k": 5})
            if res.status_code == 200:
                data = res.json()
                answer = data.get("answer", "No answer")
                source_files = [ctx.get("file") or ctx.get("source_file") or "unknown" for ctx in data.get("contexts", [])]
                sources_text = f"\n\n**📁 Sources:** {', '.join(set(source_files))}"
                full_response = answer + sources_text
            else:
                full_response = f"Backend error: {res.status_code}"
        except Exception as e:
            full_response = f"Connection error: {str(e)}"

        current_chat.append({"role": "assistant", "content": full_response})
        st.chat_message("assistant").markdown(full_response)

# ========== TAB 2: FILE SCANNER ==========
with tab2:
    st.header("🦠 File Scanner")
    uploaded_file = st.file_uploader("Choose file...", type=['exe','dll','pdf','docx'])
    
    if uploaded_file and st.button("🚨 SCAN", type="primary"):
        with st.spinner("Scanning..."):
            file_bytes = uploaded_file.read()
            file_b64 = base64.b64encode(file_bytes).decode()
            
            res = requests.post("http://localhost:8000/analyze_file", 
                              json={"file_data": file_b64, "filename": uploaded_file.name})
            
            result = res.json()
            if result["success"] and result["is_malicious"]:
                st.error("🚨 MALWARE!")
            elif result["success"]:
                st.success("✅ CLEAN")
            st.json(result)

# Footer
st.markdown("---")
st.markdown("🔒 Powered by RAG + VirusTotal | Made for cybersecurity analysis")
