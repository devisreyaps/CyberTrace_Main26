# Frontend
import streamlit as st
import requests

st.title("💻 CYBERTRACE")

if "history" not in st.session_state:
    st.session_state.history = []
st.sidebar.header("Filters")
date_filter = st.sidebar.date_input("Filter by Date")
threat_type = st.sidebar.selectbox("Threat Type", ["All", "Malware", "APT", "Vulnerability"])

user_input = st.chat_input("Ask your cybersecurity question...")

if user_input:
    res = requests.post("http://127.0.0.1:8000/chat", json={"question": user_input})
    answer = res.json()["answer"]
    st.session_state.history.append((user_input, answer))

for q, a in st.session_state.history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)




    How to run : streamlit run frontend/app.py
