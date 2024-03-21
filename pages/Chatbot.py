import streamlit as st
import logging

from llama_index import Prompt

import common

index_name = "./data/storage"
pkl_name = "./data/stored_documents.pkl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__name__")
logger.debug("調査用ログ")

st.title("💬 Chatbot")
if st.button("リセット",use_container_width=True):
    st.session_state.chat_engine.reset()
    st.session_state.messages = [{"role": "assistant", "content": "質問をどうぞ"}]
    st.experimental_rerun()
    logger.info("reset")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "質問をどうぞ"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = st.session_state.chat_engine.chat(prompt)
    msg = str(response)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
