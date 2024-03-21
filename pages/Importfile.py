import openai
import streamlit as st
import os
import pickle
import logging

from llama_index import  SimpleDirectoryReader
from llama_index.chat_engine import CondenseQuestionChatEngine;
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index import Prompt, SimpleDirectoryReader

from logging import getLogger, StreamHandler, Formatter

import common

index_name = "./data/storage"
pkl_name = "./data/stored_documents.pkl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__name__")
logger.debug("調査用ログ")

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

st.title("📝 ImportFile")

uploaded_file = st.file_uploader("Upload an article", type=("txt", "md","pdf"),key=st.session_state["file_uploader_key"])
if st.button("import",use_container_width=True):
    filepath = None
    try:
        filepath = os.path.join('documents', os.path.basename( uploaded_file.name))
        logger.info(filepath)
        with open(filepath, 'wb') as f:
            f.write(uploaded_file.getvalue())
            f.close()
        document = SimpleDirectoryReader(input_files=[filepath]).load_data()[0]
        logger.info(document)
        st.session_state.stored_docs.append(uploaded_file.name) 
        logger.info(st.session_state.stored_docs)
        st.session_state.index.insert(document=document)
        st.session_state.index.storage_context.persist(persist_dir=index_name)
        response_synthesizer = get_response_synthesizer(response_mode='refine')
        st.session_state.query_engine = st.session_state.index.as_query_engine(response_synthesizer=response_synthesizer)
        st.session_state.chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=st.session_state.query_engine, 
            verbose=True
        )
        with open(pkl_name, "wb") as f:
            print("pickle")
            pickle.dump(st.session_state.stored_docs, f)
        st.session_state["file_uploader_key"] += 1
        st.experimental_rerun()
    except Exception as e:
        # cleanup temp file
        logger.error(e)
        if filepath is not None and os.path.exists(filepath):
            os.remove(filepath)

st.subheader("Import File List")
if "stored_docs" in st.session_state: 
    logger.info(st.session_state.stored_docs)
    for docname in st.session_state.stored_docs:
      st.write(docname)
