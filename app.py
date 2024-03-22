import streamlit as st
import os
import pickle
import faiss
import logging

from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index import VectorStoreIndex, Document,Prompt, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage
from llama_index.chat_engine import CondenseQuestionChatEngine;
from llama_index.node_parser import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.constants import DEFAULT_CHUNK_OVERLAP
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.graph_stores import SimpleGraphStore
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
import tiktoken
import yaml
from logging import getLogger, StreamHandler, Formatter

index_name = "./data/storage"
pkl_name = "./data/stored_documents.pkl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__name__")
logger.debug("調査用ログ")

def initialize_index():
    logger.info("initialize_index start")
    text_splitter = TokenTextSplitter(separator="。", chunk_size=1500
      , chunk_overlap=DEFAULT_CHUNK_OVERLAP
      , tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode)
    node_parser = SimpleNodeParser(text_splitter=text_splitter)
    d = 1536
    faiss_index = faiss.IndexFlatL2(d)
    
    llama_debug_handler = LlamaDebugHandler()
    callback_manager = CallbackManager([llama_debug_handler])
    service_context = ServiceContext.from_defaults(node_parser=node_parser,callback_manager=callback_manager)
    lock = Lock()
    with lock:
        if os.path.exists(index_name):
            vectorStorePath = index_name + "/" + "default__vector_store.json"
            storage_context = StorageContext.from_defaults(
              docstore=SimpleDocumentStore.from_persist_dir(persist_dir=index_name),
              graph_store=SimpleGraphStore.from_persist_dir(persist_dir=index_name),
              vector_store=FaissVectorStore.from_persist_path(persist_path=vectorStorePath),
              index_store=SimpleIndexStore.from_persist_dir(persist_dir=index_name),
            )
            st.session_state.index = load_index_from_storage(storage_context=storage_context,service_context=service_context)
            response_synthesizer = get_response_synthesizer(response_mode='refine')
            st.session_state.query_engine = st.session_state.index.as_query_engine(response_synthesizer=response_synthesizer,service_context=service_context)
            st.session_state.chat_engine = CondenseQuestionChatEngine.from_defaults(
                query_engine=st.session_state.query_engine, 
                verbose=True
            )
        else:
            documents = SimpleDirectoryReader("./documents").load_data()
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            st.session_state.index = VectorStoreIndex.from_documents(documents, storage_context=storage_context,service_context=service_context)
            st.session_state.index.storage_context.persist(persist_dir=index_name)
            response_synthesizer = get_response_synthesizer(response_mode='refine')
            st.session_state.query_engine = st.session_state.index.as_query_engine(response_synthesizer=response_synthesizer,service_context=service_context)
            st.session_state.chat_engine = CondenseQuestionChatEngine.from_defaults(
                query_engine=st.session_state.query_engine, 
                verbose=True
            )
        if os.path.exists(pkl_name):
            with open(pkl_name, "rb") as f:
                st.session_state.stored_docs = pickle.load(f)
        else:
            st.session_state.stored_docs=list()

initialize_index()

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
