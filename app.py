import streamlit as st  # Streamlitライブラリをインポート
import os  # OSライブラリをインポート（ファイルパス操作用）
import pickle  # オブジェクトの永続化（保存・読み込み）用ライブラリをインポート
import faiss  # 高速な類似度検索ライブラリをインポート
import logging  # ログ出力用ライブラリをインポート

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
from logging import getLogger, StreamHandler, Formatter

index_name = "./data/storage"
pkl_name = "./data/stored_documents.pkl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__name__")
logger.debug("調査用ログ")

def initialize_index():  # インデックス初期化関数を定義
    logger.info("initialize_index start")  # 初期化開始をログに記録
    # テキスト分割器を設定。区切り文字は「。」、チャンクサイズは1500、オーバーラップはデフォルト値を使用
    text_splitter = TokenTextSplitter(separator="。", chunk_size=1500, chunk_overlap=DEFAULT_CHUNK_OVERLAP, tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode)
    node_parser = SimpleNodeParser(text_splitter=text_splitter)  # ノードパーサーをテキスト分割器で初期化
    d = 1536  # ベクトルの次元数
    faiss_index = faiss.IndexFlatL2(d)  # FAISSインデックスをL2距離で初期化
    
    llama_debug_handler = LlamaDebugHandler()  # デバッグハンドラーを初期化
    callback_manager = CallbackManager([llama_debug_handler])  # コールバックマネージャーをデバッグハンドラーで初期化
    # サービスコンテキストをデフォルト設定で初期化（ノードパーサーとコールバックマネージャーを含む）
    service_context = ServiceContext.from_defaults(node_parser=node_parser,callback_manager=callback_manager)
    lock = Lock()  # マルチプロセッシング用のロックを初期化
    with lock:  # ロックを使用して以下の処理をスレッドセーフに実行
        if os.path.exists(index_name):  # 既存のインデックスがある場合
            vectorStorePath = index_name + "/" + "default__vector_store.json"  # ベクトルストアのパスを設定
            # ストレージコンテキストを既存のパスから初期化
            storage_context = StorageContext.from_defaults(
              docstore=SimpleDocumentStore.from_persist_dir(persist_dir=index_name),
              graph_store=SimpleGraphStore.from_persist_dir(persist_dir=index_name),
              vector_store=FaissVectorStore.from_persist_path(persist_path=vectorStorePath),
              index_store=SimpleIndexStore.from_persist_dir(persist_dir=index_name),
            )
            # ストレージコンテキストとサービスコンテキストからインデックスをロード
            st.session_state.index = load_index_from_storage(storage_context=storage_context,service_context=service_context)
            response_synthesizer = get_response_synthesizer(response_mode='refine')  # 応答合成器を設定
            # インデックスをクエリエンジンとして設定し、応答合成器とサービスコンテキストを使用
            st.session_state.query_engine = st.session_state.index.as_query_engine(response_synthesizer=response_synthesizer,service_context=service_context)
            # チャットエンジンをデフォルト設定で初期化（クエリエンジンを使用）
            st.session_state.chat_engine = CondenseQuestionChatEngine.from_defaults(
                query_engine=st.session_state.query_engine, 
                verbose=True
            )
        else:  # 既存のインデックスがない場合
            documents = SimpleDirectoryReader("./documents").load_data()  # ドキュメントを読み込み
            vector_store = FaissVectorStore(faiss_index=faiss_index)  # ベクトルストアをFAISSインデックスで初期化
            storage_context = StorageContext.from_defaults(vector_store=vector_store)  # ストレージコンテキストをベクトルストアで初期化
            # ドキュメントとストレージコンテキスト、サービスコンテキストからインデックスを作成
            st.session_state.index = VectorStoreIndex.from_documents(documents, storage_context=storage_context,service_context=service_context)
            st.session_state.index.storage_context.persist(persist_dir=index_name)  # インデックスを永続化
            response_synthesizer = get_response_synthesizer(response_mode='refine')  # 応答合成器を設定
            # インデックスをクエリエンジンとして設定し、応答合成器とサービスコンテキストを使用
            st.session_state.query_engine = st.session_state.index.as_query_engine(response_synthesizer=response_synthesizer,service_context=service_context)
            # チャットエンジンをデフォルト設定で初期化（クエリエンジンを使用）
            st.session_state.chat_engine = CondenseQuestionChatEngine.from_defaults(
                query_engine=st.session_state.query_engine, 
                verbose=True
            )
        if os.path.exists(pkl_name):  # 保存されたドキュメントがある場合
            with open(pkl_name, "rb") as f:  # ドキュメントを読み込み
                st.session_state.stored_docs = pickle.load(f)
        else:  # 保存されたドキュメントがない場合
            st.session_state.stored_docs=list()  # 空のリストを設定

initialize_index()  # インデックス初期化関数を呼び出し

st.title("💬 Chatbot")  # Streamlitでページタイトルを設定
if st.button("リセット",use_container_width=True):  # Streamlitでリセットボタンを設定
    st.session_state.chat_engine.reset()  # チャットエンジンをリセット
    st.session_state.messages = [{"role": "assistant", "content": "質問をどうぞ"}]  # メッセージリストを初期化
    st.experimental_rerun()  # Streamlitアプリを再実行
    logger.info("reset")  # ログにリセット情報を出力

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "質問をどうぞ"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():  # Streamlitでチャット入力を受け取り、入力があれば以下を実行
    st.session_state.messages.append({"role": "user", "content": prompt})  # ユーザーのメッセージを追加
    st.chat_message("user").write(prompt)  # ユーザーメッセージをチャットに表示
    response = st.session_state.chat_engine.chat(prompt)  # チャットエンジンで応答を生成
    msg = str(response)  # 応答を文字列に変換
    st.session_state.messages.append({"role": "assistant", "content": msg})  # アシスタントのメッセージを追加
    st.chat_message("assistant").write(msg)  # アシスタントメッセージをチャットに表示
