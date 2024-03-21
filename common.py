import extra_streamlit_components as stx
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__name__")
logger.debug("調査用ログ")

#ログインの確認
def check_login():
    if 'authentication_status' not in st.session_state:
        st.session_state['authentication_status'] = None
    if st.session_state["authentication_status"] is None or False:
        st.warning("**ログインしてください**")
        st.stop()
