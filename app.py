import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 페이지 설정
st.set_page_config(
    page_title="Unity 챗봇",
    page_icon="🤖",
    layout="wide"
)

# CSS 스타일
st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# 제목
st.title("🎮 Unity 스크립트/메모리 최적화 챗봇")
st.markdown("---")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "llm" not in st.session_state:
    st.session_state.llm = OllamaLLM(
        model="deepseek-r1:8b",
        temperature=0.7,
    )

if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = Chroma(
        persist_directory="data/chroma_db",
        embedding_function=st.session_state.embeddings
    )

# RAG 프롬프트
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 유니티 스크립트와 메모리 최적화에 대한 친절한 어시스턴트입니다.
한국어로 답변하세요. 이전 대화를 기억하고 맥락을 유지하세요.

다음은 Unity 공식 문서에서 검색된 관련 정보입니다:
{context}

이 정보를 활용하여 사용자의 질문에 답변하세요."""),
    ("placeholder", "{chat_history}"),
    ("user", "{question}")
])

chain = rag_prompt | st.session_state.llm

def search_documents(query, k=3):
    """문서 검색"""
    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
    return retriever.invoke(query)

def format_context(documents):
    """검색 결과 포맷팅"""
    context = ""
    for i, doc in enumerate(documents, 1):
        context += f"\n[문서 {i}]\n{doc.page_content[:500]}...\n"
    return context

def add_to_history(role, content):
    """히스토리에 추가"""
    if role == "human":
        st.session_state.chat_history.append(HumanMessage(content=content))
    elif role == "ai":
        st.session_state.chat_history.append(AIMessage(content=content))

def filter_thinking(text):
    """DeepSeek-R1 <think> 태그 및 내용 제거"""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")

    temperature = st.slider(
        "Temperature (창의성)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

    k = st.slider(
        "검색 결과 개수",
        min_value=1,
        max_value=5,
        value=3
    )

    if st.button("🗑️ 대화 초기화"):
        st.session_state.chat_history = []
        st.success("대화 히스토리가 초기화되었습니다.")

    st.markdown("---")
    st.markdown("### 📊 통계")
    st.metric("총 메시지", len(st.session_state.chat_history))

# 메인 영역
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 대화")

    # 기존 메시지 표시
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)

# 입력 영역 (맨 아래 고정)
user_input = st.chat_input("질문을 입력하세요...")

if user_input:
    # 명령어 처리 (LLM으로 보내지 않음)
    if user_input.lower().strip("/") in ("history", "clear"):
        if user_input.lower().strip("/") == "clear":
            st.session_state.chat_history = []
            st.rerun()
        else:
            st.info("대화 기록은 사이드바 하단 '통계'에서 확인하거나, '대화 초기화' 버튼을 사용하세요.")
    else:
        with col1:
            # 사용자 메시지 표시
            with st.chat_message("user"):
                st.write(user_input)

            add_to_history("human", user_input)

            # 문서 검색
            with st.spinner("🔍 관련 문서 검색 중..."):
                documents = search_documents(user_input, k=k)
                context = format_context(documents)

            # AI 응답 생성
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                raw_response = ""

                for chunk in chain.stream({
                    "context": context,
                    "chat_history": st.session_state.chat_history,
                    "question": user_input
                }):
                    raw_response += chunk
                    response_placeholder.write(filter_thinking(raw_response))

                filtered_response = filter_thinking(raw_response)
                add_to_history("ai", filtered_response)

                # 소스 표시
                with st.expander("📚 참고 자료"):
                    for i, doc in enumerate(documents, 1):
                        source = doc.metadata.get('source', '알 수 없음')
                        st.markdown(f"**{i}. {source}**")
                        st.text(doc.page_content[:300] + "...")

# 오른쪽 사이드 (정보)
with col2:
    st.markdown("### ℹ️ 정보")
    st.markdown("""
    - **모델**: DeepSeek-R1 8B
    - **벡터 DB**: Chroma
    - **임베딩**: all-MiniLM-L6-v2
    - **문서**: Unity 공식 매뉴얼
    """)
