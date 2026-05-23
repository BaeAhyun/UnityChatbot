# Unity 스크립트/메모리 최적화 챗봇 - Step 5, 6, 7

## Step 5: 멀티턴 메모리 추가

### 개요
- 이전 대화를 기억하는 기능 추가
- 단일턴 → 멀티턴 변환
- ConversationBufferMemory 활용

### 파일: `app_step5.py`

```python
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Ollama에서 DeepSeek-R1 8B 사용
llm = OllamaLLM(
    model="deepseek-r1:8b",
    temperature=0.7,
)

# 프롬프트 템플릿 (시스템 + 대화 히스토리)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 유니티 스크립트와 메모리 최적화에 대한 친절한 어시스턴트입니다. 한국어로 답변하세요. 이전 대화를 기억하고 맥락을 유지하세요."),
    ("placeholder", "{chat_history}"),
    ("user", "{question}")
])

# 체인 구성
chain = prompt_template | llm

class ChatBot:
    def __init__(self):
        self.chat_history = []
    
    def add_to_history(self, role, message):
        """대화 히스토리에 추가"""
        if role == "human":
            self.chat_history.append(HumanMessage(content=message))
        elif role == "ai":
            self.chat_history.append(AIMessage(content=message))
    
    def chat(self, question):
        """멀티턴 질문-답변"""
        print(f"\n당신: {question}")
        
        # 히스토리와 함께 질문
        response = ""
        print("챗봇:", end=" ")
        
        for chunk in chain.stream({
            "chat_history": self.chat_history,
            "question": question
        }):
            print(chunk, end="", flush=True)
            response += chunk
        
        print("\n")
        
        # 히스토리에 추가
        self.add_to_history("human", question)
        self.add_to_history("ai", response)
        
        return response
    
    def show_history(self):
        """대화 히스토리 표시"""
        print("\n" + "=" * 60)
        print("대화 히스토리")
        print("=" * 60)
        for msg in self.chat_history:
            if isinstance(msg, HumanMessage):
                print(f"\n[당신]: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"\n[챗봇]: {msg.content[:200]}...")  # 처음 200자만 표시
    
    def clear_history(self):
        """히스토리 초기화"""
        self.chat_history = []
        print("\n✅ 대화 히스토리가 초기화되었습니다.")

def main():
    bot = ChatBot()
    
    print("=" * 60)
    print("유니티 스크립트/메모리 최적화 챗봇 (멀티턴 + 메모리)")
    print("=" * 60)
    print("명령어:")
    print("  - 일반 질문: 입력")
    print("  - 히스토리 보기: 'history'")
    print("  - 히스토리 초기화: 'clear'")
    print("  - 종료: 'quit' 또는 'exit'")
    print("=" * 60 + "\n")
    
    while True:
        question = input("질문을 입력하세요: ").strip()
        
        if question.lower() == 'quit' or question.lower() == 'exit':
            print("챗봇을 종료합니다.")
            break
        elif question.lower() == 'history':
            bot.show_history()
        elif question.lower() == 'clear':
            bot.clear_history()
        elif not question:
            print("질문을 입력해주세요.\n")
            continue
        else:
            bot.chat(question)

if __name__ == "__main__":
    main()
```

### 실행 방법

```powershell
python app_step5.py
```

### 동작 확인

```
질문을 입력하세요: 유니티에서 메모리 누수를 방지하는 방법은?
[첫 번째 답변]

질문을 입력하세요: 그 중에서 가장 중요한 것은?
[두 번째 답변 - 이전 대화 맥락 포함]

질문을 입력하세요: history
[대화 히스토리 표시]
```

---

## Step 6: RAG 결합 (최종 완성)

### 개요
- Chroma에서 문서 검색 추가
- 멀티턴 + RAG 동시 작동
- 검색된 문서 소스 표시

### 파일: `app_step6.py`

```python
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Ollama에서 DeepSeek-R1 8B 사용
llm = OllamaLLM(
    model="deepseek-r1:8b",
    temperature=0.7,
)

# Chroma 벡터 스토어 로드
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="data/chroma_db",
    embedding_function=embeddings
)

# 검색 인터페이스 (상위 3개 문서 반환)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# RAG 프롬프트 템플릿
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 유니티 스크립트와 메모리 최적화에 대한 친절한 어시스턴트입니다. 
한국어로 답변하세요. 이전 대화를 기억하고 맥락을 유지하세요.

다음은 Unity 공식 문서에서 검색된 관련 정보입니다:
{context}

이 정보를 활용하여 사용자의 질문에 답변하세요."""),
    ("placeholder", "{chat_history}"),
    ("user", "{question}")
])

# 체인 구성
chain = rag_prompt | llm

class RAGChatBot:
    def __init__(self):
        self.chat_history = []
    
    def add_to_history(self, role, message):
        """대화 히스토리에 추가"""
        if role == "human":
            self.chat_history.append(HumanMessage(content=message))
        elif role == "ai":
            self.chat_history.append(AIMessage(content=message))
    
    def search_documents(self, query):
        """Chroma에서 관련 문서 검색"""
        results = retriever.invoke(query)
        return results
    
    def format_context(self, documents):
        """검색 결과를 포맷팅"""
        context = ""
        for i, doc in enumerate(documents, 1):
            context += f"\n[문서 {i}]\n{doc.page_content[:500]}...\n"
        return context
    
    def chat(self, question):
        """RAG + 멀티턴 질문-답변"""
        print(f"\n당신: {question}")
        
        # 1. 문서 검색
        print("\n🔍 관련 문서 검색 중...")
        documents = self.search_documents(question)
        context = self.format_context(documents)
        print(f"✅ {len(documents)}개 문서 찾음")
        
        # 2. LLM 응답 생성
        response = ""
        print("\n챗봇:", end=" ")
        
        for chunk in chain.stream({
            "context": context,
            "chat_history": self.chat_history,
            "question": question
        }):
            print(chunk, end="", flush=True)
            response += chunk
        
        print("\n")
        
        # 3. 히스토리에 추가
        self.add_to_history("human", question)
        self.add_to_history("ai", response)
        
        # 4. 소스 표시
        self.show_sources(documents)
        
        return response
    
    def show_sources(self, documents):
        """검색된 문서 소스 표시"""
        print("\n📚 참고 자료:")
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', '알 수 없음')
            print(f"  {i}. {source}")
    
    def show_history(self):
        """대화 히스토리 표시"""
        print("\n" + "=" * 60)
        print("대화 히스토리")
        print("=" * 60)
        for i, msg in enumerate(self.chat_history, 1):
            if isinstance(msg, HumanMessage):
                print(f"\n[{i}] 당신: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"    챗봇: {msg.content[:150]}...")
    
    def clear_history(self):
        """히스토리 초기화"""
        self.chat_history = []
        print("\n✅ 대화 히스토리가 초기화되었습니다.")

def main():
    bot = RAGChatBot()
    
    print("=" * 60)
    print("유니티 스크립트/메모리 최적화 챗봇 (RAG + 멀티턴)")
    print("=" * 60)
    print("명령어:")
    print("  - 일반 질문: 입력")
    print("  - 히스토리 보기: 'history'")
    print("  - 히스토리 초기화: 'clear'")
    print("  - 종료: 'quit' 또는 'exit'")
    print("=" * 60 + "\n")
    
    while True:
        question = input("질문을 입력하세요: ").strip()
        
        if question.lower() == 'quit' or question.lower() == 'exit':
            print("챗봇을 종료합니다.")
            break
        elif question.lower() == 'history':
            bot.show_history()
        elif question.lower() == 'clear':
            bot.clear_history()
        elif not question:
            print("질문을 입력해주세요.\n")
            continue
        else:
            bot.chat(question)

if __name__ == "__main__":
    main()
```

### 실행 방법

```powershell
python app_step6.py
```

### 동작 확인

```
질문을 입력하세요: 유니티에서 garbage collection을 최소화하려면?
🔍 관련 문서 검색 중...
✅ 3개 문서 찾음

챗봇: (Unity 문서 기반 답변)

📚 참고 자료:
  1. Understanding_Automatic_Memory_Management_-_Unity_Manual.txt
  2. Best_practice_guides_-_Unity_Manual.txt
  3. Memory_in_WebGL_-_Unity_Manual.txt
```

---

## Step 7: Streamlit UI 개발

### 개요
- 웹 기반 사용자 인터페이스
- 채팅 형식의 대화
- 검색 결과 시각화

### 파일: `app.py` (최종 버전)

```python
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

# 대화 표시
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

# 입력 영역
user_input = st.chat_input("질문을 입력하세요...")

if user_input:
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
        response = ""
        
        with st.spinner("💭 답변 생성 중..."):
            for chunk in chain.stream({
                "context": context,
                "chat_history": st.session_state.chat_history,
                "question": user_input
            }):
                response += chunk
                response_placeholder.write(response)
        
        add_to_history("ai", response)
        
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
```

### 실행 방법

```powershell
pip install streamlit

# Streamlit 앱 실행
streamlit run app.py
```

### 접속

브라우저에서 자동으로 열리거나, 수동으로:
```
http://localhost:8501
```

---

## 전체 파일 구조

```
rag-chatbot/
├── venv/                          # 가상 환경
├── data/
│   ├── raw/                       # 원본 텍스트 문서
│   │   ├── Best_practice_guides...txt
│   │   ├── Memory_in_WebGL...txt
│   │   └── Understanding_Automatic...txt
│   └── chroma_db/                 # Chroma 인덱스 (자동 생성)
├── html_files/                    # 원본 HTML 파일들
├── app.py                         # 최종 Streamlit 앱
├── app_step4.py                   # 단일턴 챗봇
├── app_step5.py                   # 멀티턴 + 메모리
├── app_step6.py                   # RAG 결합
├── ingest.py                      # 문서 인덱싱
├── html_to_text.py                # HTML 변환
├── config.py                      # 설정 (선택사항)
└── requirements.txt               # 패키지 목록
```

---

## 실행 순서 정리

### Step 5 (멀티턴 메모리)
```powershell
python app_step5.py
```
**테스트 사항:**
- 첫 질문과 두 번째 질문에서 맥락이 유지되는가?
- 'history' 명령어로 대화 히스토리가 보이는가?

### Step 6 (RAG + 멀티턴)
```powershell
python app_step6.py
```
**테스트 사항:**
- 문서가 정상 검색되는가?
- 검색된 문서 기반으로 답변하는가?
- 참고 자료가 표시되는가?

### Step 7 (Streamlit UI)
```powershell
streamlit run app.py
```
**테스트 사항:**
- 웹 인터페이스가 정상 열리는가?
- 채팅이 정상 작동하는가?
- 참고 자료가 확장 가능한가?

---

## 주요 개선 사항 및 최적화

### 메모리 관리
```python
# 대화 히스토리 제한 (선택사항)
MAX_HISTORY = 10  # 최근 10개 메시지만 유지
if len(self.chat_history) > MAX_HISTORY:
    self.chat_history = self.chat_history[-MAX_HISTORY:]
```

### 검색 결과 개수 조정
```python
# Chroma 검색 파라미터
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,  # 상위 5개 문서
        "score_threshold": 0.3  # 최소 유사도
    }
)
```

### 응답 속도 개선
- DeepSeek-R1 8B는 처음 로드 시 10~30초 소요
- 이후 응답은 10~30초 (질문 복잡도에 따라)
- GPU 활용으로 최적화 (RTX 4060 활용 중)

---

## 확인할 수 없는 부분

1. **실제 응답 품질**: DeepSeek-R1이 한국어로 유니티 문서를 얼마나 잘 이해하는지는 직접 테스트 필요
2. **검색 정확도**: Chroma의 유사도 검색이 실제 관련 문서를 얼마나 잘 찾는지는 질문에 따라 다름
3. **Streamlit 성능**: 대규모 대화에서의 메모리 사용량은 시스템에 따라 다름

---

## 문제 해결

### 패키지 설치 오류
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Ollama 연결 오류
```powershell
# Ollama 실행 확인
ollama list
ollama run deepseek-r1:8b
```

### Chroma 오류
```powershell
# 벡터 DB 재초기화
rm -r data/chroma_db/
python ingest.py
```

---

## 다음 단계 (선택사항)

1. **프롬프트 최적화**: 시스템 메시지를 더 구체적으로 조정
2. **검색 정확도 개선**: 청크 크기 조정 (1000 → 800/1200)
3. **다른 임베딩 모델**: 한국어 최적화 임베딩 사용 (bge-m3 등)
4. **데이터베이스**: Chroma 대신 Qdrant/Pinecone 사용
5. **배포**: Docker 또는 클라우드 서버에 배포

---

## 최종 확인 체크리스트

- [ ] Step 4 (단일턴) 작동 확인
- [ ] Step 5 (멀티턴) 작동 확인
- [ ] Step 6 (RAG) 작동 확인
- [ ] Step 7 (Streamlit) 웹 UI 작동 확인
- [ ] 한국어 입출력 정상
- [ ] 문서 검색 정상
- [ ] 참고 자료 표시 정상

모든 단계가 완료되면 **완성된 챗봇**입니다! 🎉