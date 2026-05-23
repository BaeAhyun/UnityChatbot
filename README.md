# Unity 스크립트/메모리 최적화 챗봇

Unity 공식 문서를 기반으로 스크립트 및 메모리 최적화 질문에 답변하는 RAG 챗봇입니다.

## 주요 기능

- **RAG (Retrieval-Augmented Generation)**: Unity 공식 문서에서 관련 내용을 검색해 답변
- **멀티턴 대화**: 이전 대화 맥락을 기억
- **Streamlit UI**: 웹 기반 채팅 인터페이스
- **참고 자료 표시**: 답변에 사용된 문서 출처 제공

## 기술 스택

- **LLM**: DeepSeek-R1 8B (Ollama)
- **벡터 DB**: Chroma
- **임베딩**: sentence-transformers/all-MiniLM-L6-v2
- **프레임워크**: LangChain, Streamlit

## 참고 문서

- Best practice guides - Unity Manual
- Memory in WebGL - Unity Manual
- Understanding Automatic Memory Management - Unity Manual

## 설치 및 실행

### 1. 사전 요구사항

- [Ollama](https://ollama.com) 설치 후 DeepSeek-R1 8B 모델 다운로드
```bash
ollama pull deepseek-r1:8b
```

### 2. 패키지 설치

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 문서 인덱싱

```bash
python ingest.py
```

### 4. 실행

```bash
# Streamlit 웹 UI
streamlit run app.py

# 터미널 버전 (RAG + 멀티턴)
python rag.py

# 터미널 버전 (멀티턴만)
python multiturn.py

# 터미널 버전 (단일턴)
python basic.py
```

## 파일 구조

```
UnityChatbot/
├── app.py              # Streamlit 웹 UI
├── rag.py              # RAG + 멀티턴 터미널 챗봇
├── multiturn.py        # 멀티턴 터미널 챗봇
├── basic.py            # 단일턴 터미널 챗봇
├── ingest.py           # 문서 인덱싱 (Chroma)
├── html_to_text.py     # HTML → 텍스트 변환
├── data/
│   └── raw/            # Unity 공식 문서 텍스트
└── html_files/         # 원본 HTML 파일
```
