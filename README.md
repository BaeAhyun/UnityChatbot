# Unity 스크립트/메모리 최적화 RAG 챗봇

Unity 공식 매뉴얼을 기반으로 스크립트 작성과 메모리 최적화 관련 질문에 답변하는 한국어 RAG(Retrieval-Augmented Generation) 챗봇입니다. 로컬 LLM(DeepSeek-R1)을 사용하여 외부 API 호출 없이 동작합니다.

---

## 주요 기능

- **로컬 LLM 기반 응답**: Ollama로 구동되는 DeepSeek-R1 8B 모델 사용 (외부 API 키 불필요)
- **RAG(검색 증강 생성)**: Unity 공식 매뉴얼에서 관련 문서를 검색해 답변에 활용
- **멀티턴 대화**: 이전 대화 맥락을 기억하여 연속된 질문에 대응
- **참고 자료 표시**: 답변 생성에 사용된 원본 문서 출처를 함께 제공
- **웹 UI**: Streamlit 기반 채팅 인터페이스 제공
- **한국어 입출력**: 한국어 질문/답변 지원

---

## 기술 스택

| 구분 | 사용 기술 |
|------|----------|
| LLM | DeepSeek-R1 8B (Ollama) |
| 임베딩 모델 | sentence-transformers/all-MiniLM-L6-v2 |
| 벡터 데이터베이스 | Chroma |
| 프레임워크 | LangChain |
| 웹 UI | Streamlit |
| 언어 | Python |

> **참고**: Python 정확한 버전 및 각 패키지의 호환 버전은 `requirements.txt`를 참조하세요.

---

## 데이터 소스

Unity 공식 매뉴얼 페이지를 HTML로 저장한 뒤 텍스트로 변환하여 사용합니다.

- `Understanding Automatic Memory Management - Unity Manual`
- `Memory in WebGL - Unity Manual`
- `Best practice guides - Unity Manual`

---

## 실행 방법

### Streamlit 웹 앱 (권장)

```powershell
streamlit run app.py
```

브라우저에서 자동으로 열리거나, 다음 주소로 직접 접속합니다.

```
http://localhost:8501
```

### CLI 버전 (단계별)

학습 및 디버깅 목적으로 단계별 스크립트를 직접 실행할 수 있습니다.

```powershell
python app_step4.py    # 단일턴
python app_step5.py    # 멀티턴 + 메모리
python app_step6.py    # 멀티턴 + RAG
```

CLI 명령어:

| 명령어 | 동작 |
|--------|------|
| (질문 입력) | 챗봇에 질문 |
| `history` | 대화 히스토리 표시 |
| `clear` | 히스토리 초기화 |
| `quit` / `exit` | 종료 |

---

## 동작 방식

```
사용자 질문
   ↓
[1] Chroma 벡터 검색 (상위 k개 관련 문서 추출)
   ↓
[2] 검색된 문서 + 대화 히스토리 + 질문을 프롬프트로 구성
   ↓
[3] DeepSeek-R1 8B (Ollama) 응답 생성 (스트리밍)
   ↓
[4] 답변 + 참고 자료 출처 표시
```

---

## 주요 설정값

`app.py` 사이드바에서 실시간으로 조정 가능합니다.

| 항목 | 기본값 | 설명 |
|------|--------|------|
| Temperature | 0.7 | 응답의 창의성 (0.0~1.0) |
| 검색 결과 개수 (k) | 3 | Chroma에서 가져올 문서 수 (1~5) |

---

## 알려진 제약 사항

- **응답 속도**: 개발 환경(RTX 4060) 기준, 최초 로드 시 10~30초, 이후 응답 생성도 질문 복잡도에 따라 10~30초 소요.
- **검색 정확도**: 임베딩 모델이 영문 중심이므로 한국어 질의의 검색 품질은 질문에 따라 편차가 있을 수 있음.
- **데이터 범위**: 현재 인덱싱된 문서는 Unity 매뉴얼 3개 페이지로 한정됨.

---

## 참고 자료

- [Ollama 공식 문서](https://ollama.com)
- [LangChain 공식 문서](https://python.langchain.com)
- [Streamlit 공식 문서](https://docs.streamlit.io)
- [Chroma 공식 문서](https://docs.trychroma.com)
- [Unity Manual (KR)](https://docs.unity3d.com/kr/2018.3/Manual/index.html)
