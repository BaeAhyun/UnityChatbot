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
        print("\n검색 중...")
        documents = self.search_documents(question)
        context = self.format_context(documents)
        print(f"{len(documents)}개 문서 찾음")

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
        print("\n참고 자료:")
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
        print("\n대화 히스토리가 초기화되었습니다.")

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
