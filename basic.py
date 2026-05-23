from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Ollama에서 DeepSeek-R1 8B 사용
llm = OllamaLLM(
    model="deepseek-r1:8b",
    temperature=0.7,
)

# 프롬프트 템플릿 (한국어)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 유니티 스크립트와 메모리 최적화에 대한 친절한 어시스턴트입니다. 한국어로 답변하세요."),
    ("user", "{question}")
])

# 체인 구성
chain = prompt_template | llm

def chat(question):
    """단일턴 질문-답변"""
    print(f"\n당신: {question}")
    print("\n챗봇:", end=" ")
    
    # 스트리밍 응답
    for chunk in chain.stream({"question": question}):
        print(chunk, end="", flush=True)
    
    print("\n")

def main():
    print("=" * 60)
    print("유니티 스크립트/메모리 최적화 챗봇 (단일턴)")
    print("=" * 60)
    print("'quit' 또는 'exit'를 입력하면 종료합니다.\n")
    
    while True:
        question = input("질문을 입력하세요: ").strip()
        
        if question.lower() in ['quit', 'exit']:
            print("챗봇을 종료합니다.")
            break
        
        if not question:
            print("질문을 입력해주세요.\n")
            continue
        
        chat(question)

if __name__ == "__main__":
    main()