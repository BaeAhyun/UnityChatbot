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