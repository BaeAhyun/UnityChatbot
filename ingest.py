import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 설정
DATA_PATH = "data/raw"
CHROMA_PATH = "data/chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_documents():
    """문서 로드"""
    documents = []
    
    # txt 파일 로드
    for txt_file in Path(DATA_PATH).glob("*.txt"):
        print(f"Loading {txt_file.name}...")
        loader = TextLoader(str(txt_file), encoding="utf-8")
        documents.extend(loader.load())
    
    print(f"Loaded {len(documents)} documents")
    return documents

def chunk_documents(documents):
    """문서를 작은 청크로 분할"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

def create_vectorstore(chunks):
    """Chroma 벡터 스토어 생성"""
    print("Creating embeddings and storing in Chroma...")
    
    # HuggingFace 임베딩 (영어 전용, 무료)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Chroma에 저장
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"Vectorstore saved to {CHROMA_PATH}")
    return vectorstore

def main():
    # 기존 Chroma 삭제 (재생성)
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)
        print(f"Deleted existing {CHROMA_PATH}")
    
    # 파이프라인 실행
    documents = load_documents()
    if not documents:
        print("No documents found in data/raw/")
        print("Please add .txt files to data/raw/ first")
        return
    
    chunks = chunk_documents(documents)
    create_vectorstore(chunks)
    print("Ingest complete!")

if __name__ == "__main__":
    main()