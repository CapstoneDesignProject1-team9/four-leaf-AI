"""
RAG (Retrieval Augmented Generation) 파이프라인

흐름:
  사용자 질문
    → 질문 임베딩
    → ChromaDB에서 유사 문서 검색
    → 검색된 문서 + 질문을 HyperCLOVA X에게 전달
    → 답변 생성

지식베이스:
  ./knowledge/ 폴더에 PDF/TXT 파일 넣으면 자동 인덱싱
  예시:
    - 학교생활_가이드.pdf
    - 진로_상담_FAQ.txt
    - 전공별_취업정보.pdf
"""

import logging
import os
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

logger = logging.getLogger(__name__)

# 전역 벡터스토어 인스턴스
_vectorstore: Chroma | None = None


def _get_embeddings():
    """
    한국어 임베딩 모델
    - 로컬 실행: sentence-transformers (KLUE/RoBERTa 기반)
    - 프로덕션: NAVER Clova Embedding API로 교체 가능
    """
    return HuggingFaceEmbeddings(
        model_name="snunlp/KR-ELECTRA-discriminator",  # 한국어 특화 모델
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


async def init_vectorstore():
    """앱 시작 시 벡터스토어 초기화"""
    global _vectorstore

    embeddings = _get_embeddings()
    persist_dir = settings.CHROMA_PERSIST_DIR
    knowledge_dir = settings.KNOWLEDGE_BASE_DIR

    # 기존 ChromaDB가 있으면 로드
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        logger.info("기존 ChromaDB 로드: %s", persist_dir)
        _vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        return

    # 새로 생성: knowledge/ 폴더 문서 인덱싱
    logger.info("ChromaDB 새로 생성 중...")
    documents = []

    if os.path.exists(knowledge_dir):
        # PDF 로드
        pdf_files = list(Path(knowledge_dir).glob("**/*.pdf"))
        for pdf_path in pdf_files:
            loader = PyPDFLoader(str(pdf_path))
            documents.extend(loader.load())
            logger.info("PDF 로드: %s", pdf_path.name)

        # TXT 로드
        txt_files = list(Path(knowledge_dir).glob("**/*.txt"))
        for txt_path in txt_files:
            loader = TextLoader(str(txt_path), encoding="utf-8")
            documents.extend(loader.load())
            logger.info("TXT 로드: %s", txt_path.name)

    if not documents:
        # 지식베이스가 없으면 기본 샘플 데이터로 초기화
        logger.warning("knowledge/ 폴더가 비어있습니다. 샘플 데이터로 초기화합니다.")
        from langchain_core.documents import Document
        documents = [
            Document(
                page_content="사학년 1학기에 졸업논문을 제출해야 합니다. 지도교수와 미리 주제를 협의하세요.",
                metadata={"source": "sample", "category": "학사정보"},
            ),
            Document(
                page_content="취업 준비를 위해 학교 취업지원센터를 활용하세요. 이력서 첨삭, 모의면접 등 서비스를 제공합니다.",
                metadata={"source": "sample", "category": "취업"},
            ),
            Document(
                page_content="복수전공 신청은 2학년 1학기부터 가능하며, 학점 3.0 이상이어야 합니다.",
                metadata={"source": "sample", "category": "학사정보"},
            ),
        ]

    # 문서 청크 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    logger.info("총 %d개 청크 생성", len(chunks))

    # ChromaDB에 저장
    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    logger.info("ChromaDB 초기화 완료")


def get_vectorstore() -> Chroma:
    if _vectorstore is None:
        raise RuntimeError("벡터스토어가 초기화되지 않았습니다.")
    return _vectorstore


def build_rag_chain():
    """RAG QA 체인 생성"""
    vectorstore = get_vectorstore()

    # AI 튜터 시스템 프롬프트
    system_prompt = """당신은 대학생들을 위한 친근한 AI 튜터입니다.
대학생활 적응, 학사 정보, 진로 상담, 취업 준비 등 다양한 주제로 도움을 드립니다.

아래 참고 자료를 활용하여 정확하고 도움이 되는 답변을 한국어로 제공하세요.
참고 자료에 없는 내용은 솔직하게 모른다고 말하고, 관련 부서(학생처, 취업지원센터 등)를 안내하세요.

참고 자료:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},  # 상위 3개 문서 검색
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return chain
