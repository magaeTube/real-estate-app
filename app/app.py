# from app.db.chromadb import connect_chromadb
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from core.config import get_config_info
from db.chromadb import get_chromadb
from dotenv import load_dotenv

import streamlit as st

def main():
    load_dotenv()
    config = get_config_info()

    embeddings_model = OpenAIEmbeddings()

    collection_name = "test_pypdfloader"
    db = get_chromadb(site=config["chromadb"]["local"],
                      collection_name=collection_name,
                      embeddings_model=embeddings_model)
    print(db)

    llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")
    retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(search_kwargs={'k': 5}), llm=llm
    )

    template = """
        당신은 주어진 정보를 바탕으로 질문에 답변하는 AI 어시스턴트입니다.
        반드시 제공된 '컨텍스트' 안의 내용만을 사용하여 답변해야 합니다.
        만약 컨텍스트에 답변의 근거가 없다면, "제공된 정보에서 답변을 찾을 수 없습니다."라고 솔직하게 답변하세요.
        답변은 한국어로, 명확하고 간결하게 작성해주세요.

        컨텍스트:
        {context}

        질문:
        {question}

        답변:
        """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | output_parser
    )

    st.write("청약 봇")
    query = st.text_input("청약 관련해서 궁금한 사항을 물어보세요")
    response = rag_chain.invoke(query)
    st.write(response)

if __name__ == "__main__":
    main()