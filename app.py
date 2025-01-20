from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from bs4 import BeautifulSoup
import streamlit as st
import openai

url = "https://developers.cloudflare.com/sitemap-0.xml"


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self, *args, **kwargs):
        self.message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


answers_prompt = ChatPromptTemplate.from_template(
    """
        Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                    
        Then, give a score to the answer between 0 and 5.
        If the answer answers the user question the score should be high, else it should be low.
        Make sure to always include the answer's score even if it's 0.
        Context: {context}
                                                    
        Examples:
                                                    
        Question: How far away is the moon?
        Answer: The moon is 384,400 km away.
        Score: 5
                                                    
        Question: How far away is the sun?
        Answer: I don't know
        Score: 0
                                                    
        Your turn!
        Question: {question}
    """
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                Use ONLY the following pre-existing answers to answer the user's question.
                Use the answers that have the highest score (more helpful) and favor the most recent ones.
                Cite sources and return the sources of the answers as they are, do not change them.
                
                Answers: {answers}
                """,
        ),
        ("human", "{question}"),
    ]
)


# API 키 유효성 검사 함수
def validate_openai_api_key(api_key):
    try:
        openai.api_key = api_key
        openai.Model.list()
        return True
    except openai.error.AuthenticationError:
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return False


def parse_page(soup: BeautifulSoup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")


@st.cache_resource(show_spinner="Loading and splitting...")
def load_and_split_url(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_resource(show_spinner="Building vector store...")
def build_vector_store(serialized_docs):
    docs = [
        Document(page_content=doc["page_content"], metadata=doc["metadata"])
        for doc in serialized_docs
    ]
    vector_store = FAISS.from_documents(
        docs,
        OpenAIEmbeddings(
            openai_api_key=openai_api_key,
        ),
    )
    return vector_store


def serialize_documents(docs):
    """Document 객체를 JSON 직렬화 가능한 형태로 변환"""
    return [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
    ]


def load_website(url):
    docs = load_and_split_url(url)
    serialiezed_docs = serialize_documents(docs)
    vector_store = build_vector_store(serialiezed_docs)
    retriever = vector_store.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def main():

    def get_answers(inputs):
        docs = inputs["docs"]
        question = inputs["question"]
        answers_chain = answers_prompt | get_answers_llm
        return {
            "question": question,
            "answers": [
                {
                    "answer": answers_chain.invoke(
                        {"question": question, "context": doc.page_content}
                    ).content,
                    "source": doc.metadata["source"],
                    "date": doc.metadata["lastmod"],
                }
                for doc in docs
            ],
        }

    def choose_answer(inputs):
        answers = inputs["answers"]
        question = inputs["question"]
        choose_chain = choose_prompt | choose_answer_llm
        condensed = "\n\n".join(
            f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
            for answer in answers
        )
        return choose_chain.invoke(
            {
                "question": question,
                "answers": condensed,
            }
        )

    st.markdown(
        """
        # SiteGPT
                
        Ask questions about the content of the Cloudflare website.
        """
    )
    if not openai_api_key:
        st.markdown("### Step 1. Get started by adding OpenAI API Key first.")
        return
    else:

        get_answers_llm = ChatOpenAI(
            temperature=0.1,
            openai_api_key=openai_api_key,
        )
        choose_answer_llm = ChatOpenAI(
            temperature=0.1,
            openai_api_key=openai_api_key,
            streaming=True,
            callbacks=[
                ChatCallbackHandler(),
            ],
        )
        retriever = load_website(url)
        # print(f"Metadata: {retriever.vectorstore.docstore._dict}")
        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about the Cloudflare...")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            with st.chat_message("ai"):
                chain.invoke(message).content.replace("$", "\$")


with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI_API_KEY", placeholder="Add your OpenAI API Key", type="password"
    )
    if openai_api_key:
        if validate_openai_api_key(openai_api_key):
            st.success("Your API Key is valid!")
        else:
            st.error("Invalid OpenAI API Key. Please check and try again.")
    st.link_button(
        "GitHub Repo",
        "https://github.com/verobeach7/sitegpt/commits/main/",
    )


try:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    main()
except Exception as e:
    st.error("Check your OpenAI API Key or File")
    st.write(e)
