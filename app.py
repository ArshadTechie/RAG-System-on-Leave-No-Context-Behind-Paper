import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate
from IPython.display import Markdown as md
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\Users\arsha\Downloads\2404.07143.pdf")
pages = loader.load_and_split()

from langchain_text_splitters import NLTKTextSplitter

text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)

chunks = text_splitter.split_documents(pages)

with open(file=r"C:\rag\generative.txt", mode='r') as f:
    api_key = f.read()

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key, 
                                               model="models/embedding-001")

# Initialize Chroma DB connection

db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
db.persist()
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

# Create a retriever
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# Define chat template
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the question from the user and answer if you have specific information related to the question."""),
    HumanMessagePromptTemplate.from_template("""Answer the following question: {question}Answer: """)
])


# Initialize chat model
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key)

# Initialize output parser
output_parser = StrOutputParser()

# Define function to format documents for display
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the conversation chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

# Streamlit UI
st.title("The RAG Systems Journey Beyond Leave No Context Behind")
st.write("This application leverages the RAG (Retrieve, Answer, Generate) system to provide answers to questions, drawing inspiration from the principles outlined in the 'Leave No Context Behind' paper.")

# Input question
input_text = st.text_input("Enter your question:")

# Button to generate answer
if st.button("Generate Answer"):
    with st.spinner('Generating answer...'):
        answer = rag_chain.invoke(input_text)
    st.success("Answer:")
    st.markdown(answer)

