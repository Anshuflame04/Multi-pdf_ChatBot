# **PDF-Based Question Answering using LangChain & Gemini**  
🚀 A **Retrieval-Augmented Generation (RAG) System** that allows users to ask questions based on **multiple PDF documents**. It uses **Google Gemini**, **LangChain**, and **ChromaDB** to process and retrieve the most relevant answers.  

---

## **📝 Features**  
✅ **Upload multiple PDFs** and extract text automatically.  
✅ **Chunk large documents** for better processing.  
✅ **Vectorize text using Google Gemini embeddings.**  
✅ **Store & retrieve document chunks using ChromaDB.**  
✅ **Answer questions based on document content using Gemini LLM.**  

---

## **⚡ Tech Stack**  
- **Python** 🐍  
- **LangChain** (For text processing & retrieval)  
- **Google Gemini API** (For embeddings & LLM)  
- **ChromaDB** (For vector storage)  
- **PyPDFLoader** (For PDF parsing)  

---

## **📜 Code Explanation**  

### **1️⃣ Load Environment Variables**  
```python
from dotenv import load_dotenv
load_dotenv()
```
- Loads the `.env` file, which contains the **Google API Key** required for Gemini.  

### **2️⃣ Load Multiple PDFs from a Folder**  
```python
import os
from langchain_community.document_loaders import PyPDFLoader

pdf_folder = "pdfs"  # Change this to your actual folder path
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

docs = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    docs.extend(loader.load())  # Extracts text from each PDF
```
- Scans the **pdfs/** folder and loads all PDFs dynamically.  
- Uses **PyPDFLoader** to extract text from each PDF.  

### **3️⃣ Split Text into Smaller Chunks**  
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
split_docs = text_splitter.split_documents(docs)
```
- Splits the extracted text into **chunks of 1000 characters**.  
- Helps in **better embedding and retrieval** later.  

### **4️⃣ Initialize Google Gemini Embeddings**  
```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
```
- Converts document text into **vector embeddings** for similarity search.  

### **5️⃣ Store Documents in ChromaDB**  
```python
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)
```
- Stores document chunks in **ChromaDB**, a fast vector search database.  
- Helps in quickly **retrieving relevant document parts** for a given query.  

### **6️⃣ Create a Retriever for Similarity Search**  
```python
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
```
- Retrieves **top 10 most relevant document chunks** for a given user query.  

### **7️⃣ Initialize Gemini LLM for Answer Generation**  
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)
```
- Uses **Google Gemini LLM** to generate answers based on retrieved content.  
- `temperature=0.3` ensures **concise & accurate responses**.  

### **8️⃣ Define the System Prompt**  
```python
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following retrieved context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise."
    "\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
```
- Defines the **system behavior** and formatting of the prompt.  
- Ensures **brief, to-the-point answers** using retrieved context.  

### **9️⃣ Create the Retrieval & Response Chains**  
```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```
- **Retrieves relevant document parts** using ChromaDB.  
- **Passes the retrieved context to Gemini LLM** for final answer generation.  

### **🔟 Get User Query & Generate Response**  
```python
user_query = input("\n Enter your question? ")

response = rag_chain.invoke({"input": user_query})

print("\n💡 Answer:", response["answer"])
```
- Takes **user input** (a question).  
- Passes it through the **RAG pipeline**.  
- **Prints the generated answer** based on the document contents.  
