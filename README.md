# RAG System on "Leave No Context Behind" Paper

## Overview
This project implements the RAG system based on the "Leave No Context Behind" paper. The system combines retrieval-based and generation-based methods to provide accurate and contextually relevant responses. It comprehends context from a given document and generates precise answers to user queries.

## Key Features

### 1. Document Loader
- Utilizes PyPDFLoader to efficiently handle PDF documents, extracting text and metadata for processing.

### 2. Text Splitter
- Employs NLTKTextSplitter to segment documents into smaller, manageable parts, enhancing understanding of complex content.

### 3. Vector Store
- Utilizes Chroma as the vector store for storing and retrieving document embeddings, capturing semantic meaning for accurate responses.

### 4. Generative AI
- Leverages GoogleGenerativeAI to craft responses based on context, employing advanced NLP techniques for precision.

### 5. User Interface
- Powered by Streamlit, offering a seamless experience for users to input questions and receive context-aware responses in real-time. Intuitive and user-friendly interface facilitates easy interaction.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `streamlit run app.py`

## Contributing
1. Fork the repository
2. Make your contributions
3. Submit a pull request

## License
This project is licensed under the [MIT License](LICENSE).

---

