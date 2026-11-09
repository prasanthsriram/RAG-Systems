Secure PDF QA – Project workflow:
---------------------------------

Explanation of each stage

<img width="1654" height="4236" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/08f5ec1e-7814-4a58-ab4b-de6d52b00e4c" />

1. PDF upload and text extraction

User uploads a PDF. pypdf extracts its text locally.

2. Sensitive data detection and masking

Presidio detects sensitive entities and replaces their values while preserving surrounding text.


3. Text chunking and vector storage

LangChain splits redacted text into chunks, Gemini creates embeddings, and ChromaDB stores the vectors.

4. Question processing and retrieval

Presidio scans the user's question. ChromaDB retrieves relevant redacted document chunks.

5. Gemini answer generation

LangChain sends the safe context, security summary and safe question to Gemini.

6. Final security validation

Presidio scans the generated answer before Streamlit displays it to the user.



Technologies used:
-----------------
Technology	        Purpose
Python	            Main programming language
Streamlit	          User interface
PyPDF	              PDF text extraction
Microsoft Presidio	Sensitive data detection and masking
LangChain	          QA pipeline orchestration
Gemini Embeddings	  Text vectorization
ChromaDB	          Vector database
Gemini LLM	        Answer generation

Core security principle:
-----------------------
Original sensitive information stays local. Only redacted document content, redacted questions and non-sensitive detection summaries should reach external AI services.
