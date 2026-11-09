Secure PDF QA – Project workflow

Explanation of each stage

<img width="241" height="617" alt="image" src="https://github.com/user-attachments/assets/0941c85c-3d81-475f-a83c-76d4493b1578" />


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
