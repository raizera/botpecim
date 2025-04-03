# PECIM's Chatbot

## Overview
PECIM's Chatbot is an interactive application developed to assist graduate students from the Science and Mathematics Education Program (PECIM) at UNICAMP with questions about the course. The system uses generative artificial intelligence (Google Gemini) to answer questions based on information contained on the program's official website.

## Features
- Intuitive chat interface built with Streamlit
- Processing and analysis of program PDF documents
- Semantic search to find relevant information
- Detailed answers to student questions
- Multilingual support (Portuguese and English)
- Cache system for frequent answers

## Technologies Used
- Python 3.7+
- Streamlit for web interface
- Google Generative AI (Gemini 1.5 Flash)
- LangChain for document processing and RAG chain creation
- FAISS for vector storage and search
- PDFPlumber for PDF text extraction

## Requirements
```
python>=3.7
streamlit
pdfplumber
langchain
langchain-google-genai
faiss-cpu
python-dotenv
google-generativeai
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/pecim-chatbot.git
cd pecim-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

4. Add the PECIM PDF file to the project root (default: "database-site-pecim.pdf").

## How to Run
Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default browser, usually at http://localhost:8501

## Usage
1. When starting the application for the first time, it will process the PDF and create the search index
2. Type your questions about PECIM in the text input box at the bottom
3. The chatbot will provide answers based on the PDF content
4. You can switch between Portuguese and English using the selector in the sidebar
5. Use the "Clear conversation history" button to start a new session

## Limitations
- Answers are based only on the provided documents and may not include more recent information not present in the PDF
- Like any AI system, it may generate inaccurate or incomplete answers
- The system does not answer questions unrelated to PECIM

## Project Files
- `app.py`: Main file containing the application code
- `database-site-pecim.pdf`: Base document with PECIM information
- `faiss_index/`: Directory automatically created to store the vector index
- `response_cache.json`: Response cache to improve performance

## Contributions
Contributions are welcome! If you find bugs or have suggestions for improvements, please:
1. Open an issue describing the problem or suggestion
2. Submit a pull request with your changes

## Contact
For questions and suggestions: r147725@dac.unicamp.br

## Legal Notice
This is an unofficial application developed by a PECIM student for testing purposes. The information provided by the chatbot may contain inaccuracies and should not replace consulting the program's official sources.
