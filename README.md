# RAG Tutor - Society and the Engineer Course Assistant

## Overview
RAG Tutor is an AI-powered tutoring system designed specifically for the "Society and the Engineer" (ENG3004) course. It uses Retrieval-Augmented Generation (RAG) to provide contextually relevant responses to student queries and feedback on their work by referencing course materials including lectures, tutorials, assignments, and example submissions.

## Features
- Interactive web interface built with Streamlit
- Intelligent document processing for various file formats (PDF, DOCX, PPTX)
- Context-aware responses using course materials
- Automated feedback on student work
- Quality-aware retrieval system that considers example submissions of different grades
- Persistent vector storage for efficient retrieval

## Prerequisites
- Python 3.8+
- Gemini API Key
   - [How to get a Gemini API key?](https://ai.google.dev/gemini-api/docs?gad_source=1&gclid=Cj0KCQiAvvO7BhC-ARIsAGFyToWGI4cqFsQsnqj9i3OzoB1JXuv76beex2eRxKUONigeEKsicCEM944aAktYEALw_wcB)
- Course materials organized in the specified folder structure

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-tutor
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with:
```
GOOGLE_API_KEY=your_google_api_key_here
```

## Folder Structure
The system expects course materials to be organized as follows:
```
ENG3004_222/
├── Lecture Notes/
├── Tutorial sheets/
├── Assignment1/
│   ├── Poor/
│   ├── Average/
│   ├── Good/
│   └── assignment1.pdf
├── Assignment2/
│   ├── Poor/
│   ├── Average/
│   ├── Good/
│   └── assignment2.pdf
└── rubric.docx
```

## Usage

1. Start the application:
```bash
streamlit run ragTestBeta.py
```

2. Access the web interface at `http://localhost:8501`

3. Use the interface to:
   - Ask questions about course content
   - Submit work for feedback
   - Get contextual responses based on course materials

## Key Components

### DocumentProcessor
- Handles reading and processing of various document formats
- Implements chunking strategies for optimal retrieval
- Maintains source tracking for accurate citations

### VectorStore
- Manages persistent storage of document embeddings
- Implements efficient save and load operations
- Handles atomic updates to prevent corruption

### RAGTutor
- Core class that coordinates all functionality
- Implements retrieval and response generation
- Manages integration with Gemini Pro for response generation

## Technical Details

### Document Processing
- PDFs: Extracts text with page tracking
- PPTX: Processes slides while maintaining slide numbers
- DOCX: Processes documents with section tracking
- All documents are chunked into manageable segments for retrieval

### Embedding and Retrieval
- Uses SentenceTransformer ('all-MiniLM-L6-v2') for embedding generation
- Implements cosine similarity for relevant context retrieval
- Maintains metadata for source attribution

### Response Generation
- Uses Google's Gemini Pro model for response generation
- Incorporates retrieved context and source information
- Provides structured feedback based on query type

## Error Handling
- Comprehensive error logging system
- Graceful handling of file processing errors
- User-friendly error messages in the UI
