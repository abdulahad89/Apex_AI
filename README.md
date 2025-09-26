# APEX College AI Chatbot with Google AI & RAG

A comprehensive AI-powered chatbot for APEX Group of Institutions built with Streamlit, Google AI (Gemini), and advanced RAG (Retrieval-Augmented Generation) capabilities.

## 🌟 Features

- **🕷️ Intelligent Web Scraping**: Automatically scrapes APEX college website for the latest information
- **🧠 Advanced RAG Pipeline**: Uses Google AI's embedding model for semantic search and retrieval
- **💬 Natural Conversations**: Powered by Google's Gemini 1.5 Flash for human-like responses
- **📊 Vector Database**: ChromaDB for efficient storage and retrieval of embedded documents
- **🎯 Smart Chunking**: Intelligent text chunking with overlap for better context preservation
- **📱 Modern UI**: Beautiful Streamlit interface with chat functionality
- **🔍 Source Attribution**: Shows confidence scores and sources for transparency
- **⚡ Real-time Processing**: Fast responses with efficient caching

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS
- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: Google AI Text Embedding 004
- **Vector DB**: ChromaDB with DuckDB backend
- **Web Scraping**: BeautifulSoup + Requests
- **Language**: Python 3.8+

## 📋 Prerequisites

- Python 3.8 or higher
- Google AI API key (free tier available)
- 4GB+ RAM recommended for vector operations

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd apex-college-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get Google AI API Key

1. Go to [Google AI Studio](https://ai.google.dev/)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your API key
GOOGLE_AI_API_KEY=your_actual_api_key_here
```

### 4. Run the Application

```bash
# Start the Streamlit app
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

### Initial Setup
1. Enter your Google AI API key in the sidebar
2. Choose data source:
   - **Scrape Fresh Data**: Scrapes latest info from APEX website
   - **Use Sample Data**: Uses pre-defined college information

### Chatting
1. Type your questions in the chat input
2. Use quick question buttons for common queries
3. View source attributions and confidence scores
4. Clear chat history when needed

### Sample Questions
- "What B.Tech programs does APEX offer?"
- "How can I apply for admission to APEX?"
- "Tell me about placement opportunities"
- "What are the campus facilities?"
- "What is the fee structure for engineering?"

## 🏗️ Architecture

```
streamlit_app.py          # Main Streamlit application
├── web_scraper.py        # APEX website scraper
├── rag_pipeline.py       # Google AI RAG implementation
├── requirements.txt      # Python dependencies
└── .env                  # Environment variables
```

### Data Flow
1. **Web Scraping**: Extracts content from APEX website
2. **Text Processing**: Cleans and chunks documents intelligently
3. **Embedding Generation**: Creates vector embeddings using Google AI
4. **Vector Storage**: Stores in ChromaDB for fast retrieval
5. **Query Processing**: Finds relevant chunks for user questions
6. **Response Generation**: Uses Gemini to create contextual answers

## 🔧 Configuration Options

### Web Scraper Settings
```python
# In web_scraper.py
scraper = APEXWebScraper(
    base_url="https://www.apex.ac.in",
    max_pages=50  # Adjust based on needs
)
```

### RAG Pipeline Settings
```python
# In rag_pipeline.py
rag = GoogleAIRAGPipeline(
    api_key=api_key,
    collection_name="apex_knowledge_base"
)

# Chunking parameters
chunk_size = 1000  # Characters per chunk
overlap = 100      # Character overlap between chunks
```

### Generation Parameters
```python
generation_config = {
    temperature: 0.3,    # Lower = more focused
    top_p: 0.9,         # Nucleus sampling
    max_output_tokens: 1024
}
```

## 📊 Data Sources

The chatbot can gather information from:

- **APEX Website Sections**:
  - Programs and courses
  - Admission procedures
  - Placement information
  - Campus facilities
  - Faculty details
  - Fee structure
  - Scholarships
  - Contact information

- **Fallback Data**: Pre-defined sample data if scraping fails

## 🎯 Key Features Explained

### Smart Web Scraping
- Respects robots.txt and implements delays
- Prioritizes important college sections
- Cleans and structures extracted content
- Handles dynamic content and navigation

### Advanced RAG Pipeline
- **Semantic Chunking**: Preserves context across chunk boundaries
- **Embedding Quality**: Uses Google's latest embedding model
- **Efficient Retrieval**: ChromaDB with optimized search
- **Context Assembly**: Intelligent context window management

### Conversation Management
- **Session Persistence**: Maintains chat history during session
- **Context Awareness**: Considers conversation flow
- **Source Attribution**: Shows where information comes from
- **Confidence Scoring**: Indicates answer reliability

## 🔍 Troubleshooting

### Common Issues

**"API Key Error"**
- Verify your Google AI API key is correct
- Check API key has proper permissions
- Ensure you haven't exceeded rate limits

**"No Data Found"**
- Click "Use Sample Data" for immediate testing
- Check internet connection for web scraping
- Verify APEX website is accessible

**"Embedding Generation Failed"**
- Check Google AI API quota
- Ensure stable internet connection
- Try reducing batch size in rag_pipeline.py

**"ChromaDB Errors"**
- Delete `chroma_db` folder to reset database
- Check disk space availability
- Ensure write permissions in project directory

### Performance Optimization

**For Better Speed**:
- Reduce `max_pages` in web scraper
- Use smaller `chunk_size` in RAG pipeline
- Enable ChromaDB persistence for faster restarts

**For Better Quality**:
- Increase `overlap` in text chunking
- Use more `n_results` in retrieval
- Fine-tune generation parameters

## 📈 Monitoring & Analytics

The app provides built-in monitoring:

- **Data Statistics**: Document count, chunk distribution
- **Query Analytics**: Confidence scores, source relevance
- **System Status**: API health, database status
- **Performance Metrics**: Response times, success rates

## 🔐 Security Notes

- API keys are stored in environment variables
- No sensitive data is logged or cached
- Web scraping respects website policies
- Vector database is stored locally

## 🚀 Deployment Options

### Local Development
```bash
streamlit run streamlit_app.py
```

### Streamlit Cloud
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Add `GOOGLE_AI_API_KEY` to secrets
4. Deploy application

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

## 📞 Support

For technical issues:
- Check troubleshooting section above
- Review error messages in Streamlit logs
- Ensure all dependencies are correctly installed

For APEX College information:
- **Phone**: +91-7351408009
- **Email**: admissions@apex.ac.in
- **Website**: www.apex.ac.in

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google AI for providing powerful LLM and embedding APIs
- ChromaDB team for excellent vector database
- Streamlit for the amazing web framework
- BeautifulSoup for web scraping capabilities
- APEX Group of Institutions for being our use case

---

**Built with ❤️ for APEX College students and prospective applicants**