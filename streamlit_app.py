import streamlit as st
import os
from dotenv import load_dotenv
import json
from web_scraper import APEXWebScraper, create_sample_data
from rag_pipeline import GoogleAIRAGPipeline, load_scraped_data
import time
import pandas as pd

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="APEX College AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        padding: 20px 0;
        border-bottom: 3px solid #2E86AB;
        margin-bottom: 30px;
    }
    .chat-container {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .user-message {
        background: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 20px;
        margin: 10px 0;
        margin-left: 20%;
        text-align: right;
    }
    .assistant-message {
        background: #e9ecef;
        color: #333;
        padding: 15px;
        border-radius: 20px;
        margin: 10px 0;
        margin-right: 20%;
        border-left: 4px solid #28a745;
    }
    .source-info {
        background: #f1f3f4;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        font-size: 12px;
        color: #666;
    }
    .confidence-score {
        background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
        height: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Hello! 👋 I'm your APEX College AI Assistant. I have comprehensive information about APEX Group of Institutions including programs, admissions, facilities, placements, and more. How can I help you today?"
            }
        ]
    
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    
    if "scraping_status" not in st.session_state:
        st.session_state.scraping_status = "Not started"

def setup_rag_pipeline(api_key: str):
    """Setup RAG pipeline with Google AI"""
    try:
        # Initialize RAG pipeline
        rag = GoogleAIRAGPipeline(api_key)
        
        # Check if data already exists
        if not st.session_state.data_loaded:
            # Try to load existing data
            documents = load_scraped_data("apex_college_data.json")
            
            if not documents:
                st.warning("No existing data found. Using sample data...")
                documents = create_sample_data()
                
                # Save sample data
                with open("apex_college_data.json", 'w', encoding='utf-8') as f:
                    json.dump(documents, f, indent=2, ensure_ascii=False)
            
            # Process documents into RAG pipeline
            with st.spinner("Processing documents and creating embeddings..."):
                rag.process_documents(documents)
            
            st.session_state.data_loaded = True
            st.success(f"✅ Successfully loaded {len(documents)} documents!")
        
        return rag
        
    except Exception as e:
        st.error(f"❌ Error setting up RAG pipeline: {e}")
        return None

def scrape_fresh_data():
    """Scrape fresh data from APEX website"""
    try:
        with st.spinner("Scraping APEX website... This may take a few minutes."):
            scraper = APEXWebScraper(max_pages=30)
            st.session_state.scraping_status = "In progress..."
            
            # Update progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Start scraping
            data = scraper.scrape_website()
            
            if data and len(data) > 0:
                # Save data
                scraper.save_data("apex_college_data.json")
                
                # Update session state
                st.session_state.scraping_status = "Completed"
                st.session_state.data_loaded = False  # Force reload
                
                progress_bar.progress(100)
                status_text.success(f"✅ Successfully scraped {len(data)} pages!")
                
                # Show summary
                summary = scraper.get_summary()
                st.json(summary)
                
                return True
            else:
                st.warning("⚠️ Scraping returned no data. Using sample data instead.")
                sample_data = create_sample_data()
                
                with open("apex_college_data.json", 'w', encoding='utf-8') as f:
                    json.dump(sample_data, f, indent=2, ensure_ascii=False)
                
                st.session_state.scraping_status = "Used sample data"
                return True
                
    except Exception as e:
        st.error(f"❌ Error during scraping: {e}")
        st.session_state.scraping_status = f"Error: {e}"
        
        # Fallback to sample data
        st.info("Creating sample data as fallback...")
        sample_data = create_sample_data()
        
        with open("apex_college_data.json", 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        return True

def display_message(message):
    """Display a chat message"""
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header"><h1>🎓 APEX College AI Assistant</h1><p>Powered by Google AI & Advanced RAG Technology</p></div>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google AI API Key", 
            type="password", 
            value=os.getenv("GOOGLE_AI_API_KEY", ""),
            help="Get your API key from https://ai.google.dev/"
        )
        
        if not api_key:
            st.error("⚠️ Please enter your Google AI API Key to use the chatbot")
            st.info("Get your free API key from [Google AI Studio](https://ai.google.dev/)")
            st.stop()
        
        st.divider()
        
        # Data management section
        st.header("📊 Data Management")
        
        # Check current data status
        data_file_exists = os.path.exists("apex_college_data.json")
        
        if data_file_exists:
            try:
                with open("apex_college_data.json", 'r') as f:
                    current_data = json.load(f)
                st.success(f"✅ Data file exists ({len(current_data)} documents)")
                
                # Show data info
                if st.button("📋 Show Data Summary"):
                    st.json({
                        "total_documents": len(current_data),
                        "sample_titles": [doc.get('title', 'No title')[:50] + "..." for doc in current_data[:3]]
                    })
                    
            except Exception as e:
                st.error(f"❌ Error reading data file: {e}")
        else:
            st.warning("⚠️ No data file found")
        
        st.divider()
        
        # Scraping controls
        st.header("🕷️ Web Scraping")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Scrape Fresh Data", help="Scrape latest data from APEX website"):
                scrape_fresh_data()
                st.rerun()
        
        with col2:
            if st.button("📝 Use Sample Data", help="Use pre-defined sample data"):
                sample_data = create_sample_data()
                with open("apex_college_data.json", 'w', encoding='utf-8') as f:
                    json.dump(sample_data, f, indent=2, ensure_ascii=False)
                st.session_state.data_loaded = False
                st.success("✅ Sample data created!")
                st.rerun()
        
        # Show scraping status
        st.info(f"Status: {st.session_state.scraping_status}")
        
        st.divider()
        
        # System status
        st.header("⚡ System Status")
        
        if st.session_state.rag_pipeline:
            stats = st.session_state.rag_pipeline.get_collection_stats()
            st.json(stats)
        else:
            st.info("RAG pipeline not initialized")
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Initialize RAG pipeline if not done
        if st.session_state.rag_pipeline is None:
            st.session_state.rag_pipeline = setup_rag_pipeline(api_key)
        
        if st.session_state.rag_pipeline is None:
            st.error("❌ Failed to initialize RAG pipeline. Please check your API key and try again.")
            st.stop()
        
        # Chat interface
        st.subheader("💬 Chat with APEX Assistant")
        
        # Display chat messages
        for message in st.session_state.messages:
            display_message(message)
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about APEX College..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_message({"role": "user", "content": prompt})
            
            # Generate response
            with st.spinner("🤔 Thinking..."):
                try:
                    result = st.session_state.rag_pipeline.query(prompt)
                    
                    # Add assistant response
                    st.session_state.messages.append({"role": "assistant", "content": result['answer']})
                    display_message({"role": "assistant", "content": result['answer']})
                    
                    # Show sources and confidence (in sidebar or expandable section)
                    if result['sources']:
                        with st.expander("📚 Sources & Confidence"):
                            st.write(f"**Confidence Score:** {result['confidence']:.3f}")
                            st.write(f"**Retrieved Chunks:** {result['retrieved_chunks']}")
                            
                            st.write("**Sources:**")
                            for i, source in enumerate(result['sources'], 1):
                                st.write(f"{i}. **{source['title']}** (Similarity: {source['similarity']:.3f})")
                                if source['url']:
                                    st.write(f"   URL: {source['url']}")
                
                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question or contact APEX College directly at +91-7351408009."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    display_message({"role": "assistant", "content": error_msg})
            
            # Rerun to update the display
            st.rerun()
    
    with col2:
        # Quick actions and examples
        st.subheader("💡 Quick Questions")
        
        example_questions = [
            "What B.Tech programs does APEX offer?",
            "How can I apply for admission?",
            "Tell me about placement opportunities",
            "What are the campus facilities?",
            "What is the fee structure?",
            "Are there any scholarships available?",
            "What are the eligibility criteria for engineering?",
            "How is the placement record?",
            "Tell me about the faculty",
            "What extracurricular activities are available?"
        ]
        
        st.info("Click on any question below to ask it:")
        
        for question in example_questions:
            if st.button(question, key=question):
                # Add to chat
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Generate response
                try:
                    result = st.session_state.rag_pipeline.query(question)
                    st.session_state.messages.append({"role": "assistant", "content": result['answer']})
                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error. Please try again or contact APEX College at +91-7351408009."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                st.rerun()
        
        st.divider()
        
        # Contact information
        st.subheader("📞 Contact APEX")
        st.info("""
        **Phone:** +91-7351408009  
        **Email:** admissions@apex.ac.in  
        **Website:** www.apex.ac.in
        
        For immediate assistance, please contact the college directly.
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = [
                {
                    "role": "assistant", 
                    "content": "Hello! 👋 I'm your APEX College AI Assistant. How can I help you today?"
                }
            ]
            st.rerun()

if __name__ == "__main__":
    main()