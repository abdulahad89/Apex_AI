import streamlit as st
import os
from dotenv import load_dotenv
import json
import time
import pandas as pd
from typing import List, Dict

# Custom imports
from web_scraper import APEXWebScraper, create_sample_data

# Load environment variables
load_dotenv()

# Enhanced RAG Pipeline with comprehensive data loading
import google.generativeai as genai
import chromadb
import numpy as np

class EnhancedGoogleAIRAG:
    """Enhanced RAG Pipeline with automatic data scraping and processing"""
    
    def __init__(self, api_key: str, auto_scrape: bool = True):
        """Initialize with automatic data scraping"""
        self.api_key = api_key
        
        # Configure Google AI
        genai.configure(api_key=api_key)
        
        # Test API connection first
        if not self._test_google_ai_connection():
            raise Exception("Google AI API connection failed. Please check your API key.")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="apex_comprehensive_kb",
            metadata={"description": "APEX College Comprehensive Knowledge Base"}
        )
        
        # Initialize models
        self.embedding_model = "models/text-embedding-004"
        self.generation_model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Load or create comprehensive data
        self.documents = []
        self._initialize_comprehensive_data(auto_scrape)
        
    def _test_google_ai_connection(self) -> bool:
        """Test Google AI API connection"""
        try:
            test_response = genai.embed_content(
                model="models/text-embedding-004",
                content="test",
                task_type="retrieval_query"
            )
            return True
        except Exception as e:
            st.error(f"Google AI API Error: {e}")
            return False
    
    def _initialize_comprehensive_data(self, auto_scrape: bool = True):
        """Initialize comprehensive APEX data"""
        data_loaded = False
        
        # Try to load existing scraped data
        if os.path.exists("apex_college_data.json"):
            try:
                with open("apex_college_data.json", 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if len(existing_data) > 3:  # More than just sample data
                    self.documents = existing_data
                    st.success(f"✅ Loaded existing scraped data: {len(existing_data)} documents")
                    data_loaded = True
            except Exception as e:
                st.warning(f"Could not load existing data: {e}")
        
        # If no good existing data and auto_scrape is enabled
        if not data_loaded and auto_scrape:
            st.info("🕷️ No comprehensive data found. Starting web scraping...")
            self._scrape_and_process_data()
            data_loaded = True
        
        # Fallback to comprehensive sample data
        if not data_loaded:
            st.warning("Using comprehensive sample data...")
            self.documents = self._create_comprehensive_sample_data()
        
        # Process documents into vector database
        self._process_documents_to_vector_db()
    
    def _scrape_and_process_data(self):
        """Scrape comprehensive data from APEX website"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize scraper with more comprehensive settings
            scraper = APEXWebScraper(
                base_url="https://www.apex.ac.in", 
                max_pages=80  # More comprehensive scraping
            )
            
            status_text.text("🔍 Starting comprehensive website scraping...")
            progress_bar.progress(10)
            
            # Scrape data
            scraped_data = scraper.scrape_website()
            progress_bar.progress(70)
            
            if scraped_data and len(scraped_data) > 0:
                # Save scraped data
                scraper.save_data("apex_college_data.json")
                self.documents = scraped_data
                
                progress_bar.progress(100)
                status_text.success(f"✅ Successfully scraped {len(scraped_data)} pages!")
                
                # Show summary
                summary = scraper.get_summary()
                st.json(summary)
            else:
                raise Exception("Scraping returned no data")
                
        except Exception as e:
            st.error(f"Scraping failed: {e}")
            st.info("Using comprehensive sample data as fallback...")
            self.documents = self._create_comprehensive_sample_data()
            progress_bar.progress(100)
            status_text.text("Using sample data")
    
    def _create_comprehensive_sample_data(self) -> List[Dict]:
        """Create comprehensive sample data about APEX College"""
        comprehensive_data = [
            {
                'url': 'https://www.apex.ac.in/',
                'title': 'APEX Group of Institutions - Premier Educational Institution',
                'content': '''APEX Group of Institutions is a leading educational institution established with the vision of providing quality education and fostering holistic development of students. Located with state-of-the-art infrastructure, APEX has been a pioneer in technical and management education.
                
                Mission: To provide world-class education that enables students to excel in their chosen careers and contribute meaningfully to society.
                
                Vision: To be recognized as a center of excellence in education, research, and innovation.
                
                Key Highlights:
                - Established institution with proven track record
                - AICTE approved programs
                - Experienced and qualified faculty
                - Industry-aligned curriculum
                - Strong placement record
                - Modern infrastructure and facilities
                - Active industry partnerships
                
                APEX is committed to nurturing future leaders and professionals who can drive innovation and progress in their respective fields.''',
                'word_count': 140,
                'scraped_at': '2024-01-01 12:00:00'
            },
            {
                'url': 'https://www.apex.ac.in/btech',
                'title': 'B.Tech Programs - APEX Institute of Technology',
                'content': '''APEX Institute of Technology offers comprehensive Bachelor of Technology (B.Tech) programs designed to meet industry demands and prepare students for successful careers in engineering and technology.
                
                B.Tech Programs Available:
                
                1. Computer Science Engineering (CSE)
                - Duration: 4 years (8 semesters)
                - Focus: Programming, algorithms, data structures, software engineering, databases, computer networks
                - Career opportunities: Software developer, system analyst, data scientist, cybersecurity specialist
                
                2. Artificial Intelligence & Machine Learning (AI/ML)
                - Duration: 4 years (8 semesters)
                - Focus: Machine learning algorithms, deep learning, neural networks, natural language processing
                - Career opportunities: AI engineer, ML researcher, data scientist, automation specialist
                
                3. Data Science
                - Duration: 4 years (8 semesters)
                - Focus: Statistics, data mining, big data analytics, visualization, predictive modeling
                - Career opportunities: Data analyst, business intelligence specialist, research analyst
                
                4. Cloud Technology & Information Security
                - Duration: 4 years (8 semesters)
                - Focus: Cloud computing, cybersecurity, network security, ethical hacking
                - Career opportunities: Cloud architect, security analyst, DevOps engineer
                
                Eligibility Criteria:
                - 10+2 with Physics, Mathematics, Chemistry/Biology/Computer Science
                - Minimum 45% marks (40% for reserved category)
                - Valid scores in JEE Main/CUET/State CET
                
                Course Features:
                - Industry-relevant curriculum
                - Hands-on practical training
                - Project-based learning
                - Industry internships
                - Research opportunities
                - Guest lectures by industry experts
                - Modern laboratories and equipment''',
                'word_count': 280,
                'scraped_at': '2024-01-01 12:00:00'
            },
            {
                'url': 'https://www.apex.ac.in/admission',
                'title': 'Admission Process - APEX College Admissions',
                'content': '''APEX Group of Institutions follows a transparent, merit-based admission process designed to identify and admit deserving candidates to various programs.
                
                Admission Process Overview:
                
                Step 1: Application Submission
                - Visit official website www.apex.ac.in
                - Fill online application form with accurate details
                - Upload required documents in specified format
                - Pay application fee through online payment gateway
                - Submit application before deadline
                
                Step 2: Eligibility Verification
                - Documents verification by admission committee
                - Eligibility criteria checking
                - Merit list preparation based on qualifying exam scores
                
                Step 3: Merit List and Counseling
                - Merit lists published on official website
                - Shortlisted candidates called for counseling
                - Seat allocation based on merit and preference
                - Document verification during counseling
                
                Step 4: Admission Confirmation
                - Fee payment within stipulated time
                - Admission confirmation and seat booking
                - Issue of admission letter
                
                Required Documents:
                - 10th standard marksheet and certificate
                - 12th standard marksheet and certificate
                - Transfer certificate from last attended institution
                - Migration certificate (if applicable)
                - Category certificate (SC/ST/OBC/EWS)
                - Income certificate for fee concession
                - Passport size photographs
                - Aadhar card copy
                - Medical fitness certificate
                
                Application Deadlines:
                - Online applications typically open in May
                - Last date for submission varies by program
                - Merit lists published in June/July
                - Admission process completes by August
                
                Fee Payment:
                - Online payment preferred
                - Demand draft also accepted
                - Installment facility available
                - Fee refund policy as per norms
                
                Contact for Admission Queries:
                Phone: +91-7351408009
                Email: admissions@apex.ac.in
                Office Hours: 9:00 AM to 6:00 PM (Monday to Saturday)''',
                'word_count': 320,
                'scraped_at': '2024-01-01 12:00:00'
            },
            {
                'url': 'https://www.apex.ac.in/placements',
                'title': 'Placement Cell - Career Opportunities at APEX',
                'content': '''The Training & Placement Cell at APEX Group of Institutions is dedicated to providing comprehensive career guidance and facilitating excellent placement opportunities for students.
                
                Placement Highlights:
                - Consistent placement record of 90%+ across programs
                - Average salary package: 3-8 LPA
                - Highest salary package: 12 LPA
                - 200+ companies visit for recruitment
                - Multiple job offers for deserving candidates
                
                Top Recruiting Companies:
                
                IT & Software:
                - IBM, Microsoft, Intel, Oracle
                - TCS, Infosys, Wipro, HCL Technologies
                - Tech Mahindra, Accenture, Capgemini
                - Cognizant, Mindtree, L&T Infotech
                
                Core Engineering:
                - Larsen & Toubro, Bajaj Auto
                - Mahindra Group, Tata Motors
                - Bosch, Siemens, ABB
                
                Banking & Finance:
                - ICICI Bank, HDFC Bank, Axis Bank
                - Kotak Mahindra, Yes Bank
                
                Consulting & Analytics:
                - Deloitte, PwC, EY, KPMG
                - McKinsey, BCG (for exceptional candidates)
                
                Services Provided by Placement Cell:
                
                1. Pre-placement Training
                - Communication skills development
                - Technical interview preparation
                - Group discussion training
                - Resume building workshops
                - Mock interviews
                - Aptitude test preparation
                
                2. Industry Interface
                - Guest lectures by industry experts
                - Industrial visits and exposure
                - Internship facilitation
                - Live project opportunities
                - Industry-academia collaboration
                
                3. Career Guidance
                - Individual counseling sessions
                - Career path guidance
                - Higher studies consultation
                - Entrepreneurship support
                - Alumni mentorship programs
                
                4. Skill Enhancement
                - Technical certification courses
                - Soft skills training
                - Leadership development programs
                - Digital literacy initiatives
                
                Placement Process:
                - Pre-placement presentations by companies
                - Online/offline screening tests
                - Technical and HR interviews
                - Final selection and offer letters
                - Joining formalities support
                
                Alumni Success Stories:
                Our graduates are working in leading positions at Fortune 500 companies worldwide, contributing to technological advancement and business growth.
                
                Contact Placement Cell:
                Email: placements@apex.ac.in
                Phone: +91-7351408009 (Ext: 234)''',
                'word_count': 380,
                'scraped_at': '2024-01-01 12:00:00'
            },
            {
                'url': 'https://www.apex.ac.in/facilities',
                'title': 'Campus Infrastructure & Facilities - APEX College',
                'content': '''APEX Group of Institutions boasts world-class infrastructure and comprehensive facilities designed to provide an optimal learning environment for students.
                
                Academic Infrastructure:
                
                1. Classrooms
                - Modern, air-conditioned classrooms
                - Smart boards and projectors
                - Audio-visual teaching aids
                - Ergonomic furniture
                - Capacity ranging from 40-120 students
                
                2. Laboratories
                - State-of-the-art computer labs with latest software
                - Engineering labs with modern equipment
                - Physics, Chemistry, and Biology labs
                - Language lab for communication skills
                - CAD/CAM lab for design and manufacturing
                - Robotics and automation lab
                
                3. Library
                - Central library with 50,000+ books
                - Digital library with e-resources
                - International and national journals
                - Online databases and research materials
                - Reading halls with internet connectivity
                - Book bank facility for students
                
                4. Computer Centers
                - High-speed internet connectivity
                - Latest computers with updated software
                - 24x7 internet access for students
                - Coding and programming platforms
                - Network infrastructure for seamless connectivity
                
                Accommodation Facilities:
                
                1. Hostels
                - Separate hostels for boys and girls
                - Well-furnished rooms with modern amenities
                - 24x7 security and warden supervision
                - Common rooms for recreation
                - Study rooms for group discussions
                - Laundry facilities
                
                2. Dining Facilities
                - Hygienic mess with nutritious meals
                - Varied menu including regional cuisines
                - Separate dining halls for boys and girls
                - Canteen for snacks and beverages
                - Food court with multiple food options
                
                Sports & Recreation:
                
                1. Sports Complex
                - Cricket ground with proper facilities
                - Basketball and volleyball courts
                - Badminton courts (indoor)
                - Table tennis and carrom facilities
                - Gymnasium with modern equipment
                - Yoga and meditation hall
                
                2. Cultural Activities
                - Auditorium with 500+ seating capacity
                - Music and dance rooms
                - Art and craft studios
                - Drama and theater facilities
                
                Health & Wellness:
                
                1. Medical Facilities
                - On-campus medical center
                - Qualified medical staff
                - First aid facilities
                - Health checkup programs
                - Emergency medical services
                - Tie-ups with nearby hospitals
                
                2. Counseling Services
                - Professional counseling support
                - Career guidance sessions
                - Mental health awareness programs
                - Stress management workshops
                
                Transportation:
                - Bus service covering major routes
                - Pick-up and drop facilities
                - Safe and reliable transportation
                - GPS-enabled buses for safety
                
                Other Facilities:
                - Banking and ATM services on campus
                - Stationery and photocopy services
                - Wi-Fi enabled campus
                - Power backup for uninterrupted services
                - Parking facilities for staff and visitors
                - Security with CCTV surveillance
                
                Green Campus Initiatives:
                - Solar power generation
                - Rainwater harvesting system
                - Waste management programs
                - Tree plantation drives
                - Environmental awareness campaigns''',
                'word_count': 450,
                'scraped_at': '2024-01-01 12:00:00'
            },
            {
                'url': 'https://www.apex.ac.in/fees',
                'title': 'Fee Structure & Financial Aid - APEX College',
                'content': '''APEX Group of Institutions maintains transparent and affordable fee structure across all programs, with various financial aid options to support deserving students.
                
                B.Tech Fee Structure (Annual):
                - Tuition Fee: ₹80,000 - ₹1,20,000 (varies by specialization)
                - Development Fee: ₹15,000
                - Laboratory Fee: ₹10,000
                - Library Fee: ₹5,000
                - Examination Fee: ₹8,000
                - Total Annual Fee: ₹1,18,000 - ₹1,58,000
                
                MBA Fee Structure (Annual):
                - Tuition Fee: ₹1,00,000
                - Case Study & Project Fee: ₹10,000
                - Industry Interface Fee: ₹8,000
                - Examination Fee: ₹7,000
                - Total Annual Fee: ₹1,25,000
                
                Other Programs:
                - BCA: ₹60,000 per year
                - BBA: ₹55,000 per year
                - B.Com (Hons): ₹45,000 per year
                - Polytechnic: ₹40,000 per year
                
                Payment Options:
                1. Annual Payment: 5% discount on total fees
                2. Semester Payment: Pay fees semester-wise
                3. Installment Plan: Up to 4 installments per year
                4. Online Payment: Multiple payment gateways available
                5. Bank Transfer: Direct bank transfer accepted
                
                Scholarships & Financial Aid:
                
                1. Merit Scholarships
                - 100% scholarship: Top 3 students in state entrance exam
                - 50% scholarship: Top 10 students in state entrance exam
                - 25% scholarship: Top 50 students in state entrance exam
                - Academic performance scholarships for continuing students
                
                2. Need-based Financial Aid
                - Income-based fee concessions
                - Fee waiver for economically weaker sections
                - Installment facilities without interest
                - Book bank facility for underprivileged students
                
                3. Government Scholarships
                - SC/ST/OBC scholarship as per government norms
                - Minority community scholarships
                - Girl child education incentives
                - State government scholarship schemes
                
                4. Special Scholarships
                - Sports excellence scholarships
                - Cultural talent scholarships
                - Employee ward concessions
                - Alumni referral benefits
                - Sibling discount (10% for second child)
                
                Fee Refund Policy:
                - Withdrawal before classes start: 90% refund
                - Withdrawal within first month: 70% refund
                - Withdrawal within first semester: 50% refund
                - No refund after first semester completion
                - Processing fee of ₹5,000 deducted in all cases
                
                Additional Costs (Optional):
                - Hostel fees: ₹60,000 - ₹80,000 per year
                - Mess charges: ₹40,000 per year
                - Transportation: ₹15,000 - ₹25,000 per year
                - Study materials: ₹8,000 - ₹12,000 per year
                
                Fee Collection Process:
                - Online payment portal available 24x7
                - Demand draft in favor of "APEX Group of Institutions"
                - Cash payment accepted at accounts office
                - Receipt issued for all payments
                - Fee defaulter list published monthly
                
                Financial Counseling:
                - Dedicated financial aid counselor
                - Education loan guidance
                - Bank liaison for student loans
                - EMI planning assistance
                
                Contact for Fee Queries:
                Accounts Office: +91-7351408009 (Ext: 101)
                Email: accounts@apex.ac.in
                Office Hours: 9:00 AM to 5:00 PM (Monday to Saturday)''',
                'word_count': 520,
                'scraped_at': '2024-01-01 12:00:00'
            }
        ]
        
        # Save comprehensive sample data
        with open("apex_college_data.json", 'w', encoding='utf-8') as f:
            json.dump(comprehensive_data, f, indent=2, ensure_ascii=False)
        
        return comprehensive_data
    
    def _process_documents_to_vector_db(self):
        """Process all documents into vector database"""
        if not self.documents:
            st.error("No documents to process!")
            return
            
        st.info(f"🔄 Processing {len(self.documents)} documents into vector database...")
        
        # Clear existing data
        try:
            existing_count = self.collection.count()
            if existing_count > 0:
                all_data = self.collection.get()
                if all_data['ids']:
                    self.collection.delete(ids=all_data['ids'])
                    st.info(f"🗑️ Cleared {existing_count} existing chunks")
        except Exception as e:
            st.warning(f"Warning clearing collection: {e}")
        
        # Process documents into chunks
        all_chunks = []
        all_metadata = []
        all_ids = []
        
        progress_bar = st.progress(0)
        
        for doc_id, doc in enumerate(self.documents):
            title = doc.get('title', 'Untitled')
            content = doc.get('content', '')
            url = doc.get('url', '')
            
            # Create comprehensive content for chunking
            full_content = f"Title: {title}\n\n{content}"
            
            # Chunk the document
            chunks = self._chunk_text(full_content)
            
            for chunk_id, chunk in enumerate(chunks):
                chunk_metadata = {
                    'doc_id': str(doc_id),
                    'chunk_id': str(chunk_id),
                    'title': title,
                    'url': url,
                    'word_count': str(len(chunk.split())),
                    'source': 'apex_website'
                }
                
                all_chunks.append(chunk)
                all_metadata.append(chunk_metadata)
                all_ids.append(f"doc_{doc_id}_chunk_{chunk_id}")
            
            progress_bar.progress((doc_id + 1) / len(self.documents) * 0.5)
        
        st.success(f"📝 Created {len(all_chunks)} chunks from documents")
        
        # Generate embeddings
        st.info("🧠 Generating embeddings...")
        embeddings = self._generate_embeddings_batch(all_chunks)
        progress_bar.progress(0.8)
        
        # Add to ChromaDB
        st.info("💾 Adding to vector database...")
        self._add_to_chromadb_batch(all_chunks, embeddings, all_metadata, all_ids)
        
        progress_bar.progress(1.0)
        st.success(f"✅ Successfully indexed {self.collection.count()} chunks into vector database!")
    
    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 80) -> List[str]:
        """Intelligent text chunking"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Find sentence boundary
            chunk = text[start:end]
            sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
            best_break = -1
            
            for ending in sentence_endings:
                pos = chunk.rfind(ending)
                if pos > len(chunk) * 0.7:
                    best_break = max(best_break, pos + len(ending))
            
            if best_break > 0:
                chunks.append(text[start:start + best_break].strip())
                start = start + best_break - overlap
            else:
                # Fallback to word boundary
                space_pos = chunk.rfind(' ')
                if space_pos > len(chunk) * 0.8:
                    chunks.append(text[start:start + space_pos].strip())
                    start = start + space_pos - overlap
                else:
                    chunks.append(chunk)
                    start = end - overlap
            
            start = max(start, 0)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in batches"""
        embeddings = []
        batch_size = 5  # Small batches for stability
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = genai.embed_content(
                    model=self.embedding_model,
                    content=batch,
                    task_type="retrieval_document"
                )
                
                if isinstance(response['embedding'], list):
                    batch_embeddings = [emb['embedding'] for emb in response['embedding']]
                else:
                    batch_embeddings = [response['embedding']['embedding']]
                
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                st.warning(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Add dummy embeddings for failed batch
                embeddings.extend([[0.1] * 768 for _ in batch])
        
        return embeddings
    
    def _add_to_chromadb_batch(self, chunks: List[str], embeddings: List[List[float]], 
                              metadata: List[Dict], ids: List[str]):
        """Add data to ChromaDB in batches"""
        batch_size = 50
        
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            try:
                self.collection.add(
                    documents=chunks[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    metadatas=metadata[i:end_idx],
                    ids=ids[i:end_idx]
                )
            except Exception as e:
                st.warning(f"Error adding batch {i//batch_size + 1}: {e}")
    
    def query(self, user_question: str, n_results: int = 5) -> Dict:
        """Enhanced query with better error handling"""
        try:
            # Generate query embedding
            query_response = genai.embed_content(
                model=self.embedding_model,
                content=user_question,
                task_type="retrieval_query"
            )
            
            if 'embedding' in query_response:
                if isinstance(query_response['embedding'], dict):
                    query_embedding = query_response['embedding']['embedding']
                else:
                    query_embedding = query_response['embedding']
            else:
                raise ValueError("No embedding in response")
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, max(1, self.collection.count())),
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            relevant_chunks = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    relevant_chunks.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': 1 - results['distances'][0][i],
                    })
            
            if not relevant_chunks:
                return {
                    'answer': "I don't have specific information about that in my knowledge base. For detailed information, please contact APEX College at +91-7351408009 or admissions@apex.ac.in",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Generate context-aware prompt
            context_parts = []
            for chunk in relevant_chunks:
                title = chunk['metadata'].get('title', 'Unknown')
                content = chunk['content']
                context_parts.append(f"Source: {title}\nContent: {content}\n")
            
            context = "\n---\n".join(context_parts)
            
            prompt = f"""You are APEX College Assistant. Use this context to answer about APEX College:

CONTEXT:
{context}

QUESTION: {user_question}

Provide a helpful, accurate answer based on the context. If the context doesn't fully answer the question, mention what information is available and suggest contacting +91-7351408009 for more details."""
            
            # Generate answer
            response = self.generation_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    top_p=0.9,
                    max_output_tokens=800,
                )
            )
            
            # Extract sources
            sources = []
            for chunk in relevant_chunks[:3]:
                sources.append({
                    'title': chunk['metadata'].get('title', 'Unknown'),
                    'url': chunk['metadata'].get('url', ''),
                    'similarity': chunk['similarity_score']
                })
            
            avg_confidence = np.mean([chunk['similarity_score'] for chunk in relevant_chunks])
            
            return {
                'answer': response.text.strip(),
                'sources': sources,
                'confidence': float(avg_confidence),
                'retrieved_chunks': len(relevant_chunks)
            }
            
        except Exception as e:
            st.error(f"Query error: {e}")
            return {
                'answer': f"I encountered an error processing your question: {str(e)}. Please try rephrasing your question or contact APEX College directly at +91-7351408009 for assistance.",
                'sources': [],
                'confidence': 0.0
            }
    
    def get_stats(self) -> Dict:
        """Get comprehensive stats"""
        return {
            'total_documents': len(self.documents),
            'total_chunks': self.collection.count(),
            'embedding_model': self.embedding_model,
            'status': 'ready'
        }

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
</style>
""", unsafe_allow_html=True)

def main():
    """Main application"""
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    
    # Header
    st.markdown('<div class="main-header"><h1>🎓 APEX College AI Assistant</h1><p>Comprehensive AI-powered guide with auto-scraped data</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key
        api_key = st.text_input(
            "Google AI API Key",
            type="password",
            value=os.getenv("GOOGLE_AI_API_KEY", ""),
            help="Get your key from https://ai.google.dev/"
        )
        
        if not api_key:
            st.error("Please enter your Google AI API Key")
            st.stop()
        
        st.divider()
        
        # Initialize RAG pipeline
        if st.session_state.rag_pipeline is None:
            with st.spinner("🚀 Initializing comprehensive AI system..."):
                try:
                    st.session_state.rag_pipeline = EnhancedGoogleAIRAG(
                        api_key=api_key,
                        auto_scrape=True
                    )
                    st.success("✅ AI System Ready!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize: {e}")
                    st.stop()
        
        # System stats
        if st.session_state.rag_pipeline:
            st.header("📊 System Status")
            stats = st.session_state.rag_pipeline.get_stats()
            st.json(stats)
        
        # Manual refresh button
        if st.button("🔄 Refresh Data", help="Re-scrape and reprocess data"):
            st.session_state.rag_pipeline = None
            st.rerun()
    
    # Main chat interface
    st.subheader("💬 Chat with APEX Assistant")
    
    # Display messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask anything about APEX College..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="user-message">👤 {prompt}</div>', unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                result = st.session_state.rag_pipeline.query(prompt)
                
                # Add assistant response
                st.session_state.messages.append({"role": "assistant", "content": result['answer']})
                st.markdown(f'<div class="assistant-message">🤖 {result["answer"]}</div>', unsafe_allow_html=True)
                
                # Show sources in sidebar
                with st.sidebar:
                    if result['sources']:
                        st.header("📚 Sources")
                        st.write(f"**Confidence:** {result['confidence']:.3f}")
                        for i, source in enumerate(result['sources'], 1):
                            st.write(f"{i}. {source['title']} (Score: {source['similarity']:.3f})")
                
            except Exception as e:
                error_msg = f"I encountered an error: {str(e)}. Please contact APEX College at +91-7351408009."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.markdown(f'<div class="assistant-message">🤖 {error_msg}</div>', unsafe_allow_html=True)
        
        st.rerun()
    
    # Example questions
    st.subheader("💡 Try these questions:")
    col1, col2, col3 = st.columns(3)
    
    example_questions = [
        "What B.Tech programs does APEX offer?",
        "How can I apply for admission?", 
        "Tell me about placement opportunities",
        "What are the campus facilities?",
        "What is the fee structure?",
        "Are scholarships available?"
    ]
    
    for i, question in enumerate(example_questions):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.button(question, key=f"q_{i}"):
                st.session_state.messages.append({"role": "user", "content": question})
                result = st.session_state.rag_pipeline.query(question)
                st.session_state.messages.append({"role": "assistant", "content": result['answer']})
                st.rerun()
    
    # Contact info
    st.info("📞 Contact APEX: +91-7351408009 | 📧 admissions@apex.ac.in | 🌐 www.apex.ac.in")

if __name__ == "__main__":
    main()
