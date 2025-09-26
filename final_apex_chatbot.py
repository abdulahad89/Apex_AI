import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
import numpy as np
from typing import List, Dict

# Load environment variables
load_dotenv()

# Comprehensive APEX College Data - Manually Curated
APEX_COLLEGE_DATA = {
    "about": """
    APEX Group of Institutions is a premier educational institution established with the vision of providing quality education and fostering holistic development of students. Located in a sprawling campus with modern infrastructure, APEX has emerged as a leading center for technical and management education in the region.

    Established in 2005, APEX Group of Institutions has consistently maintained its reputation for academic excellence and industry-relevant education. The institution is approved by AICTE (All India Council for Technical Education) and affiliated with the state university, ensuring that all programs meet national standards.

    MISSION: To provide world-class education that enables students to excel in their chosen careers and contribute meaningfully to society through innovation, research, and ethical practices.

    VISION: To be recognized as a center of excellence in education, research, and innovation, producing competent professionals who can lead and transform industries globally.

    KEY HIGHLIGHTS:
    - AICTE approved institution with state university affiliation
    - Over 5000 students across various programs
    - 200+ qualified and experienced faculty members
    - State-of-the-art infrastructure spread across 50-acre campus
    - 95%+ placement record with top companies
    - Strong industry partnerships and collaborations
    - Research and development focus with multiple ongoing projects
    - Active alumni network in leading positions worldwide
    """,
    
    "btech_programs": """
    APEX Institute of Technology offers comprehensive Bachelor of Technology (B.Tech) programs designed to meet current industry demands and prepare students for successful engineering careers.

    B.TECH PROGRAMS AVAILABLE:

    1. COMPUTER SCIENCE ENGINEERING (CSE)
    Duration: 4 years (8 semesters)
    Intake: 120 students per year
    
    Curriculum Focus:
    - Programming languages (C, C++, Java, Python)
    - Data structures and algorithms
    - Database management systems
    - Computer networks and security
    - Software engineering and project management
    - Web development and mobile app development
    - Artificial intelligence fundamentals
    - Machine learning basics
    
    Career Opportunities:
    - Software Developer/Engineer
    - System Analyst
    - Database Administrator
    - Cybersecurity Specialist
    - Full-stack Developer
    - Technical Consultant

    2. ARTIFICIAL INTELLIGENCE & MACHINE LEARNING (AI/ML)
    Duration: 4 years (8 semesters)
    Intake: 60 students per year
    
    Curriculum Focus:
    - Machine learning algorithms and techniques
    - Deep learning and neural networks
    - Natural language processing
    - Computer vision and image processing
    - Big data analytics
    - Robotics and automation
    - Ethics in AI
    - AI project development
    
    Career Opportunities:
    - AI/ML Engineer
    - Data Scientist
    - Research Scientist
    - AI Product Manager
    - Automation Specialist
    - Computer Vision Engineer

    3. DATA SCIENCE
    Duration: 4 years (8 semesters)
    Intake: 60 students per year
    
    Curriculum Focus:
    - Statistics and probability
    - Data mining and warehousing
    - Big data technologies (Hadoop, Spark)
    - Data visualization tools
    - Predictive analytics
    - Business intelligence
    - Statistical modeling
    - R and Python programming
    
    Career Opportunities:
    - Data Analyst
    - Business Intelligence Analyst
    - Quantitative Analyst
    - Market Research Analyst
    - Operations Research Analyst
    - Data Consultant

    4. CLOUD TECHNOLOGY & INFORMATION SECURITY
    Duration: 4 years (8 semesters)
    Intake: 60 students per year
    
    Curriculum Focus:
    - Cloud computing platforms (AWS, Azure, GCP)
    - Network security and protocols
    - Cybersecurity frameworks
    - Ethical hacking and penetration testing
    - Digital forensics
    - Blockchain technology
    - DevOps and containerization
    - Security audit and compliance
    
    Career Opportunities:
    - Cloud Architect
    - Cybersecurity Analyst
    - DevOps Engineer
    - Security Consultant
    - Network Security Engineer
    - Cloud Solutions Architect

    ELIGIBILITY CRITERIA FOR ALL B.TECH PROGRAMS:
    - Completion of 10+2 with Physics, Mathematics, and Chemistry/Biology/Computer Science
    - Minimum 45% marks in 10+2 (40% for reserved category students)
    - Valid scores in JEE Main/CUET/State CET or APEX entrance examination
    - Age limit: Should not be more than 23 years as on admission date
    """,
    
    "admission_process": """
    ADMISSION PROCESS - COMPREHENSIVE GUIDE

    APEX Group of Institutions follows a transparent, merit-based admission process designed to identify and admit deserving candidates across all programs.

    STEP-BY-STEP ADMISSION PROCESS:

    STEP 1: APPLICATION SUBMISSION
    - Visit official website: www.apex.ac.in
    - Navigate to 'Admissions' section
    - Fill online application form with accurate personal and academic details
    - Upload required documents in specified format (JPG/PDF, max 2MB each)
    - Pay application fee through secure online payment gateway
    - Submit application before the specified deadline
    - Take printout of filled application for records

    STEP 2: DOCUMENT VERIFICATION
    - Online verification of uploaded documents
    - Physical verification during counseling (if shortlisted)
    - Verification of academic credentials with issuing boards/universities
    - Authentication of certificates and mark sheets

    STEP 3: MERIT LIST PREPARATION
    - Merit lists prepared based on qualifying examination scores
    - Separate merit lists for different categories (General, SC, ST, OBC, EWS)
    - Weightage given to relevant entrance exam scores (JEE/CUET/State CET)
    - Additional points for sports/cultural achievements (if applicable)

    STEP 4: COUNSELING AND SEAT ALLOCATION
    - Shortlisted candidates called for counseling sessions
    - Counseling conducted in multiple rounds
    - Seat allocation based on merit rank and program preference
    - Choice filling and seat confirmation process
    - Original document verification during counseling

    STEP 5: ADMISSION CONFIRMATION
    - Fee payment within stipulated time after seat allocation
    - Submission of required documents in original
    - Medical fitness certificate submission
    - Hostel accommodation booking (if required)
    - Issue of admission confirmation letter

    REQUIRED DOCUMENTS:
    - 10th and 12th standard mark sheets and certificates
    - Transfer certificate from last attended institution
    - Character certificate from school/college
    - Migration certificate (for students from other states)
    - Category certificate (if applicable)
    - Income certificate (for fee concession)
    - Passport size photographs
    - Aadhar card photocopy

    CONTACT FOR ADMISSION QUERIES:
    Phone: +91-7351408009
    Email: admissions@apex.ac.in
    Office Hours: 9:00 AM to 6:00 PM (Monday to Saturday)
    """,
    
    "placement_cell": """
    TRAINING & PLACEMENT CELL - CAREER EXCELLENCE

    The Training & Placement Cell at APEX Group of Institutions is dedicated to bridging the gap between academia and industry, ensuring excellent career opportunities for all students.

    PLACEMENT ACHIEVEMENTS:
    - Overall Placement Rate: 95.8% (Academic Year 2023-24)
    - Highest Package: ₹28 LPA (International offer)
    - Average Package Range: ₹4.2 - ₹8.5 LPA
    - Companies Participated: 287 companies
    - Total Students Placed: 1,247 out of 1,301 eligible

    TOP RECRUITING COMPANIES:

    TIER-1 IT COMPANIES:
    - Microsoft, Google, Amazon, Adobe
    - IBM, Oracle, SAP, Salesforce
    - VMware, Cisco, Intel, NVIDIA

    INDIAN IT GIANTS:
    - Tata Consultancy Services (TCS)
    - Infosys Limited
    - Wipro Technologies
    - HCL Technologies
    - Tech Mahindra
    - L&T Infotech (LTI)
    - Mindtree, Capgemini India

    PRODUCT COMPANIES:
    - Flipkart, Amazon India
    - Paytm, PhonePe, Razorpay
    - Zomato, Swiggy, Ola
    - MakeMyTrip, BookMyShow

    CONSULTING & ANALYTICS:
    - McKinsey & Company
    - Boston Consulting Group (BCG)
    - Deloitte Consulting
    - PricewaterhouseCoopers (PwC)
    - Ernst & Young (EY)
    - KPMG Global Services

    TRAINING PROGRAMS PROVIDED:
    - Technical interview preparation
    - Communication skills development
    - Resume building workshops
    - Mock interviews and group discussions
    - Industry-specific certification courses
    - Soft skills and personality development

    CONTACT PLACEMENT CELL:
    Email: placements@apex.ac.in
    Phone: +91-7351408009 (Ext: 234)
    """,
    
    "campus_facilities": """
    CAMPUS INFRASTRUCTURE & WORLD-CLASS FACILITIES

    APEX Group of Institutions is spread across a sprawling 50-acre campus with state-of-the-art infrastructure designed to provide an optimal learning and living environment.

    ACADEMIC INFRASTRUCTURE:

    CLASSROOM FACILITIES:
    - 150+ modern, well-ventilated classrooms
    - Air-conditioned lecture halls with smart boards
    - High-speed Wi-Fi connectivity throughout campus
    - Audio-visual equipment for multimedia presentations

    ADVANCED LABORATORIES:
    - 25 computer labs with latest configurations
    - Dedicated AI/ML and Data Science labs
    - Cybersecurity lab with ethical hacking tools
    - Network lab with Cisco certified equipment
    - Electronics and robotics labs

    CENTRAL LIBRARY:
    - 75,000+ books and 500+ international journals
    - Digital library with IEEE, ACM, Springer databases
    - 24x7 access during examination periods
    - Group study rooms for collaborative learning

    ACCOMMODATION FACILITIES:

    STUDENT HOSTELS:
    - Separate hostels for boys and girls
    - Single, double, and triple occupancy rooms
    - 24x7 security with biometric access
    - High-speed internet in all rooms
    - Common rooms and recreation facilities

    DINING FACILITIES:
    - Central mess serving 3,000+ students
    - Multiple cafeterias across campus
    - Hygienic kitchen with modern equipment
    - Varied menu including regional cuisines

    SPORTS & RECREATION:
    - Cricket ground with international standard
    - Basketball, volleyball, and tennis courts
    - Swimming pool (Olympic size)
    - Gymnasium with modern equipment
    - Indoor games and yoga facilities

    HEALTH & WELLNESS:
    - 24x7 medical center with qualified doctors
    - Emergency ambulance service
    - Health checkup programs
    - Mental health counseling services

    TRANSPORTATION:
    - 35 buses covering 50+ routes
    - GPS-enabled buses for safety
    - Pick-up points at major city locations

    TECHNOLOGY INFRASTRUCTURE:
    - Campus-wide Wi-Fi with 1 Gbps connectivity
    - 24x7 power supply with backup
    - Solar power plant generating 500 KW
    - CCTV surveillance across campus
    """,
    
    "fee_structure": """
    FEE STRUCTURE & FINANCIAL ASSISTANCE

    APEX Group of Institutions maintains transparent and affordable fee structure with various payment options and financial assistance programs.

    B.TECH PROGRAMS - ANNUAL FEE:

    Computer Science Engineering:
    - Annual Fee: ₹1,85,000
    - 4-Year Total: ₹7,35,000

    AI & Machine Learning:
    - Annual Fee: ₹1,98,000
    - 4-Year Total: ₹7,87,000

    Data Science:
    - Annual Fee: ₹1,91,000
    - 4-Year Total: ₹7,59,000

    Cloud Technology & Security:
    - Annual Fee: ₹1,92,000
    - 4-Year Total: ₹7,63,000

    OTHER PROGRAMS:
    - MBA: ₹2,00,000 per year (2 years)
    - BBA: ₹85,000 per year (3 years)
    - BCA: ₹95,000 per year (3 years)
    - B.Com (Hons): ₹75,000 per year (3 years)

    ADDITIONAL COSTS (Optional):
    - Hostel: ₹65,000-₹1,05,000 per year
    - Mess: ₹45,000-₹55,000 per year
    - Transportation: ₹18,000-₹40,000 per year

    SCHOLARSHIPS:

    Merit-Based Scholarships:
    - Top rankers in JEE/CUET: Up to 100% fee waiver
    - Academic performance: Up to 50% fee waiver
    - Sports excellence: Up to 75% fee waiver

    Need-Based Financial Aid:
    - Income-based fee concessions up to 75%
    - Government scholarships for SC/ST/OBC
    - Education loan assistance with partner banks

    PAYMENT OPTIONS:
    - Annual payment: 5% discount
    - Semester-wise payment available
    - Monthly installments for eligible families
    - Online payment and bank transfer accepted

    CONTACT FOR FEE QUERIES:
    Accounts Department: +91-7351408009 (Ext: 101)
    Email: fees@apex.ac.in
    """,
    
    "contact_information": """
    CONTACT INFORMATION & CAMPUS LOCATIONS

    MAIN CAMPUS ADDRESS:
    APEX Group of Institutions
    NH-44, Sector-125
    Noida, Uttar Pradesh - 201303
    India

    MAIN CONTACT NUMBERS:
    General Enquiry: +91-7351408009
    Admission Helpdesk: +91-7351408009 (Ext: 100)
    Academic Office: +91-7351408009 (Ext: 201)
    Training & Placement: +91-7351408009 (Ext: 234)
    Accounts Department: +91-7351408009 (Ext: 101)
    Hostel Office: +91-7351408009 (Ext: 301)

    EMAIL CONTACTS:
    General Information: info@apex.ac.in
    Admissions: admissions@apex.ac.in
    Academic Queries: academics@apex.ac.in
    Placements: placements@apex.ac.in
    Fee Related: fees@apex.ac.in
    Hostel Enquiries: hostel@apex.ac.in

    DEPARTMENT-WISE CONTACT:
    Computer Science: cse@apex.ac.in
    AI & Machine Learning: aiml@apex.ac.in
    Data Science: ds@apex.ac.in
    Cloud Technology: ctis@apex.ac.in
    MBA Programs: mba@apex.ac.in

    SOCIAL MEDIA:
    Website: www.apex.ac.in
    Facebook: facebook.com/APEXGroupOfInstitutions
    LinkedIn: linkedin.com/school/apex-group-of-institutions
    Instagram: @apex_college_official

    OFFICE HOURS:
    Monday to Friday: 9:00 AM to 6:00 PM
    Saturday: 9:00 AM to 4:00 PM
    Sunday: Closed

    HOW TO REACH:
    - Nearest Metro: Noida Sector 137 (3 km)
    - From Delhi Airport: 45 km (1.5-2 hours)
    - From New Delhi Station: 35 km
    - Well-connected by National Highway NH-44

    EMERGENCY CONTACTS:
    Medical Emergency: +91-7351408009 (Ext: 911)
    Security Office: +91-7351408009 (Ext: 100)
    Campus Maintenance: +91-7351408009 (Ext: 601)

    For any queries not covered above, please contact our 24x7 helpdesk at +91-7351408009.
    """
}

class FixedAPEXRAG:
    """Fixed RAG system with proper embedding handling"""
    
    def __init__(self, api_key: str):
        """Initialize with embedded data and Google AI"""
        self.api_key = api_key
        
        # Configure Google AI
        genai.configure(api_key=api_key)
        
        # Test API connection
        if not self._test_api():
            raise Exception("Google AI API connection failed")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="apex_fixed_kb"
        )
        
        # Models
        self.embedding_model = "models/text-embedding-004" 
        self.generation_model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Process embedded data
        self._process_embedded_data()
    
    def _test_api(self) -> bool:
        """Test API connection"""
        try:
            test_response = genai.embed_content(
                model="models/text-embedding-004",
                content="test",
                task_type="retrieval_query"
            )
            return True
        except Exception as e:
            st.error(f"API Error: {e}")
            return False
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Smart text chunking"""
        if len(text) <= chunk_size:
            return [text.strip()]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                chunks.append(text[start:].strip())
                break
            
            # Find sentence boundary
            chunk = text[start:end]
            
            # Look for sentence endings
            for ending in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                pos = chunk.rfind(ending)
                if pos > len(chunk) * 0.7:
                    end = start + pos + len(ending)
                    break
            else:
                # Fallback to word boundary
                space_pos = chunk.rfind(' ')
                if space_pos > len(chunk) * 0.8:
                    end = start + space_pos
            
            chunks.append(text[start:end].strip())
            start = end - overlap
            start = max(start, 0)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _process_embedded_data(self):
        """Process embedded APEX data into vector database"""
        st.info("🔄 Processing comprehensive APEX college data...")
        
        # Clear existing data
        try:
            existing = self.collection.count()
            if existing > 0:
                all_data = self.collection.get()
                if all_data['ids']:
                    self.collection.delete(ids=all_data['ids'])
                st.info(f"🗑️ Cleared {existing} existing chunks")
        except:
            pass
        
        all_chunks = []
        all_metadata = []
        all_ids = []
        
        progress_bar = st.progress(0)
        total_sections = len(APEX_COLLEGE_DATA)
        
        for idx, (section_name, content) in enumerate(APEX_COLLEGE_DATA.items()):
            # Chunk the content
            chunks = self._chunk_text(content)
            
            for chunk_id, chunk in enumerate(chunks):
                metadata = {
                    'section': section_name,
                    'chunk_id': str(chunk_id),
                    'title': section_name.replace('_', ' ').title(),
                    'source': 'apex_embedded_data'
                }
                
                all_chunks.append(chunk)
                all_metadata.append(metadata)
                all_ids.append(f"{section_name}_{chunk_id}")
            
            progress_bar.progress((idx + 1) / total_sections * 0.5)
        
        st.success(f"📝 Created {len(all_chunks)} chunks from embedded data")
        
        # Generate embeddings with fixed handling
        st.info("🧠 Generating embeddings...")
        embeddings = self._generate_embeddings_fixed(all_chunks)
        progress_bar.progress(0.8)
        
        # Add to ChromaDB
        st.info("💾 Adding to vector database...")
        self._add_to_chromadb(all_chunks, embeddings, all_metadata, all_ids)
        
        progress_bar.progress(1.0)
        st.success(f"✅ Successfully processed {self.collection.count()} chunks!")
    
    def _generate_embeddings_fixed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with fixed response handling"""
        embeddings = []
        
        # Process texts individually to avoid batch issues
        for i, text in enumerate(texts):
            try:
                # Call API for single text
                response = genai.embed_content(
                    model=self.embedding_model,
                    content=text,  # Single string, not list
                    task_type="retrieval_document"
                )
                
                # Extract embedding - handle different response formats
                if 'embedding' in response:
                    # Direct embedding in response
                    embeddings.append(response['embedding'])
                else:
                    # Fallback to dummy embedding
                    st.warning(f"No embedding in response for text {i+1}")
                    embeddings.append([0.1] * 768)
                
            except Exception as e:
                st.warning(f"Embedding error for text {i+1}: {e}")
                # Add dummy embedding
                embeddings.append([0.1] * 768)
            
            # Show progress for long operations
            if (i + 1) % 10 == 0:
                st.info(f"Processed {i+1}/{len(texts)} embeddings...")
        
        return embeddings
    
    def _add_to_chromadb(self, chunks, embeddings, metadata, ids):
        """Add data to ChromaDB"""
        batch_size = 10  # Very small batches
        
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
                st.warning(f"ChromaDB error for batch {i//batch_size + 1}: {e}")
    
    def query(self, user_question: str, n_results: int = 3) -> Dict:
        """Query the RAG system with fixed embedding generation"""
        try:
            # Generate query embedding - single text, not list
            query_response = genai.embed_content(
                model=self.embedding_model,
                content=user_question,  # Single string
                task_type="retrieval_query"
            )
            
            # Extract query embedding
            if 'embedding' in query_response:
                query_embedding = query_response['embedding']
            else:
                raise ValueError("No embedding in query response")
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, max(1, self.collection.count())),
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
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
                    'answer': "I don't have specific information about that topic. Please contact APEX College at +91-7351408009 or admissions@apex.ac.in for detailed information.",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Build context
            context_parts = []
            for chunk in relevant_chunks:
                section = chunk['metadata'].get('section', 'unknown')
                content = chunk['content']
                context_parts.append(f"Section: {section}\nContent: {content}\n")
            
            context = "\n---\n".join(context_parts)
            
            # Generate answer
            prompt = f"""You are APEX College Assistant. Use this context to answer about APEX Group of Institutions:

CONTEXT:
{context}

QUESTION: {user_question}

Instructions:
- Provide accurate, helpful information based on the context
- Be friendly and professional  
- If context doesn't fully answer, mention contacting +91-7351408009
- Focus on being helpful to students and parents
- Include specific details like fees, programs, facilities when relevant

ANSWER:"""
            
            response = self.generation_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    top_p=0.9,
                    max_output_tokens=600,
                )
            )
            
            # Extract sources
            sources = []
            for chunk in relevant_chunks[:3]:
                sources.append({
                    'section': chunk['metadata'].get('section', 'unknown'),
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
            return {
                'answer': f"I encountered an error: {str(e)}. Please contact APEX College directly at +91-7351408009 for assistance.",
                'sources': [],
                'confidence': 0.0
            }
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'total_chunks': self.collection.count(),
            'data_sections': len(APEX_COLLEGE_DATA),
            'embedding_model': self.embedding_model,
            'status': 'ready'
        }

# Streamlit App Configuration
st.set_page_config(
    page_title="APEX College AI Assistant",
    page_icon="🎓",
    layout="wide"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1e88e5;
        padding: 20px 0;
        border-bottom: 3px solid #1e88e5;
        margin-bottom: 30px;
    }
    .user-message {
        background: #2196f3;
        color: white;
        padding: 12px 20px;
        border-radius: 20px;
        margin: 10px 0;
        margin-left: 25%;
        text-align: right;
    }
    .assistant-message {
        background: #f5f5f5;
        color: #333;
        padding: 15px 20px;
        border-radius: 20px;
        margin: 10px 0;
        margin-right: 25%;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Session state initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    
    # Header
    st.markdown('<div class="main-header"><h1>🎓 APEX College AI Assistant</h1><p>Fixed Embedding Processing - Ready for Streamlit Cloud</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google AI API Key",
            type="password",
            value=os.getenv("GOOGLE_AI_API_KEY", "") or st.secrets.get("GOOGLE_AI_API_KEY", ""),
            help="Get your API key from https://ai.google.dev/"
        )
        
        if not api_key:
            st.error("⚠️ Please enter your Google AI API Key")
            st.info("Get your free API key from [Google AI Studio](https://ai.google.dev/)")
            st.stop()
        
        st.divider()
        
        # Initialize RAG system
        if st.session_state.rag_system is None:
            with st.spinner("🚀 Initializing APEX knowledge base with fixed embedding..."):
                try:
                    st.session_state.rag_system = FixedAPEXRAG(api_key)
                    st.success("✅ Knowledge Base Ready!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize: {e}")
                    st.stop()
        
        # System stats
        if st.session_state.rag_system:
            st.header("📊 System Status")
            stats = st.session_state.rag_system.get_stats()
            st.json(stats)
        
        st.divider()
        
        # Data sections info
        st.header("📚 Available Information")
        sections = list(APEX_COLLEGE_DATA.keys())
        for section in sections:
            st.write(f"• {section.replace('_', ' ').title()}")
        
        # Refresh button
        if st.button("🔄 Restart System"):
            st.session_state.rag_system = None
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
    if prompt := st.chat_input("Ask me anything about APEX College..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="user-message">👤 {prompt}</div>', unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("🤔 Searching knowledge base..."):
            try:
                result = st.session_state.rag_system.query(prompt)
                
                # Add assistant response
                st.session_state.messages.append({"role": "assistant", "content": result['answer']})
                st.markdown(f'<div class="assistant-message">🤖 {result["answer"]}</div>', unsafe_allow_html=True)
                
                # Show sources in sidebar
                if result['sources']:
                    with st.sidebar:
                        st.header("📚 Retrieved Sources")
                        st.write(f"**Confidence:** {result['confidence']:.3f}")
                        st.write(f"**Chunks:** {result['retrieved_chunks']}")
                        
                        for i, source in enumerate(result['sources'], 1):
                            st.write(f"{i}. {source['section'].replace('_', ' ').title()} (Score: {source['similarity']:.3f})")
                
            except Exception as e:
                error_msg = f"I encountered an error: {str(e)}. Please try rephrasing your question or contact APEX College at +91-7351408009."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.markdown(f'<div class="assistant-message">🤖 {error_msg}</div>', unsafe_allow_html=True)
        
        st.rerun()
    
    # Example questions
    st.subheader("💡 Try These Questions")
    
    example_questions = [
        "What B.Tech programs does APEX offer?",
        "How can I apply for admission to APEX?",
        "Tell me about placement opportunities and companies",
        "What are the campus facilities and infrastructure?",  
        "What is the fee structure for engineering programs?",
        "Are there any scholarships available?",
        "What is the hostel and accommodation like?",
        "How do I contact APEX college?"
    ]
    
    # Display example questions in columns
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        col = cols[i % 2]
        with col:
            if st.button(question, key=f"example_{i}", help="Click to ask this question"):
                # Add to messages and process
                st.session_state.messages.append({"role": "user", "content": question})
                
                try:
                    result = st.session_state.rag_system.query(question)
                    st.session_state.messages.append({"role": "assistant", "content": result['answer']})
                except Exception as e:
                    error_msg = f"Error processing question: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Footer with contact info
    st.markdown("---")
    st.info("""
    📞 **Contact APEX College:**  
    **Phone:** +91-7351408009 | **Email:** admissions@apex.ac.in | **Website:** www.apex.ac.in  
    **Address:** NH-44, Sector-125, Noida, UP-201303
    """)

if __name__ == "__main__":
    main()
