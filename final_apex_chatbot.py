import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
import numpy as np
from typing import List, Dict
import json
import uuid

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

    COURSE FEATURES:
    - Industry-aligned curriculum updated regularly
    - Hands-on laboratory training with latest equipment
    - Project-based learning approach
    - Industrial training and internships in 6th/7th semester
    - Guest lectures by industry experts
    - Coding competitions and technical events
    - Research project opportunities
    - International certification programs
    """,
    
    "other_programs": """
    BEYOND B.TECH: DIVERSE ACADEMIC PROGRAMS

    1. MASTER OF BUSINESS ADMINISTRATION (MBA)
    Duration: 2 years (4 semesters)
    Intake: 120 students per year
    Specializations:
    - Marketing Management
    - Human Resource Management
    - Finance Management
    - Operations Management
    - Information Technology Management
    - International Business

    Eligibility:
    - Bachelor's degree in any discipline with minimum 50% marks
    - Valid CAT/MAT/XAT/CMAT/KMAT/APEX management entrance test scores
    - Minimum 2 years work experience preferred but not mandatory

    2. BACHELOR OF BUSINESS ADMINISTRATION (BBA)
    Duration: 3 years (6 semesters)
    Intake: 60 students per year
    Focus Areas:
    - Business fundamentals and management principles
    - Marketing and sales management
    - Financial accounting and management
    - Organizational behavior and HR
    - Business communication and entrepreneurship

    3. BACHELOR OF COMPUTER APPLICATIONS (BCA)
    Duration: 3 years (6 semesters)
    Intake: 60 students per year
    Curriculum:
    - Programming languages and software development
    - Database management and web technologies
    - Computer networks and system administration
    - Mobile application development
    - Project management and software testing

    4. BACHELOR OF COMMERCE (B.COM HONORS)
    Duration: 3 years (6 semesters)
    Intake: 60 students per year
    Specializations:
    - Accounting and Finance
    - Banking and Insurance
    - International Business
    - E-commerce and Digital Marketing

    5. BACHELOR OF EDUCATION (B.ED)
    Duration: 2 years (4 semesters)
    Intake: 50 students per year
    Teaching Methods:
    - Mathematics and Science teaching methodology
    - English and Social Science pedagogy
    - Educational psychology and assessment
    - Classroom management and technology integration

    6. POLYTECHNIC COURSES
    Duration: 3 years (6 semesters)
    Programs Available:
    - Diploma in Computer Science Engineering
    - Diploma in Electronics & Communication Engineering
    - Diploma in Mechanical Engineering
    - Diploma in Civil Engineering

    7. PHARMACY PROGRAMS
    Duration: D.Pharm (2 years), B.Pharm (4 years)
    Intake: D.Pharm (60), B.Pharm (60)
    Focus: Pharmaceutical sciences, drug development, clinical pharmacy, hospital pharmacy management

    8. BACHELOR OF SCIENCE PROGRAMS
    Available Programs:
    - B.Sc Computer Science (3 years)
    - B.Sc Information Technology (3 years)
    - B.Sc Mathematics (3 years)
    - B.Sc Physics (3 years)
    - B.Sc Chemistry (3 years)

    ADMISSION REQUIREMENTS VARY BY PROGRAM:
    - UG Programs: 10+2 with relevant subjects and minimum percentage
    - PG Programs: Bachelor's degree with required percentage and entrance test scores
    - Professional Programs: Specific eligibility as per regulatory body requirements
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

    REQUIRED DOCUMENTS CHECKLIST:

    FOR ALL APPLICANTS:
    - 10th standard mark sheet and certificate
    - 12th standard mark sheet and certificate
    - Transfer certificate from last attended institution
    - Character certificate from school/college
    - Migration certificate (for students from other states)
    - Passport size photographs (6 copies)
    - Aadhar card photocopy
    - Income certificate from competent authority

    FOR CATEGORY STUDENTS:
    - Caste certificate (SC/ST/OBC) issued by competent authority
    - EWS certificate (for economically weaker section)
    - Non-creamy layer certificate (for OBC candidates)

    FOR FEE CONCESSION:
    - Income certificate with annual family income details
    - Domicile certificate (if applicable)
    - Bank account details for scholarship disbursement

    ENTRANCE EXAMINATIONS ACCEPTED:
    - JEE Main (for B.Tech programs)
    - CUET (Common University Entrance Test)
    - State CET (Conducted by state government)
    - APEX Merit Test (Institution's own entrance examination)
    - CAT/MAT/XAT/CMAT (for MBA program)

    APPLICATION DEADLINES AND IMPORTANT DATES:
    - Online applications typically open: May 1st
    - Last date for B.Tech applications: June 30th
    - Last date for other UG programs: July 15th
    - Last date for PG programs: July 31st
    - Merit list publication: Within 7 days of application deadline
    - Counseling rounds: July-August
    - Final admission confirmation: By August 31st
    - Classes commencement: First week of September

    APPLICATION FEE STRUCTURE:
    - General/OBC candidates: ₹1,500
    - SC/ST candidates: ₹750
    - EWS candidates: ₹750
    - Late application fee: Additional ₹500

    CONTACT FOR ADMISSION QUERIES:
    Admission Helpdesk: +91-7351408009
    Email: admissions@apex.ac.in
    WhatsApp Support: +91-7351408009
    Office Hours: 9:00 AM to 6:00 PM (Monday to Saturday)
    Address: APEX Group of Institutions, NH-44, Sector-125, Noida, UP-201303
    """,
    
    "placement_cell": """
    TRAINING & PLACEMENT CELL - CAREER EXCELLENCE

    The Training & Placement Cell at APEX Group of Institutions is a dedicated department committed to bridging the gap between academia and industry, ensuring excellent career opportunities for all students.

    PLACEMENT ACHIEVEMENTS AND STATISTICS:

    OVERALL PLACEMENT RECORD:
    - Overall Placement Rate: 95.8% (Academic Year 2023-24)
    - Number of Students Placed: 1,247 out of 1,301 eligible students
    - Highest Package Offered: ₹28 LPA (International offer)
    - Highest Domestic Package: ₹18 LPA
    - Average Package Range: ₹4.2 - ₹8.5 LPA
    - Median Package: ₹6.2 LPA
    - Companies Participated: 287 companies
    - Total Job Offers: 1,456 (multiple offers per student)

    PROGRAM-WISE PLACEMENT STATISTICS:

    B.Tech Computer Science Engineering:
    - Placement Rate: 98.5%
    - Average Package: ₹7.8 LPA
    - Highest Package: ₹28 LPA
    - Top Recruiters: Microsoft, Amazon, Google, IBM, TCS Innovation Labs

    B.Tech AI & Machine Learning:
    - Placement Rate: 97.2%
    - Average Package: ₹8.2 LPA
    - Highest Package: ₹22 LPA
    - Top Recruiters: NVIDIA, Intel, Qualcomm, Flipkart, Paytm

    B.Tech Data Science:
    - Placement Rate: 96.8%
    - Average Package: ₹7.5 LPA
    - Highest Package: ₹18 LPA
    - Top Recruiters: Analytics Quotient, Fractal Analytics, Mu Sigma

    MBA Programs:
    - Placement Rate: 94.5%
    - Average Package: ₹9.2 LPA
    - Highest Package: ₹15 LPA
    - Top Recruiters: Deloitte, KPMG, Accenture, Wipro, HCL

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
    - Mindtree (now part of L&T)
    - Capgemini India

    PRODUCT COMPANIES:
    - Flipkart, Amazon India
    - Paytm, PhonePe, Razorpay
    - Zomato, Swiggy, Ola
    - MakeMyTrip, BookMyShow
    - Freshworks, Zoho Corporation

    CORE ENGINEERING COMPANIES:
    - Larsen & Toubro (L&T)
    - Bajaj Auto Limited
    - Mahindra Group
    - Tata Motors
    - Bosch India
    - Siemens Limited
    - ABB India

    CONSULTING & ANALYTICS:
    - McKinsey & Company
    - Boston Consulting Group (BCG)
    - Deloitte Consulting
    - PricewaterhouseCoopers (PwC)
    - Ernst & Young (EY)
    - KPMG Global Services

    BANKING & FINANCIAL SERVICES:
    - ICICI Bank, HDFC Bank
    - Axis Bank, Kotak Mahindra Bank
    - Yes Bank, IndusInd Bank
    - Goldman Sachs, Morgan Stanley
    - JP Morgan Chase, Deutsche Bank

    TRAINING PROGRAMS PROVIDED:

    PRE-PLACEMENT TRAINING:
    - Technical interview preparation and mock interviews
    - Aptitude and reasoning test preparation
    - Communication skills and personality development
    - Group discussion training and leadership skills
    - Resume building and LinkedIn profile optimization
    - Industry-specific skill development workshops

    SOFT SKILLS DEVELOPMENT:
    - Professional communication and presentation skills
    - Team building and collaborative working
    - Time management and productivity enhancement
    - Stress management and emotional intelligence
    - Professional etiquette and workplace behavior

    TECHNICAL SKILL ENHANCEMENT:
    - Programming and coding bootcamps
    - Data structures and algorithms intensive training
    - System design and architecture workshops
    - Domain-specific certification courses
    - Latest technology trends and tools training

    INDUSTRY INTERFACE PROGRAMS:
    - Regular guest lectures by industry experts
    - Industrial visits to leading companies
    - Live project assignments with industry partners
    - Internship facilitation with top companies
    - Mentorship programs with alumni and industry professionals

    ENTREPRENEURSHIP SUPPORT:
    - Startup incubation and funding guidance
    - Business plan development support
    - Networking with investors and venture capitalists
    - Legal and regulatory compliance assistance
    - Market research and customer validation support

    PLACEMENT PROCESS:

    COMPANY REGISTRATION:
    - Companies register through placement portal
    - Job description and requirements submission
    - Salary and benefits package disclosure
    - Interview process and timeline planning

    STUDENT PREPARATION:
    - Eligibility criteria communication
    - Resume screening and shortlisting
    - Pre-placement presentation (PPT) by companies
    - Student registration for interested companies

    SELECTION PROCESS:
    - Online assessments and coding tests
    - Technical interviews (multiple rounds)
    - HR and behavioral interviews
    - Final selection and offer letter distribution

    PLACEMENT SUPPORT SERVICES:
    - 24/7 placement portal access for students
    - Regular placement updates and notifications
    - Individual career counseling sessions
    - Placement preparation workshops and seminars
    - Alumni network support and guidance
    - Post-placement support and feedback collection

    CONTACT PLACEMENT CELL:
    Director, Training & Placement: Dr. Rajesh Kumar
    Phone: +91-7351408009 (Ext: 234)
    Email: placements@apex.ac.in
    WhatsApp: +91-9876543210
    Office: Training & Placement Block, 2nd Floor, APEX Campus
    """,
    
    "campus_facilities": """
    CAMPUS INFRASTRUCTURE & WORLD-CLASS FACILITIES

    APEX Group of Institutions is spread across a sprawling 50-acre campus with state-of-the-art infrastructure designed to provide an optimal learning and living environment for students.

    ACADEMIC INFRASTRUCTURE:

    CLASSROOM FACILITIES:
    - 150+ modern, well-ventilated classrooms
    - Air-conditioned lecture halls with 50-150 seating capacity
    - Smart boards and LED projectors in every classroom
    - High-speed Wi-Fi connectivity throughout campus
    - Ergonomic furniture for comfortable learning
    - Audio-visual equipment for multimedia presentations
    - Accessibility features for differently-abled students

    ADVANCED LABORATORIES:

    Computer Science & IT Labs:
    - 25 computer labs with 40-50 systems each
    - Latest configuration computers with high-end processors
    - Licensed software: Microsoft Visual Studio, Oracle, SAP, MATLAB
    - Dedicated labs for AI/ML, Data Science, and Cloud Computing
    - Cybersecurity lab with ethical hacking tools
    - Mobile app development lab with Android/iOS environments
    - Network lab with Cisco certified equipment

    Engineering Labs:
    - Electronics and Communication lab with modern equipment
    - Microprocessor and microcontroller lab
    - Digital signal processing lab
    - VLSI design lab with Xilinx and Altera tools
    - Embedded systems development lab
    - Robotics and automation lab
    - IoT (Internet of Things) development center

    Science Laboratories:
    - Physics lab with modern experimental setups
    - Chemistry lab with fume hoods and safety equipment
    - Mathematics lab with statistical software
    - Research lab for final year projects
    - Innovation lab for prototype development

    CENTRAL LIBRARY - KNOWLEDGE HUB:
    - Built-up area: 15,000 sq. ft. across three floors
    - Collection: 75,000+ books, 500+ international journals
    - Digital library with access to IEEE, ACM, Springer databases
    - E-book collection: 25,000+ titles
    - Separate reading halls for UG and PG students
    - Group study rooms for collaborative learning
    - 24x7 access during examination periods
    - Book bank facility for economically weaker students
    - Photocopying and printing facilities
    - Library automation system with barcode scanning

    ACCOMMODATION & RESIDENTIAL FACILITIES:

    STUDENT HOSTELS:
    Boys Hostels:
    - 4 separate hostel blocks accommodating 2,000 students
    - Single, double, and triple occupancy rooms available
    - Attached bathrooms with 24x7 water supply
    - Individual study tables and storage facilities
    - Common rooms with TV, indoor games, and newspapers
    - High-speed internet in all rooms
    - Laundry service and ironing facilities

    Girls Hostels:
    - 3 separate hostel blocks accommodating 1,200 students
    - Enhanced security with biometric access control
    - Lady wardens available 24x7
    - Beauty salon and medical room facilities
    - Separate visitor areas with CCTV monitoring
    - Recreation room with entertainment facilities

    DINING & FOOD SERVICES:
    Central Mess Facility:
    - Capacity to serve 3,000+ students simultaneously
    - Hygienic kitchen with modern cooking equipment
    - Nutritious meals prepared by experienced chefs
    - Menu includes North Indian, South Indian, and Continental cuisine
    - Special diet arrangements for medical conditions
    - Quality control through regular health inspections

    Multiple Cafeterias:
    - 5 cafeterias across campus serving snacks and beverages
    - Food court with 15+ food stalls offering variety
    - 24x7 cafeteria near hostels for emergency needs
    - Affordable pricing with student-friendly rates
    - Cashless payment options available

    SPORTS & RECREATION COMPLEX:

    Outdoor Sports Facilities:
    - Cricket ground with international standard pitch
    - Football field with natural grass and floodlights
    - Basketball courts (4) with international dimensions
    - Volleyball courts (6) with proper net systems
    - Tennis courts (2) with synthetic surface
    - Badminton courts (8) in covered area
    - Athletic track (400m) around the main ground
    - Swimming pool (Olympic size) with changing rooms

    Indoor Sports Complex:
    - Table tennis hall with 10 tables
    - Carrom and chess facilities
    - Billiards and snooker tables
    - Indoor badminton courts (4)
    - Gymnasium with modern equipment
    - Separate fitness center for girls
    - Yoga and meditation hall
    - Aerobics and dance studio

    HEALTH & WELLNESS FACILITIES:

    Medical Center:
    - 24x7 medical facility with qualified doctors
    - Separate male and female medical officers
    - Emergency ambulance service
    - Pharmacy with essential medicines
    - Regular health checkup programs
    - Mental health counseling services
    - First aid training for students and staff

    Healthcare Partnerships:
    - Tie-ups with nearby multi-specialty hospitals
    - Health insurance coverage for all students
    - Specialist consultation arrangements
    - Emergency medical evacuation facilities

    TRANSPORTATION SERVICES:
    - 35 buses covering 50+ routes across the city
    - GPS-enabled buses for route tracking
    - Female attendants in buses for girls' safety
    - Maintenance workshop for bus servicing
    - Real-time bus tracking mobile app
    - Pick-up points at major locations in the city

    ADDITIONAL CAMPUS FACILITIES:

    Banking & Financial Services:
    - On-campus branch of nationalized bank
    - 3 ATM machines at different locations
    - Mobile banking and digital payment facilities
    - Educational loan processing assistance

    Shopping & Services:
    - Campus store for stationery and daily needs
    - Photocopy and printing centers (5 locations)
    - Courier and postal services
    - Mobile phone and laptop repair center
    - Salon and barber shop services

    TECHNOLOGY INFRASTRUCTURE:
    - Campus-wide Wi-Fi with 1 Gbps leased line
    - 24x7 power supply with diesel generator backup
    - Solar power plant generating 500 KW
    - Water treatment plant for clean drinking water
    - Sewage treatment plant for environment protection
    - CCTV surveillance across entire campus
    - Biometric attendance system for students and staff
    - Smart card system for library and mess access

    AUDITORIUM & CONFERENCE FACILITIES:
    - Main auditorium with 1,500 seating capacity
    - AC with modern sound and lighting systems
    - Seminar halls (5) with 100-200 capacity each
    - Conference rooms for meetings and workshops
    - Video conferencing facilities for virtual events
    - Stage and backstage facilities for cultural programs

    ENVIRONMENTAL INITIATIVES:
    - Green campus with 5,000+ trees planted
    - Rainwater harvesting system
    - Waste segregation and recycling programs
    - Solar street lighting across campus
    - Organic waste composting unit
    - Plastic-free campus initiative
    - Regular environmental awareness programs

    SAFETY & SECURITY:
    - 24x7 security personnel at all entry/exit points
    - CCTV monitoring of entire campus
    - Emergency response system with panic buttons
    - Fire safety equipment and regular drills
    - Women's safety committee and measures
    - Visitor management system
    - Night patrol services in hostel areas
    """,
    
    "fee_structure": """
    FEE STRUCTURE & FINANCIAL ASSISTANCE - COMPREHENSIVE GUIDE

    APEX Group of Institutions maintains a transparent and student-friendly fee structure with various payment options and financial assistance programs to ensure quality education is accessible to deserving students.

    DETAILED FEE STRUCTURE (ACADEMIC YEAR 2024-25):

    B.TECH PROGRAMS - ANNUAL FEE BREAKDOWN:

    Computer Science Engineering:
    - Tuition Fee: ₹1,25,000
    - Development Fee: ₹20,000
    - Laboratory Fee: ₹15,000
    - Library Fee: ₹8,000
    - Examination Fee: ₹12,000
    - Registration Fee: ₹5,000 (One time)
    - Total Annual Fee: ₹1,85,000
    - 4-Year Total: ₹7,35,000 (excluding registration fee from 2nd year)

    AI & Machine Learning:
    - Tuition Fee: ₹1,35,000
    - Development Fee: ₹20,000
    - Laboratory Fee: ₹18,000
    - Library Fee: ₹8,000
    - Examination Fee: ₹12,000
    - Registration Fee: ₹5,000 (One time)
    - Total Annual Fee: ₹1,98,000
    - 4-Year Total: ₹7,87,000

    Data Science:
    - Tuition Fee: ₹1,30,000
    - Development Fee: ₹20,000
    - Laboratory Fee: ₹16,000
    - Library Fee: ₹8,000
    - Examination Fee: ₹12,000
    - Registration Fee: ₹5,000 (One time)
    - Total Annual Fee: ₹1,91,000
    - 4-Year Total: ₹7,59,000

    Cloud Technology & Information Security:
    - Tuition Fee: ₹1,30,000
    - Development Fee: ₹20,000
    - Laboratory Fee: ₹17,000
    - Library Fee: ₹8,000
    - Examination Fee: ₹12,000
    - Registration Fee: ₹5,000 (One time)
    - Total Annual Fee: ₹1,92,000
    - 4-Year Total: ₹7,63,000

    MBA PROGRAM - ANNUAL FEE:
    - Tuition Fee: ₹1,50,000
    - Case Study Material Fee: ₹15,000
    - Industry Interface Fee: ₹10,000
    - Library & Digital Resources: ₹10,000
    - Examination Fee: ₹10,000
    - Registration Fee: ₹5,000 (One time)
    - Total Annual Fee: ₹2,00,000
    - 2-Year Total: ₹3,95,000

    OTHER UNDERGRADUATE PROGRAMS:

    BBA (Bachelor of Business Administration):
    - Annual Fee: ₹85,000
    - 3-Year Total: ₹2,50,000

    BCA (Bachelor of Computer Applications):
    - Annual Fee: ₹95,000
    - 3-Year Total: ₹2,80,000

    B.Com (Hons):
    - Annual Fee: ₹75,000
    - 3-Year Total: ₹2,20,000

    DIPLOMA PROGRAMS (Polytechnic):
    - Computer Science Engineering: ₹65,000 per year
    - Electronics & Communication: ₹60,000 per year
    - Mechanical Engineering: ₹60,000 per year
    - Civil Engineering: ₹55,000 per year

    PHARMACY PROGRAMS:
    - D.Pharm (2 years): ₹70,000 per year
    - B.Pharm (4 years): ₹1,10,000 per year

    ADDITIONAL COSTS (OPTIONAL BUT RECOMMENDED):

    HOSTEL ACCOMMODATION:
    Boys Hostel:
    - Single Occupancy: ₹95,000 per year
    - Double Occupancy: ₹75,000 per year
    - Triple Occupancy: ₹65,000 per year

    Girls Hostel:
    - Single Occupancy: ₹1,05,000 per year
    - Double Occupancy: ₹85,000 per year
    - Triple Occupancy: ₹75,000 per year

    MESS CHARGES:
    - Vegetarian Meals: ₹45,000 per year
    - Non-Vegetarian Meals: ₹55,000 per year
    - Special Diet (Jain/Vegan): ₹48,000 per year

    TRANSPORTATION:
    - Within 15 km radius: ₹18,000 per year
    - 15-25 km radius: ₹25,000 per year
    - 25-35 km radius: ₹32,000 per year
    - Beyond 35 km: ₹40,000 per year

    OTHER EXPENSES:
    - Study Materials & Books: ₹12,000-₹15,000 per year
    - Laptop (Recommended): ₹45,000-₹80,000
    - Industry Certification Courses: ₹10,000-₹25,000
    - Cultural and Sports Activities: ₹5,000 per year

    PAYMENT OPTIONS & FACILITIES:

    ANNUAL PAYMENT SCHEME:
    - 5% discount on total annual fees if paid in one installment
    - Payment deadline: Within 15 days of admission confirmation
    - Online payment gateway with multiple banking options
    - Demand draft in favor of "APEX Group of Institutions"

    SEMESTER-WISE PAYMENT:
    - Fees can be paid in two equal installments per year
    - First installment: Before start of odd semester
    - Second installment: Before start of even semester
    - No additional charges for semester-wise payment

    MONTHLY INSTALLMENT PLAN:
    - Available for families with annual income below ₹5 lakhs
    - Fees can be paid in 10 monthly installments
    - Small processing fee of ₹2,000 per year
    - Direct debit facility available
    - Post-dated cheques accepted

    SCHOLARSHIP PROGRAMS & FINANCIAL ASSISTANCE:

    MERIT-BASED SCHOLARSHIPS:

    APEX Excellence Awards:
    - Rank 1-10 in JEE Main/CUET: 100% tuition fee waiver
    - Rank 11-50: 75% tuition fee waiver
    - Rank 51-100: 50% tuition fee waiver
    - Rank 101-500: 25% tuition fee waiver
    - State topper in 12th standard: 100% fee waiver

    Academic Performance Scholarships:
    - CGPA 9.5+ students: ₹25,000 cash award + 50% next year fee waiver
    - CGPA 9.0-9.49: ₹15,000 cash award + 25% next year fee waiver
    - CGPA 8.5-8.99: ₹10,000 cash award + 10% next year fee waiver

    NEED-BASED FINANCIAL AID:

    Income-Based Fee Concessions:
    - Family income below ₹2 lakhs: 75% fee concession
    - Family income ₹2-4 lakhs: 50% fee concession
    - Family income ₹4-6 lakhs: 25% fee concession
    - Single parent families: Additional 10% concession

    GOVERNMENT SCHOLARSHIPS:
    - SC/ST students: As per government norms (₹20,000-₹50,000)
    - OBC students: Non-creamy layer benefit
    - EWS category: 10% reservation with fee benefits
    - Minority community scholarships available
    - State government scholarship schemes applicable

    SPECIAL CATEGORY SCHOLARSHIPS:

    Sports Excellence Scholarships:
    - National level players: 100% fee waiver + monthly stipend
    - State level players: 75% fee waiver
    - District level players: 50% fee waiver
    - University level players: 25% fee waiver

    Cultural Talent Scholarships:
    - National award winners: ₹50,000 annual scholarship
    - State level winners: ₹30,000 annual scholarship
    - District level winners: ₹20,000 annual scholarship

    Other Special Scholarships:
    - Girl child education incentive: 10% fee concession for girls
    - Employee ward benefit: 50% concession for staff children
    - Alumni referral benefit: ₹10,000 cash back
    - Sibling discount: 15% for second child, 25% for third child
    - Defense personnel ward: 20% fee concession

    EDUCATION LOAN ASSISTANCE:
    - Partnership with 15+ nationalized and private banks
    - Loan processing assistance through dedicated counselor
    - Special tie-up with SBI, HDFC, ICICI for quick processing
    - Up to ₹15 lakhs education loan without collateral
    - Competitive interest rates starting from 8.5%
    - Moratorium period until completion of course
    - EMI planning and repayment guidance

    FEE REFUND POLICY:

    Refund Schedule:
    - Withdrawal before classes start: 90% refund
    - Withdrawal within first 15 days: 80% refund
    - Withdrawal within first month: 70% refund
    - Withdrawal within first quarter: 50% refund
    - Withdrawal after first semester: No refund
    - Processing charges: ₹10,000 in all cases
    - Caution deposit: Fully refundable after course completion

    LATE PAYMENT PENALTIES:
    - Late payment fine: ₹100 per day after due date
    - Maximum late fee: ₹5,000 per semester
    - No examination permission without fee clearance
    - Admission cancellation after 60 days of non-payment

    FEE COLLECTION CENTERS:
    - Online payment portal: Available 24x7
    - Campus accounts office: 9 AM to 5 PM (Mon-Sat)
    - Authorized collection centers in major cities
    - Bank branches with direct credit facility
    - Mobile payment apps: PhonePe, GooglePay, Paytm accepted

    CONTACT FOR FEE QUERIES:
    Accounts Department: +91-7351408009 (Ext: 101)
    Fee Helpdesk: fees@apex.ac.in
    Financial Aid Counselor: +91-7351408009 (Ext: 105)
    WhatsApp Support: +91-9876543211
    Office Hours: 9:00 AM to 5:00 PM (Monday to Saturday)
    """,
    
    "contact_information": """
    CONTACT INFORMATION & CAMPUS LOCATIONS

    MAIN CAMPUS ADDRESS:
    APEX Group of Institutions
    NH-44, Sector-125
    Noida, Uttar Pradesh - 201303
    India

    GEOGRAPHIC COORDINATES:
    Latitude: 28.5355° N
    Longitude: 77.3910° E

    MAIN CONTACT NUMBERS:
    General Enquiry: +91-7351408009
    Admission Helpdesk: +91-7351408009 (Ext: 100)
    Academic Office: +91-7351408009 (Ext: 201)
    Training & Placement: +91-7351408009 (Ext: 234)
    Accounts Department: +91-7351408009 (Ext: 101)
    Hostel Office: +91-7351408009 (Ext: 301)
    Transport Office: +91-7351408009 (Ext: 401)
    Library: +91-7351408009 (Ext: 501)

    EMAIL CONTACTS:
    General Information: info@apex.ac.in
    Admissions: admissions@apex.ac.in
    Academic Queries: academics@apex.ac.in
    Placements: placements@apex.ac.in
    Fee Related: fees@apex.ac.in
    Hostel Enquiries: hostel@apex.ac.in
    Alumni Affairs: alumni@apex.ac.in
    International Programs: international@apex.ac.in

    DEPARTMENT-WISE CONTACT:

    ENGINEERING DEPARTMENTS:
    Computer Science Engineering: cse@apex.ac.in
    AI & Machine Learning: aiml@apex.ac.in
    Data Science: ds@apex.ac.in
    Cloud Technology & IT Security: ctis@apex.ac.in

    MANAGEMENT DEPARTMENT:
    MBA Admissions: mba@apex.ac.in
    BBA Programs: bba@apex.ac.in

    OTHER DEPARTMENTS:
    BCA Programs: bca@apex.ac.in
    B.Com Programs: bcom@apex.ac.in
    Polytechnic: diploma@apex.ac.in
    Pharmacy: pharmacy@apex.ac.in

    SENIOR ADMINISTRATION:
    Director General: director@apex.ac.in
    Academic Director: academic.director@apex.ac.in
    Admission Director: admission.director@apex.ac.in
    Dean Student Affairs: dean.students@apex.ac.in

    SOCIAL MEDIA PRESENCE:
    Official Website: www.apex.ac.in
    Facebook: facebook.com/APEXGroupOfInstitutions
    Instagram: @apex_college_official
    LinkedIn: linkedin.com/school/apex-group-of-institutions
    YouTube: APEX Group of Institutions Official
    Twitter: @APEX_College

    WHATSAPP SUPPORT:
    Admission Queries: +91-7351408009
    Fee Related: +91-9876543211
    Academic Support: +91-9876543212
    Placement Support: +91-9876543213
    Hostel Queries: +91-9876543214

    OFFICE HOURS:
    Monday to Friday: 9:00 AM to 6:00 PM
    Saturday: 9:00 AM to 4:00 PM
    Sunday: Closed (Emergency contact available)

    PUBLIC HOLIDAYS: As per Government of Uttar Pradesh calendar

    EMERGENCY CONTACTS:
    Medical Emergency: +91-7351408009 (Ext: 911)
    Security Office: +91-7351408009 (Ext: 100)
    Fire Safety: +91-7351408009 (Ext: 101)
    Campus Maintenance: +91-7351408009 (Ext: 601)

    HOW TO REACH APEX CAMPUS:

    BY AIR:
    - Nearest Airport: Indira Gandhi International Airport, New Delhi (45 km)
    - Travel time from airport: 1.5-2 hours by taxi/cab
    - Airport shuttle service available on request

    BY TRAIN:
    - Nearest Railway Station: New Delhi Railway Station (35 km)
    - Hazrat Nizamuddin Railway Station (30 km)
    - Ghaziabad Railway Station (25 km)
    - Pre-paid taxi and bus services available

    BY METRO:
    - Nearest Metro Station: Noida Sector 137 (Blue Line)
    - Distance from metro station: 3 km
    - Auto-rickshaw and cab services available

    BY BUS:
    - Regular bus services from ISBT Kashmere Gate, Delhi
    - Noida Authority buses connecting major areas
    - Private bus operators serving the route

    BY CAR:
    - Well-connected by National Highway NH-44
    - GPS Navigation: Search "APEX Group of Institutions, Noida"
    - Ample parking space available on campus
    - Toll charges applicable on expressway

    NEARBY LANDMARKS:
    - Amity University (5 km)
    - Noida Stadium (8 km)
    - DLF Mall of India (12 km)
    - Worlds of Wonder (15 km)
    - India Expo Centre & Mart (10 km)

    ACCOMMODATION FOR VISITORS:
    - Guest House on campus (prior booking required)
    - Nearby hotels: Radisson Blu, Country Inn & Suites
    - Budget accommodations available in Sector 18, Noida
    - Airport hotels for international visitors

    BANKING FACILITIES NEARBY:
    - SBI Branch (on campus)
    - HDFC Bank (2 km)
    - ICICI Bank (1.5 km)
    - PNB Branch (3 km)
    - Multiple ATMs within 1 km radius

    MEDICAL FACILITIES NEARBY:
    - Apollo Hospital (8 km)
    - Fortis Hospital (6 km)
    - Max Super Speciality Hospital (7 km)
    - Local clinic and pharmacy (1 km)

    POSTAL ADDRESS FOR CORRESPONDENCE:
    The Registrar
    APEX Group of Institutions
    NH-44, Sector-125
    Noida, Uttar Pradesh - 201303
    India

    COURIER AND PARCEL DELIVERY:
    - Campus address same as postal address
    - Mention department name for quick delivery
    - Student parcels: Include enrollment number
    - Office hours for parcel collection: 10 AM to 5 PM

    GRIEVANCE REDRESSAL:
    Email: grievances@apex.ac.in
    Phone: +91-7351408009 (Ext: 999)
    In-person: Administrative Block, Room 201
    Response time: Within 48 hours for all queries

    For any additional information not covered above, please feel free to contact our 24x7 helpdesk at +91-7351408009. Our dedicated support team is always ready to assist students, parents, and visitors with their queries and concerns.
    """
}

class ComprehensiveAPEXRAG:
    """Comprehensive RAG system with embedded APEX data"""
    
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
            name="apex_embedded_kb"
        )
        
        # Models
        self.embedding_model = "models/text-embedding-004" 
        self.generation_model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Process embedded data
        self._process_embedded_data()
    
    def _test_api(self) -> bool:
        """Test API connection"""
        try:
            genai.embed_content(
                model="models/text-embedding-004",
                content="test",
                task_type="retrieval_query"
            )
            return True
        except Exception as e:
            st.error(f"API Error: {e}")
            return False
    
    def _chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 50) -> List[str]:
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
        
        # Generate embeddings
        st.info("🧠 Generating embeddings...")
        embeddings = self._generate_embeddings(all_chunks)
        progress_bar.progress(0.8)
        
        # Add to ChromaDB
        st.info("💾 Adding to vector database...")
        self._add_to_chromadb(all_chunks, embeddings, all_metadata, all_ids)
        
        progress_bar.progress(1.0)
        st.success(f"✅ Successfully processed {self.collection.count()} chunks!")
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in small batches"""
        embeddings = []
        batch_size = 3  # Very small batches for stability
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = genai.embed_content(
                    model=self.embedding_model,
                    content=batch,
                    task_type="retrieval_document"
                )
                
                # Handle response format
                if isinstance(response['embedding'], list):
                    batch_embeddings = [emb['embedding'] for emb in response['embedding']]
                else:
                    batch_embeddings = [response['embedding']['embedding']]
                
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                st.warning(f"Embedding error for batch {i//batch_size + 1}: {e}")
                # Add dummy embeddings
                embeddings.extend([[0.1] * 768 for _ in batch])
        
        return embeddings
    
    def _add_to_chromadb(self, chunks, embeddings, metadata, ids):
        """Add data to ChromaDB"""
        batch_size = 20
        
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
    
    def query(self, user_question: str, n_results: int = 4) -> Dict:
        """Query the RAG system"""
        try:
            # Generate query embedding
            query_response = genai.embed_content(
                model=self.embedding_model,
                content=user_question,
                task_type="retrieval_query"
            )
            
            # Handle response format
            if isinstance(query_response['embedding'], dict):
                query_embedding = query_response['embedding']['embedding']
            else:
                query_embedding = query_response['embedding']
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()),
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
                    max_output_tokens=800,
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
    .example-btn {
        margin: 5px;
        padding: 10px;
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
    st.markdown('<div class="main-header"><h1>🎓 APEX College AI Assistant</h1><p>Comprehensive Knowledge Base with Embedded Data</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google AI API Key",
            type="password",
            value=os.getenv("GOOGLE_AI_API_KEY", ""),
            help="Get your API key from https://ai.google.dev/"
        )
        
        if not api_key:
            st.error("⚠️ Please enter your Google AI API Key")
            st.info("Get your free API key from [Google AI Studio](https://ai.google.dev/)")
            st.stop()
        
        st.divider()
        
        # Initialize RAG system
        if st.session_state.rag_system is None:
            with st.spinner("🚀 Initializing comprehensive APEX knowledge base..."):
                try:
                    st.session_state.rag_system = ComprehensiveAPEXRAG(api_key)
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
        "How do I contact APEX college?",
        "Tell me about the faculty and teaching quality",
        "What extracurricular activities are available?"
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
