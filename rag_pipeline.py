import json
import os
from typing import List, Dict, Tuple
import google.generativeai as genai
import chromadb
import numpy as np
import pandas as pd

class GoogleAIRAGPipeline:
    """RAG Pipeline using Google AI for both embeddings and generation"""
    
    def __init__(self, api_key: str, collection_name: str = "apex_knowledge_base"):
        """Initialize RAG pipeline with Google AI"""
        self.api_key = api_key
        self.collection_name = collection_name
        
        # Configure Google AI
        genai.configure(api_key=api_key)
        
        # Initialize ChromaDB with new client approach
        try:
            # Use PersistentClient (new way)
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            print("✅ ChromaDB PersistentClient initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing ChromaDB: {e}")
            raise
        
        # Initialize embedding model
        self.embedding_model = "models/text-embedding-004"
        
        # Initialize generation model
        self.generation_model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name
            )
            print(f"✅ Collection '{collection_name}' ready!")
        except Exception as e:
            print(f"❌ Error with collection: {e}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks for better retrieval"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find end position
            end = start + chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at sentence boundary
            chunk = text[start:end]
            
            # Look for sentence endings
            sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
            best_break = -1
            
            for ending in sentence_endings:
                pos = chunk.rfind(ending)
                if pos > len(chunk) * 0.7:  # Don't break too early
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
            
            # Ensure we don't go backwards
            start = max(start, len(chunks[-1]) if chunks else 0)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Google AI embedding model"""
        try:
            embeddings = []
            
            # Process in smaller batches to be safe with API limits
            batch_size = 10  # Reduce batch size for stability
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                print(f"🔄 Processing embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                # Call Google AI embedding API
                response = genai.embed_content(
                    model=self.embedding_model,
                    content=batch,
                    task_type="retrieval_document"
                )
                
                # Handle single text vs batch response
                if isinstance(response['embedding'], list):
                    # Batch response
                    batch_embeddings = [emb['embedding'] for emb in response['embedding']]
                else:
                    # Single response
                    batch_embeddings = [response['embedding']['embedding']]
                
                embeddings.extend(batch_embeddings)
            
            print(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            # Fallback: create dummy embeddings for testing
            print("🔄 Using dummy embeddings for testing...")
            return [[0.1] * 768 for _ in texts]
    
    def process_documents(self, documents: List[Dict]):
        """Process and index documents into ChromaDB"""
        print(f"🔄 Processing {len(documents)} documents...")
        
        # Clear existing data
        try:
            existing_count = self.collection.count()
            if existing_count > 0:
                print(f"🗑️ Clearing {existing_count} existing documents...")
                # Get all IDs and delete them
                all_data = self.collection.get()
                if all_data['ids']:
                    self.collection.delete(ids=all_data['ids'])
        except Exception as e:
            print(f"⚠️ Warning clearing collection: {e}")
        
        all_chunks = []
        all_metadata = []
        all_ids = []
        
        for doc_id, doc in enumerate(documents):
            title = doc.get('title', 'Untitled')
            content = doc.get('content', '')
            url = doc.get('url', '')
            
            # Create comprehensive content for chunking
            full_content = f"Title: {title}\n\nContent: {content}"
            
            # Chunk the document
            chunks = self.chunk_text(full_content)
            
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
        
        print(f"📝 Created {len(all_chunks)} chunks from documents")
        
        # Generate embeddings
        print("🔄 Generating embeddings...")
        embeddings = self.generate_embeddings(all_chunks)
        
        # Add to ChromaDB in smaller batches
        print("💾 Adding to vector database...")
        batch_size = 50  # Add in smaller batches
        
        for i in range(0, len(all_chunks), batch_size):
            end_idx = min(i + batch_size, len(all_chunks))
            batch_chunks = all_chunks[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_metadata = all_metadata[i:end_idx]
            batch_ids = all_ids[i:end_idx]
            
            try:
                self.collection.add(
                    documents=batch_chunks,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                print(f"✅ Added batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
            except Exception as e:
                print(f"❌ Error adding batch: {e}")
        
        final_count = self.collection.count()
        print(f"✅ Successfully indexed {final_count} chunks!")
    
    def retrieve_relevant_chunks(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve relevant chunks for a query"""
        try:
            # Generate query embedding
            query_response = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )
            
            # Handle response format
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
                n_results=min(n_results, self.collection.count()),
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            relevant_chunks = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    relevant_chunks.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    })
            
            return relevant_chunks
            
        except Exception as e:
            print(f"❌ Error retrieving chunks: {e}")
            return []
    
    def generate_context_prompt(self, query: str, relevant_chunks: List[Dict]) -> str:
        """Generate context-aware prompt for the LLM"""
        
        context_parts = []
        for chunk in relevant_chunks:
            title = chunk['metadata'].get('title', 'Unknown')
            content = chunk['content']
            score = chunk['similarity_score']
            
            context_parts.append(f"Source: {title} (Relevance: {score:.3f})\nContent: {content}\n")
        
        context = "\n---\n".join(context_parts)
        
        prompt = f"""You are APEX College Assistant, a helpful AI assistant for APEX Group of Institutions. 
Use the following context information to answer the user's question about APEX College.

CONTEXT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Use the context information to provide accurate, helpful answers about APEX College
- If the answer is not fully covered in the context, clearly state what information is available and suggest contacting the college for more details
- Be friendly, professional, and encouraging
- Focus on being helpful to prospective students and their families
- Include relevant contact information when appropriate: Phone: +91-7351408009, Email: admissions@apex.ac.in
- Provide specific details from the context when available (programs, fees, admission process, etc.)

ANSWER:"""
        
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """Generate answer using Gemini model"""
        try:
            response = self.generation_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    top_p=0.9,
                    max_output_tokens=1024,
                )
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return f"I apologize, but I'm having trouble generating a response right now. Please try again or contact APEX College directly at +91-7351408009 for immediate assistance."
    
    def query(self, user_question: str, n_results: int = 5) -> Dict:
        """Main query function - retrieve relevant content and generate answer"""
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(user_question, n_results)
        
        if not relevant_chunks:
            return {
                'answer': "I don't have specific information about that topic in my knowledge base. Please contact APEX College at +91-7351408009 or admissions@apex.ac.in for detailed information.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Generate context-aware prompt
        prompt = self.generate_context_prompt(user_question, relevant_chunks)
        
        # Generate answer
        answer = self.generate_answer(prompt)
        
        # Extract source information
        sources = []
        for chunk in relevant_chunks:
            source_info = {
                'title': chunk['metadata'].get('title', 'Unknown'),
                'url': chunk['metadata'].get('url', ''),
                'similarity': chunk['similarity_score']
            }
            if source_info not in sources:
                sources.append(source_info)
        
        avg_confidence = np.mean([chunk['similarity_score'] for chunk in relevant_chunks]) if relevant_chunks else 0.0
        
        return {
            'answer': answer,
            'sources': sources[:3],  # Limit to top 3 sources
            'confidence': float(avg_confidence),
            'retrieved_chunks': len(relevant_chunks)
        }
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model,
                'status': 'ready'
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

def load_scraped_data(file_path: str = "apex_college_data.json") -> List[Dict]:
    """Load scraped college data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} documents from {file_path}")
        return data
    except FileNotFoundError:
        print(f"❌ File {file_path} not found!")
        return []
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return []

# Example usage and testing
if __name__ == "__main__":
    # This would be run to set up the knowledge base
    API_KEY = os.getenv("GOOGLE_AI_API_KEY", "your-api-key-here")
    
    if API_KEY == "your-api-key-here":
        print("❌ Please set your GOOGLE_AI_API_KEY environment variable")
    else:
        # Initialize RAG pipeline
        try:
            rag = GoogleAIRAGPipeline(API_KEY)
            
            # Load and process documents
            documents = load_scraped_data()
            if documents:
                rag.process_documents(documents)
                
                # Test queries
                test_queries = [
                    "What B.Tech programs does APEX offer?",
                    "How can I apply for admission to APEX?",
                    "What are the placement opportunities?",
                ]
                
                print("\n🧪 Testing RAG Pipeline:")
                for query in test_queries:
                    print(f"\n❓ Query: {query}")
                    result = rag.query(query)
                    print(f"📝 Answer: {result['answer'][:200]}...")
                    print(f"🎯 Confidence: {result['confidence']:.3f}")
                    print(f"📚 Sources: {len(result['sources'])}")
            else:
                print("❌ No documents to process!")
        except Exception as e:
            print(f"❌ Failed to initialize RAG pipeline: {e}")
            print("💡 Make sure you have a valid Google AI API key and internet connection")
