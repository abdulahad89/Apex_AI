#!/bin/bash

# APEX College Chatbot Setup Script
# This script sets up the complete environment for the chatbot

set -e  # Exit on any error

echo "🎓 APEX College AI Chatbot Setup"
echo "================================"

# Check Python version
python_version=$(python3 --version 2>/dev/null || echo "not found")
echo "📋 Python version: $python_version"

if [[ $python_version == "not found" ]]; then
    echo "❌ Python 3 is required but not found. Please install Python 3.8+ first."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Create directories
echo "📁 Creating directories..."
mkdir -p data
mkdir -p chroma_db

# Copy environment template
echo "⚙️ Setting up environment template..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file. Please add your Google AI API key!"
else
    echo "ℹ️ .env file already exists"
fi

# Run initial data collection
echo "🕷️ Running initial data collection..."
python web_scraper.py

echo ""
echo "✅ Setup Complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Edit .env file and add your Google AI API key"
echo "2. Run: streamlit run streamlit_app.py"
echo "3. Open browser to http://localhost:8501"
echo ""
echo "🔑 Get your free Google AI API key from: https://ai.google.dev/"
echo ""
echo "🎉 Enjoy your APEX College AI Assistant!"