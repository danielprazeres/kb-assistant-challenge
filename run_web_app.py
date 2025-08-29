#!/usr/bin/env python3
"""Simple script to run the Streamlit web application."""

import subprocess
import sys
import os

def main():
    """Run the Streamlit web application."""
    try:
        # Check if .env file exists
        if not os.path.exists('.env'):
            print("⚠️  Warning: .env file not found. Please create one from env.example")
            print("   cp env.example .env")
            print("   Then add your OpenAI API key to the .env file")
            print()
        
        # Run Streamlit
        print("🚀 Starting Matrix Script Assistant...")
        print("📱 Web interface will be available at: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop")
        print()
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "kbac/web_app.py",
            "--server.port=8501",
            "--server.address=localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
