"""Startup script to run both backend and frontend."""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def check_env_file():
    """Check if .env file exists and has API key."""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found. Please copy .env.example to .env and add your Google API key.")
        return False
    
    content = env_file.read_text()
    if "your_google_api_key_here" in content:
        print("⚠️  Please edit .env file and add your actual Google API key.")
        return False
    
    return True


def start_backend():
    """Start the FastAPI backend."""
    print("🚀 Starting FastAPI backend...")
    return subprocess.Popen([
        sys.executable, "run_backend.py"
    ])


def start_frontend():
    """Start the Streamlit frontend."""
    print("🎨 Starting Streamlit frontend...")
    return subprocess.Popen([
        "streamlit", "run", "app.py"
    ])


def main():
    """Main startup function."""
    print("🤖 AI Document Q&A System Startup")
    print("=" * 40)
    
    # Check environment
    if not check_env_file():
        return
    
    try:
        # Start backend
        backend_process = start_backend()
        print("⏳ Waiting for backend to start...")
        time.sleep(3)
        
        # Start frontend
        frontend_process = start_frontend()
        print("⏳ Waiting for frontend to start...")
        time.sleep(3)
        
        # Open browser
        print("🌐 Opening browser...")
        webbrowser.open("http://localhost:8501")
        
        print("\n✅ System started successfully!")
        print("📊 Backend API: http://localhost:8000")
        print("🎨 Frontend UI: http://localhost:8501")
        print("📚 API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop both services")
        
        # Wait for processes
        try:
            backend_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
            backend_process.terminate()
            frontend_process.terminate()
            print("✅ Services stopped")
            
    except Exception as e:
        print(f"❌ Error starting services: {e}")


if __name__ == "__main__":
    main()
