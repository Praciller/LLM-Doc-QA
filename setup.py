"""Setup script for the AI Document Q&A System."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def check_uv_installed():
    """Check if uv is installed."""
    result = run_command("uv --version", "Checking uv installation")
    if result is None:
        print("📦 Installing uv...")
        run_command("pip install uv", "Installing uv")


def install_dependencies():
    """Install project dependencies."""
    run_command("uv pip install -e .", "Installing project dependencies")
    run_command("uv pip install -e \".[dev]\"", "Installing development dependencies")


def setup_environment():
    """Set up environment file."""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_example.exists() and not env_file.exists():
        print("📝 Creating .env file from template...")
        env_file.write_text(env_example.read_text())
        print("⚠️  Please edit .env file and add your Google API key")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("❌ .env.example file not found")


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    result = run_command("pytest", "Running test suite")
    if result:
        print("✅ All tests passed")
    else:
        print("⚠️  Some tests failed - check the output above")


def main():
    """Main setup function."""
    print("🚀 Setting up AI Document Q&A System")
    print("=" * 50)
    
    # Check prerequisites
    check_python_version()
    check_uv_installed()
    
    # Install dependencies
    install_dependencies()
    
    # Setup environment
    setup_environment()
    
    # Run tests
    run_tests()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Edit .env file and add your Google API key")
    print("2. Start the backend: python run_backend.py")
    print("3. Start the frontend: streamlit run app.py")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
