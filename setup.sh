#!/bin/bash
# FTT-ML Backend Setup Script
# Automated setup for development environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
check_python() {
    print_status "Checking Python version..."
    if command_exists python3.11; then
        PYTHON_CMD="python3.11"
    elif command_exists python3; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [[ "$PYTHON_VERSION" < "3.11" ]]; then
            print_warning "Python 3.11+ recommended, found $PYTHON_VERSION"
        fi
    else
        print_error "Python 3 not found. Please install Python 3.11+"
        exit 1
    fi
    print_success "Python found: $($PYTHON_CMD --version)"
}

# Install Poetry
install_poetry() {
    print_status "Checking Poetry installation..."
    if ! command_exists poetry; then
        print_status "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
        if ! command_exists poetry; then
            print_error "Poetry installation failed"
            exit 1
        fi
    fi
    print_success "Poetry found: $(poetry --version)"
}

# Setup development environment
setup_dev_environment() {
    print_status "Setting up development environment..."
    
    # Install dependencies
    print_status "Installing dependencies with Poetry..."
    poetry install
    
    # Install pre-commit hooks
    print_status "Installing pre-commit hooks..."
    poetry run pre-commit install
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        print_status "Creating .env file from template..."
        cat > .env << EOL
# Development Environment Configuration
ENVIRONMENT=development
DEBUG=true
HOST=0.0.0.0
PORT=8000
SERVER_URL=http://localhost:8000

# OpenAI Configuration (add your key)
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4-turbo
OPENAI_MAX_TOKENS=4000
OPENAI_TEMPERATURE=0.1

# Application Settings
APP_NAME=Financial Data Extraction API
SECRET_KEY=dev-secret-key-change-in-production

# File Processing
TEMP_DIR=/tmp
BATCH_SIZE=100
MAX_FILE_SIZE=500

# CORS Settings (development)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173,http://localhost:5174

# Logging
LOG_LEVEL=debug

# Threading
FTT_ML_CORES=4

# Storage
STORAGE_TYPE=local
EOL
        print_warning "Created .env file. Please update with your actual values."
    else
        print_success ".env file already exists"
    fi
}

# Run tests
run_tests() {
    print_status "Running initial tests..."
    if poetry run python test/run_tests.py --maxfail=5; then
        print_success "Initial tests passed"
    else
        print_warning "Some tests failed. Check the output above."
    fi
}

# Run linting
run_linting() {
    print_status "Running code quality checks..."
    
    # Format code
    poetry run black app/ test/ --check || {
        print_status "Formatting code with black..."
        poetry run black app/ test/
    }
    
    # Sort imports
    poetry run isort app/ test/ --check-only || {
        print_status "Sorting imports with isort..."
        poetry run isort app/ test/
    }
    
    # Run flake8
    if poetry run flake8 app/ --count --select=E9,F63,F7,F82 --show-source --statistics; then
        print_success "Linting passed"
    else
        print_warning "Linting issues found. Run 'make lint' for details."
    fi
}

# Create directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p temp
    mkdir -p logs
    mkdir -p data
    print_success "Directories created"
}

# Check system requirements
check_system() {
    print_status "Checking system requirements..."
    
    # Check available memory
    if command_exists free; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$MEMORY_GB" -lt 4 ]; then
            print_warning "Less than 4GB RAM detected. Performance may be limited."
        else
            print_success "Memory: ${MEMORY_GB}GB"
        fi
    fi
    
    # Check CPU cores
    if command_exists nproc; then
        CORES=$(nproc)
        print_success "CPU cores: $CORES"
        if [ "$CORES" -lt 4 ]; then
            print_warning "Less than 4 CPU cores. Consider adjusting FTT_ML_CORES in .env"
        fi
    fi
    
    # Check disk space
    if command_exists df; then
        DISK_SPACE=$(df -h . | awk 'NR==2{print $4}')
        print_success "Available disk space: $DISK_SPACE"
    fi
}

# Main setup function
main() {
    echo -e "${BLUE}"
    cat << "EOF"
    ███████╗████████╗████████╗      ███╗   ███╗██╗     
    ██╔════╝╚══██╔══╝╚══██╔══╝      ████╗ ████║██║     
    █████╗     ██║      ██║   █████╗██╔████╔██║██║     
    ██╔══╝     ██║      ██║   ╚════╝██║╚██╔╝██║██║     
    ██║        ██║      ██║         ██║ ╚═╝ ██║███████╗
    ╚═╝        ╚═╝      ╚═╝         ╚═╝     ╚═╝╚══════╝
                                                        
    Backend Development Environment Setup
EOF
    echo -e "${NC}"
    
    # Check if we're in the right directory
    if [ ! -f "pyproject.toml" ]; then
        print_error "pyproject.toml not found. Please run this script from the backend directory."
        exit 1
    fi
    
    # Run setup steps
    check_python
    install_poetry
    check_system
    create_directories
    setup_dev_environment
    run_linting
    run_tests
    
    echo ""
    print_success "Setup completed successfully!"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Update .env file with your OpenAI API key"
    echo "2. Run 'make dev' to start the development server"
    echo "3. Visit http://localhost:8000/docs for API documentation"
    echo ""
    echo -e "${BLUE}Useful commands:${NC}"
    echo "  make dev          - Start development server"
    echo "  make test         - Run tests"
    echo "  make lint         - Run linting checks"
    echo "  make help         - Show all available commands"
    echo ""
    echo -e "${GREEN}Happy coding! 🚀${NC}"
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        echo "FTT-ML Backend Setup Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h      Show this help message"
        echo "  --test-only     Run tests only (skip setup)"
        echo "  --install-only  Install dependencies only"
        echo ""
        exit 0
        ;;
    --test-only)
        check_python
        run_tests
        exit 0
        ;;
    --install-only)
        check_python
        install_poetry
        setup_dev_environment
        exit 0
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac