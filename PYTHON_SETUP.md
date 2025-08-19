# Python 3.12 Setup Guide

## Issue Resolution
The pandas compilation error occurs because Python 3.13 is too new and doesn't have compatible pre-built wheels for many packages including pandas. We've updated the project to use Python 3.12.

## Setup Steps

### Option 1: Using pyenv (Recommended)
```bash
# Install pyenv if not already installed
curl https://pyenv.run | bash

# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# Reload shell
source ~/.bashrc

# Install Python 3.12
pyenv install 3.12.7

# Set as global or local version
pyenv global 3.12.7  # Sets globally
# OR
pyenv local 3.12.7   # Sets for this project only
```

### Option 2: Using conda
```bash
# Create new environment with Python 3.12
conda create -n prism-ai-backend python=3.12

# Activate environment
conda activate prism-ai-backend

# Install poetry in the environment
pip install poetry
```

### Option 3: Using system Python 3.12
```bash
# Install Python 3.12 using Homebrew (macOS)
brew install python@3.12

# Create virtual environment
python3.12 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install poetry
pip install poetry
```

## Install Dependencies

Once you have Python 3.12 set up:

```bash
# Navigate to backend directory
cd /Users/biswambarpradhan/UpSkill/prism-ai-backend

# Install dependencies with Poetry
poetry install

# OR if using pip directly
pip install -r requirements.txt
```

## Verify Installation

```bash
# Check Python version
python --version  # Should show Python 3.12.x

# Test the installation
poetry run python -c "import pandas; print('Pandas version:', pandas.__version__)"

# Run a simple test
poetry run python -c "from app.main import app; print('FastAPI app imported successfully')"
```

## Common Issues and Solutions

### Issue: Poetry not finding Python 3.12
```bash
# Tell Poetry which Python to use
poetry env use python3.12

# OR specify full path
poetry env use /path/to/python3.12
```

### Issue: Package conflicts
```bash
# Clear Poetry cache
poetry cache clear pypi --all

# Remove lock file and reinstall
rm poetry.lock
poetry install
```

### Issue: Still getting compilation errors
```bash
# Update pip and setuptools first
pip install --upgrade pip setuptools wheel

# Install with pre-built wheels only (no compilation)
pip install --only-binary=all -r requirements.txt
```

## Development Commands

After successful installation:

```bash
# Start development server
make dev-start

# OR directly with poetry
poetry run python -m uvicorn app.main:app --reload

# Run tests
make test

# Run with coverage
make test-coverage
```

## Docker Alternative

If you continue to have issues with local Python setup, you can use Docker:

```bash
# Build and run with Docker (uses Python 3.12 in container)
docker build -t prism-ai-backend .
docker run -p 8000:8000 prism-ai-backend

# OR use development compose
docker-compose -f docker-compose.dev.yml up
```

## Next Steps

1. Set up Python 3.12 using one of the methods above
2. Install dependencies with Poetry
3. Run the development server
4. Test the API at http://localhost:8000/docs

The pandas compilation issue should be resolved with Python 3.12!