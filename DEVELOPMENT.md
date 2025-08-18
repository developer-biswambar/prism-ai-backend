# FTT-ML Backend Development Guide

## 🚀 **Quick Start**

### **Automatic Setup**
```bash
# Run the setup script (recommended)
./setup.sh

# Or manually with Make
make dev-install
```

### **Manual Setup**
```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Start development server
make dev
```

## 📦 **Dependency Management**

### **Using Poetry (Recommended)**
```bash
# Add new dependency
poetry add fastapi

# Add development dependency
poetry add --group dev pytest

# Update dependencies
poetry update

# Export to requirements.txt
poetry export -f requirements.txt --output requirements.txt
```

### **Using pip (Fallback)**
```bash
# Install from requirements.txt
make install-legacy

# Add new dependency
echo "new-package==1.0.0" >> requirements.txt
pip install -r requirements.txt
```

## 🔨 **Build & Development Commands**

### **Development Server**
```bash
make dev              # Start with auto-reload
make dev-start        # Start development server
make start            # Start production mode
make production-start # Start with multiple workers
```

### **Testing**
```bash
make test                # Run all tests
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-coverage      # With coverage report
make test-parallel      # Run tests in parallel
make test-reconciliation # Reconciliation tests
make test-file-upload   # File upload tests
make test-viewer        # Viewer tests
```

### **Code Quality**
```bash
make lint           # Run linting checks
make format         # Format code (black + isort)
make format-check   # Check formatting
make security       # Security checks
make quality        # All quality checks
make pre-commit     # Run pre-commit hooks
```

### **Building & Deployment**
```bash
make build          # Build application
make docker-build   # Build Docker image
make docker-run     # Build and run container
make clean          # Clean temporary files
```

## 📁 **Project Structure**

```
backend/
├── app/                    # Main application code
│   ├── config/            # Configuration modules
│   ├── models/            # Pydantic models
│   ├── routes/            # API route definitions
│   ├── services/          # Business logic services
│   ├── utils/             # Utility functions
│   └── main.py           # FastAPI application
├── test/                  # Test files
├── docs/                  # Documentation
├── pyproject.toml         # Poetry configuration
├── requirements.txt       # Pip requirements (auto-generated)
├── Dockerfile             # Docker configuration
├── Dockerfile.poetry      # Docker with Poetry
├── Makefile              # Build automation
├── setup.sh              # Development setup script
└── .pre-commit-config.yaml # Git hooks configuration
```

## 🧪 **Testing Strategy**

### **Test Categories**
- **Unit Tests**: Individual function testing
- **Integration Tests**: API endpoint testing
- **File Upload Tests**: File processing testing
- **Reconciliation Tests**: Core business logic
- **Viewer Tests**: Data viewing functionality

### **Running Specific Tests**
```bash
# By marker
pytest -m "unit"
pytest -m "integration"
pytest -m "reconciliation"

# By file
pytest test/test_file_routes.py

# With coverage
pytest --cov=app --cov-report=html
```

### **Custom Test Runner**
```bash
# Use the custom test runner
python test/run_tests.py --coverage
python test/run_tests.py --parallel
python test/run_tests.py -m reconciliation
```

## 🐳 **Docker Development**

### **Standard Dockerfile**
```bash
# Build and run with pip
make docker-build
make docker-run
```

### **Poetry Dockerfile**
```bash
# Build with Poetry (optimized)
docker build -f Dockerfile.poetry -t ftt-ml-backend:poetry .
docker run -p 8000:8000 ftt-ml-backend:poetry
```

### **Docker Compose**
```bash
# From project root
docker-compose up -d backend
```

## 🔧 **Configuration Management**

### **Environment Variables**
```bash
# Development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=debug

# Production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info
```

### **Key Configuration Files**
- `.env` - Environment variables
- `pyproject.toml` - Poetry and tool configuration
- `app/config/` - Application configuration modules

## 🎯 **Code Quality Standards**

### **Formatting**
- **Black**: Code formatting (line length 100)
- **isort**: Import sorting
- **Pre-commit hooks**: Automatic formatting

### **Linting**
- **flake8**: Style guide enforcement
- **mypy**: Type checking
- **bandit**: Security linting

### **Security**
- **bandit**: Security vulnerability scanning
- **safety**: Dependency vulnerability checking
- **Pre-commit hooks**: Automated security checks

## 📊 **Performance Optimization**

### **Threading Configuration**
```bash
# Set cores for your machine
export FTT_ML_CORES=8

# Or in .env file
FTT_ML_CORES=8
```

### **Memory Settings**
```bash
# For large datasets
MEMORY_LIMIT_GB=16
BATCH_SIZE=200
```

### **Performance Testing**
```bash
make performance-test
```

## 🔍 **Debugging**

### **Development Debugging**
```bash
# Start with debug logging
LOG_LEVEL=debug make dev

# Health checks
make health
make health-debug
```

### **Application Monitoring**
```bash
# Check status
curl http://localhost:8000/health
curl http://localhost:8000/debug/status

# Performance metrics
curl http://localhost:8000/performance/metrics
```

## 📚 **API Documentation**

### **Interactive Documentation**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### **Generating Documentation**
```bash
make docs  # Show documentation URLs
```

## 🚢 **Deployment**

### **Local Deployment**
```bash
make production-start
```

### **Docker Deployment**
```bash
make docker-build-prod
make docker-run-prod
```

### **Environment Setup**
```bash
# Staging
make deploy-staging

# Production
make deploy-production
```

## 🛠️ **Common Development Tasks**

### **Adding New Dependencies**
```bash
# Production dependency
poetry add new-package

# Development dependency
poetry add --group dev new-dev-package

# Update requirements.txt
make update-deps
```

### **Adding New API Routes**
1. Create route file in `app/routes/`
2. Add route logic
3. Include router in `app/main.py`
4. Add tests in `test/`
5. Update documentation

### **Adding New Services**
1. Create service file in `app/services/`
2. Implement business logic
3. Add service to routes
4. Add unit tests
5. Update integration tests

### **Database Changes** (if applicable)
```bash
make db-reset  # Reset database
```

## 🐛 **Troubleshooting**

### **Common Issues**

1. **Import Errors**
   ```bash
   # Check Python path
   export PYTHONPATH=/app
   
   # Reinstall dependencies
   poetry install
   ```

2. **Port Already in Use**
   ```bash
   # Kill process on port 8000
   lsof -ti:8000 | xargs kill -9
   ```

3. **Docker Build Issues**
   ```bash
   # Clean Docker cache
   make clean-docker
   ```

4. **Poetry Issues**
   ```bash
   # Clear Poetry cache
   poetry cache clear --all pypi
   
   # Reinstall
   rm poetry.lock
   poetry install
   ```

### **Performance Issues**
```bash
# Check system resources
make env-check

# Monitor performance
make performance-test
```

## 📈 **Monitoring & Logs**

### **Application Logs**
```bash
# View logs
tail -f logs/app.log

# Docker logs
make docker-logs
```

### **Health Monitoring**
```bash
# Automated health checks
make health

# Detailed status
make health-debug
```

## 🤝 **Contributing**

### **Development Workflow**
1. Create feature branch
2. Make changes
3. Run `make quality` (lint, format, security)
4. Run `make test`
5. Commit changes (pre-commit hooks run automatically)
6. Push and create PR

### **Code Standards**
- Follow Black formatting
- Add type hints
- Write tests for new features
- Update documentation
- Follow security best practices

## 📞 **Support**

### **Getting Help**
```bash
make help          # Show all available commands
make status        # Show current configuration
make env-check     # Check environment setup
```

### **Useful URLs**
- Health: http://localhost:8000/health
- API Docs: http://localhost:8000/docs
- Debug Status: http://localhost:8000/debug/status

Your backend is now ready for scalable development with Poetry, comprehensive testing, and automated quality checks! 🚀