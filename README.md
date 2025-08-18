# Prism AI Backend

A high-performance financial data processing API built with FastAPI, designed for financial data extraction, reconciliation, transformation, and delta analysis.

## 🚀 **Quick Start**

### **Development**
```bash
# Clone the repository
git clone <repository-url>
cd prism-ai-backend

# Install dependencies with Poetry
make install

# Run development server
make dev-start
```

### **Production**
```bash
# Build and run with Docker
docker build -t prism-ai-backend .
docker run -p 8000:8000 prism-ai-backend

# Or use Docker Compose
docker-compose up -d
```

## 🏗️ **Architecture**

### **Core Features**
- **Financial Data Reconciliation** - Match transactions between multiple data sources
- **Data Transformation** - Transform and restructure financial datasets
- **Delta Generation** - Compare file versions to identify changes  
- **AI-Powered Processing** - OpenAI integration for intelligent data analysis
- **High Performance** - Optimized for 50k-100k record datasets

### **Technology Stack**
- **API Framework**: FastAPI with async support
- **Data Processing**: Pandas with vectorized operations
- **AI Integration**: OpenAI GPT models
- **Servers**: Uvicorn (ASGI) / Waitress (WSGI)
- **Testing**: pytest with comprehensive coverage
- **Build**: Poetry + Makefile automation

## 📚 **API Documentation**

### **Key Endpoints**
- **Health**: `GET /health` - System status
- **File Management**: `POST /upload`, `GET /files/{id}`
- **Reconciliation**: `POST /reconciliation/process/`
- **Transformation**: `POST /transformation/process/`  
- **Delta Generation**: `POST /delta/process/`
- **AI Assistance**: `POST /ai-assistance/generic-call`

### **Interactive Documentation**
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## ⚙️ **Configuration**

### **Environment Variables**
```bash
# Application
HOST=0.0.0.0
PORT=8000
DEBUG=false
SECRET_KEY=your-secret-key

# OpenAI Integration
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4-turbo

# Performance
BATCH_SIZE=100
MAX_FILE_SIZE=500
FTT_ML_CORES=8

# CORS (comma-separated)
ALLOWED_ORIGINS=https://your-frontend-domain.com
```

### **Server Options**

#### **Uvicorn (Default - ASGI)**
```bash
# Development
make dev-start

# Production
make production-start
```

#### **Waitress (WSGI Alternative)**
```bash
# Development
make waitress-dev

# Production  
make waitress-prod
```

## 🧪 **Testing**

### **Run Tests**
```bash
# All tests with coverage
make test-coverage

# Specific test categories
make test-file-upload
make test-reconciliation
make test-transformation

# Performance tests
make test-performance
```

### **Test Categories**
- **Unit Tests**: Core business logic
- **Integration Tests**: API endpoint testing
- **Performance Tests**: Large dataset handling
- **AI Tests**: OpenAI integration validation

## 🐳 **Docker Deployment**

### **Standard Deployment**
```bash
# Build image
docker build -t prism-ai-backend .

# Run container
docker run -d \
  --name prism-ai-backend \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -e ALLOWED_ORIGINS=https://your-domain.com \
  prism-ai-backend
```

### **Waitress Deployment**
```bash
# Build Waitress variant
docker build -f Dockerfile.waitress -t prism-ai-backend:waitress .

# Run with Waitress
docker run -d \
  --name prism-ai-backend-waitress \
  -p 8000:8000 \
  -e WAITRESS_THREADS=8 \
  -e OPENAI_API_KEY=your-key \
  prism-ai-backend:waitress
```

### **Docker Compose**
```bash
# Production deployment
docker-compose up -d

# Development with hot reload
docker-compose -f docker-compose.dev.yml up
```

## 📦 **Build Commands**

```bash
# Development
make install              # Install dependencies
make dev-start           # Start development server
make dev-stop            # Stop development server

# Testing
make test                # Run all tests
make test-coverage       # Run tests with coverage
make test-watch          # Watch mode testing

# Production
make build               # Build for production
make production-start    # Start production server
make waitress-prod       # Start with Waitress

# Docker
make docker-build        # Build Docker image
make docker-run          # Run Docker container
make docker-stop         # Stop Docker container

# Quality
make lint                # Run linting
make format              # Format code
make type-check          # Type checking
make security-check      # Security audit

# Utilities
make clean               # Clean temporary files
make logs                # View application logs
make shell               # Poetry shell
make requirements        # Generate requirements.txt
```

## 🔧 **Development**

### **Project Structure**
```
app/
├── main.py              # FastAPI application entry
├── routes/              # API route handlers
│   ├── file_routes.py
│   ├── reconciliation_routes.py
│   ├── transformation_routes.py
│   └── delta_routes.py
├── services/            # Business logic
│   ├── file_service.py
│   ├── reconciliation_service.py
│   └── openai_service.py
├── models/              # Pydantic models
├── utils/               # Utility functions
└── server.py            # Waitress server setup

test/
├── run_tests.py         # Custom test runner
├── unit/                # Unit tests
├── integration/         # Integration tests
└── performance/         # Performance tests
```

### **Adding New Features**
1. **Routes**: Add endpoint in `app/routes/`
2. **Services**: Implement business logic in `app/services/`
3. **Models**: Define data models in `app/models/`
4. **Tests**: Add tests in `test/` with appropriate markers

### **Performance Optimization**
- **Batch Processing**: Configurable batch sizes for large datasets
- **Caching**: Redis integration for frequently accessed data
- **Async Operations**: Full async/await support
- **Memory Management**: Optimized DataFrame operations

## 🔒 **Security**

### **Security Features**
- **CORS Configuration**: Strict origin controls
- **Input Validation**: Pydantic model validation
- **File Upload Security**: Type and size restrictions
- **API Key Management**: Secure OpenAI key handling
- **Container Security**: Non-root user execution

### **Production Security**
```bash
# Set secure environment variables
export SECRET_KEY=$(openssl rand -hex 32)
export ALLOWED_ORIGINS=https://your-production-domain.com
export DEBUG=false

# Use HTTPS in production
export SERVER_URL=https://api.your-domain.com
```

## 📊 **Monitoring**

### **Health Checks**
```bash
# Application health
curl http://localhost:8000/health

# Detailed system status
curl http://localhost:8000/debug/status

# Performance metrics
curl http://localhost:8000/performance/metrics
```

### **Logging**
- **Structured Logging**: JSON format with correlation IDs
- **Log Levels**: Configurable via `LOG_LEVEL` environment variable
- **Performance Tracking**: Request timing and resource usage

## 🤝 **Contributing**

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### **Development Guidelines**
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Use type hints
- Add docstrings for public functions

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 **Support**

- **Documentation**: [API Documentation](http://localhost:8000/docs)
- **Issues**: [GitHub Issues](https://github.com/your-org/prism-ai-backend/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/prism-ai-backend/discussions)

---

**Prism AI Backend** - Powering intelligent financial data processing 🚀