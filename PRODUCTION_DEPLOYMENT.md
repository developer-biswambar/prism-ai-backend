# Production Deployment Guide

## 🚀 Production-Ready Configuration

Your backend is now production-ready with all localhost references removed and configurable via environment variables.

## 📋 Required Environment Variables

### Core Configuration
```bash
# Application
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000
SERVER_URL=https://your-api-domain.com
SECRET_KEY=your-super-secret-production-key

# OpenAI
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4-turbo

# CORS (Security)
ALLOWED_ORIGINS=https://your-frontend.com,https://admin.yourdomain.com

# Logging
LOG_LEVEL=info
DEBUG=false
```

## 🐳 Docker Deployment

### Option 1: Environment Variables in Docker Run
```bash
docker run -d \
  --name ftt-ml-backend \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e SERVER_URL=https://your-api-domain.com \
  -e OPENAI_API_KEY=sk-your-key \
  -e ALLOWED_ORIGINS=https://your-frontend.com \
  -e SECRET_KEY=your-production-secret \
  -e LOG_LEVEL=info \
  -e DEBUG=false \
  your-image:tag
```

### Option 2: Environment File
```bash
# Create production.env file with your values
docker run -d \
  --name ftt-ml-backend \
  -p 8000:8000 \
  --env-file production.env \
  your-image:tag
```

### Option 3: Docker Compose
```yaml
version: '3.8'
services:
  backend:
    image: your-ftt-ml-backend:latest
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - SERVER_URL=https://your-api-domain.com
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ALLOWED_ORIGINS=${ALLOWED_ORIGINS}
      - SECRET_KEY=${SECRET_KEY}
      - LOG_LEVEL=info
      - DEBUG=false
    volumes:
      - /app/temp:/app/temp
    restart: unless-stopped
```

## ☁️ ECS Deployment

### Task Definition Environment Variables
```json
{
  "environment": [
    {"name": "ENVIRONMENT", "value": "production"},
    {"name": "HOST", "value": "0.0.0.0"},
    {"name": "PORT", "value": "8000"},
    {"name": "SERVER_URL", "value": "https://your-api-domain.com"},
    {"name": "OPENAI_API_KEY", "value": "sk-your-key"},
    {"name": "ALLOWED_ORIGINS", "value": "https://your-frontend.com"},
    {"name": "SECRET_KEY", "value": "your-production-secret"},
    {"name": "LOG_LEVEL", "value": "info"},
    {"name": "DEBUG", "value": "false"},
    {"name": "FTT_ML_CORES", "value": "16"}
  ]
}
```

## 🔧 Performance Optimization

### For High-Performance Servers
```bash
# Override CPU cores for better threading
FTT_ML_CORES=32

# Increase memory limits
MEMORY_LIMIT_GB=16

# Larger batch sizes for high-memory systems
BATCH_SIZE=200
```

### For c5.4xlarge (16 vCPU, 32GB RAM)
```bash
ENVIRONMENT=production
FTT_ML_CORES=14  # Leave 2 cores for system
MEMORY_LIMIT_GB=28
BATCH_SIZE=150
MAX_FILE_SIZE=1000
```

## 🛡️ Security Best Practices

### 1. CORS Configuration
```bash
# ❌ Development (insecure)
ALLOWED_ORIGINS=*

# ✅ Production (secure)
ALLOWED_ORIGINS=https://your-frontend.com,https://admin.yourdomain.com
```

### 2. Secret Management
```bash
# Generate strong secret key
SECRET_KEY=$(openssl rand -hex 32)

# Use AWS Secrets Manager or similar for OPENAI_API_KEY
```

### 3. Debug Mode
```bash
# ❌ Never in production
DEBUG=true

# ✅ Always in production
DEBUG=false
LOG_LEVEL=info
```

## 📊 Health Checks

### Application Health Check
```bash
curl https://your-api-domain.com/health
```

### Expected Response
```json
{
  "status": "healthy",
  "timestamp": "2025-01-14T12:00:00.000000",
  "version": "2.0.0",
  "llm_configured": true
}
```

## 🔍 Monitoring

### Key Endpoints to Monitor
- `GET /health` - Application health
- `GET /debug/status` - Detailed system status
- `GET /performance/metrics` - Performance metrics

### Important Metrics
- Response time < 500ms for health checks
- Memory usage < 80% of allocated
- CPU usage patterns during reconciliation
- File upload success rates

## 🚨 Troubleshooting

### Common Issues

1. **CORS Errors**
   ```bash
   # Check ALLOWED_ORIGINS configuration
   echo $ALLOWED_ORIGINS
   ```

2. **API Documentation Not Loading**
   ```bash
   # Verify SERVER_URL is set correctly
   echo $SERVER_URL
   ```

3. **Threading Issues**
   ```bash
   # Check detected cores vs override
   curl https://your-api-domain.com/debug/status | jq '.data.threading_config'
   ```

## 📁 Configuration Files

### Use These Files:
- `.env.production` - Production environment variables
- `docker.env.example` - Docker environment template

### Remove These Files in Production:
- `.env` (development only)
- Any files with `localhost` references

## ✅ Production Checklist

- [ ] All localhost references removed
- [ ] Environment variables configured
- [ ] CORS properly restricted
- [ ] Debug mode disabled
- [ ] Secret key generated
- [ ] Health checks working
- [ ] Logging configured
- [ ] Threading optimized for your server
- [ ] SSL/HTTPS configured
- [ ] Monitoring setup

## 🚀 Ready for Deployment!

Your backend is now production-ready and will automatically configure itself based on environment variables. No more hardcoded localhost references!