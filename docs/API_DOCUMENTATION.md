# Financial Data Transformation Platform (FTT-ML) API Documentation

## 🚀 Overview

The FTT-ML API provides comprehensive financial data processing capabilities including file upload, transformation, reconciliation, and AI-powered features.

## 🔗 Base URLs

- **Development**: `http://localhost:8000`
- **Production**: `https://api.example.com`

## 🔑 Authentication

Currently using API key authentication for LLM providers. Set your OpenAI API key in the `.env` file:

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

## 📖 Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📋 API Endpoints Overview

### Health & Status
- `GET /health` - System health check
- `GET /debug/status` - Detailed system diagnostics
- `GET /performance/metrics` - Performance metrics

### File Management
- `POST /upload` - Upload CSV/Excel files
- `GET /files/{file_id}` - Get file information
- `GET /files/{file_id}/preview` - Preview file data
- `DELETE /files/{file_id}` - Delete uploaded file

### Data Transformation
- `POST /transformation/process/` - Process data transformation
- `POST /transformation/generate-config/` - AI-generated configuration
- `GET /transformation/results/{transformation_id}` - Get transformation results
- `GET /transformation/download/{transformation_id}` - Download results
- `GET /transformation/health` - Transformation service health

### Reconciliation
- `POST /reconciliation/process/` - Process financial reconciliation
- `GET /reconciliation/results/{reconciliation_id}` - Get reconciliation results
- `POST /reconciliation/rules/` - Create reconciliation rules

### Delta Generation
- `POST /delta/process/` - Generate delta between file versions
- `GET /delta/results/{delta_id}` - Get delta results

### AI Assistance
- `POST /ai-assistance/generate-regex/` - AI-powered regex generation
- `POST /ai-assistance/analyze-data/` - AI data analysis

## 🔧 Common Request/Response Patterns

### Standard Success Response
```json
{
  "success": true,
  "message": "Operation completed successfully",
  "data": { ... }
}
```

### Standard Error Response
```json
{
  "success": false,
  "message": "Error description",
  "detail": "Detailed error information"
}
```

## 📊 File Upload Format

### Supported File Types
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)

### File Size Limits
- Default: 500MB
- Large file threshold: 100,000 rows
- Configurable via `MAX_FILE_SIZE` environment variable

### Upload Response
```json
{
  "success": true,
  "message": "File uploaded successfully",
  "data": {
    "file_id": "file_12345",
    "filename": "data.csv",
    "file_size_mb": 2.5,
    "total_rows": 1000,
    "columns": ["id", "name", "amount"],
    "file_type": "csv"
  }
}
```

## 🤖 AI-Powered Features

### Configuration Generation
The API can automatically generate transformation configurations using AI:

```bash
POST /transformation/generate-config/
Content-Type: application/json

{
  "requirements": "Transform customer data to include full name and calculate total with tax",
  "source_files": [
    {
      "file_id": "file_12345",
      "filename": "customers.csv",
      "columns": ["first_name", "last_name", "amount", "tax_rate"],
      "totalRows": 1000
    }
  ]
}
```

### Expression Support
The transformation system supports expressions with `{column_name}` syntax:

- **Mathematical**: `{quantity} * {unit_price}`
- **String concatenation**: `{first_name} {last_name}`
- **Complex calculations**: `{amount} * (1 + {tax_rate}/100)`

## 📈 Performance Optimization

### Batch Processing
The API is optimized for large datasets:

- **Recommended batch size**: 1000 records
- **Maximum concurrent operations**: 5
- **Memory optimization**: Automatic cleanup after 30 minutes

### Large File Handling
For files with 50k+ records:

- Use paginated result retrieval
- Enable streaming downloads
- Consider chunked processing

## 🔍 Error Handling

### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 400 | Bad Request | Check request format and required fields |
| 404 | Not Found | Verify file IDs and endpoint URLs |
| 413 | File Too Large | Reduce file size or increase limits |
| 422 | Validation Error | Check data types and field constraints |
| 500 | Internal Server Error | Check logs and system status |

### LLM Service Errors

| Error | Description | Solution |
|-------|-------------|----------|
| LLM service not configured | OpenAI API key missing | Set `OPENAI_API_KEY` in `.env` |
| Model not available | Specified model not accessible | Check model name and API limits |
| Rate limit exceeded | Too many API calls | Implement backoff strategy |

## 📋 Environment Configuration

### Required Environment Variables
```bash
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-3.5-turbo

# Application Settings
DEBUG=false
SECRET_KEY=your-secret-key-here

# File Processing
MAX_FILE_SIZE=500
BATCH_SIZE=100
TEMP_DIR=/tmp

# Performance
CHUNK_SIZE=10000
MEMORY_LIMIT_GB=4
```

### Optional Environment Variables
```bash
# LLM Provider Configuration
LLM_PROVIDER=openai
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key

# Database (future use)
DATABASE_URL=sqlite:///./app.db

# Logging
LOG_LEVEL=INFO

# CORS
ALLOWED_HOSTS=*
```

## 🧪 Testing

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### File Upload Test
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@test_data.csv"
```

### System Status
```bash
curl -X GET "http://localhost:8000/debug/status"
```

## 📚 Advanced Usage Examples

See the `examples/` directory for:
- Complete workflow examples
- Python client library usage
- Integration patterns
- Performance tuning guides

## 🆘 Support

- **Documentation**: Check `/docs` and `/redoc` endpoints
- **Health Status**: Monitor `/health` endpoint
- **Debug Information**: Use `/debug/status` for troubleshooting
- **Performance Metrics**: Check `/performance/metrics`

## 📄 Changelog

### Version 4.1.0
- Added pluggable LLM service architecture
- Enhanced expression evaluation system
- Improved API documentation
- Added comprehensive health checks
- Performance optimizations for large datasets

### Version 4.0.0
- AI-powered configuration generation
- Enhanced transformation rules
- Improved error handling
- Added comprehensive logging