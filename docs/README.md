# FTT-ML API Documentation

## 📚 Documentation Overview

This directory contains comprehensive documentation for the Financial Data Transformation Platform (FTT-ML) API.

## 📋 Documentation Files

### Core Documentation
- **[API_DOCUMENTATION.md](./API_DOCUMENTATION.md)** - Complete API documentation with endpoints, authentication, and usage
- **[API_SCHEMAS.md](./API_SCHEMAS.md)** - Detailed request/response schemas and validation rules

### Workflow Examples
- **[examples/transformation_workflow.md](./examples/transformation_workflow.md)** - Complete transformation workflow examples
- **[examples/reconciliation_workflow.md](./examples/reconciliation_workflow.md)** - Financial reconciliation workflow examples
- **[examples/python_client.py](./examples/python_client.py)** - Python client library with examples

### Testing Resources
- **[testing/](./testing/)** - Comprehensive testing documentation and sample data organized by feature
- **[postman/FTT-ML_API_Collection.json](./postman/FTT-ML_API_Collection.json)** - Postman collection for API testing

## 🚀 Quick Start

1. **Start the API Server**
   ```bash
   cd backend
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Access Interactive Documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

3. **Test the API**
   - Import the Postman collection from `postman/FTT-ML_API_Collection.json`
   - Or use the Python client examples in `examples/python_client.py`

## 📖 Documentation Structure

### 1. Health & Status Endpoints
- System health checks
- Performance metrics
- Debug information

### 2. File Management
- File upload (CSV, Excel)
- File information retrieval
- Data preview

### 3. Data Transformation
- AI-powered configuration generation
- Rule-based transformation processing
- Expression evaluation
- Result download in multiple formats

### 4. Financial Reconciliation
- Multi-criteria matching
- Tolerance-based comparison
- Unmatched record identification

### 5. Delta Generation
- File version comparison
- Change tracking
- New/modified/deleted record identification

### 6. AI Assistance
- Regex pattern generation
- Data structure analysis
- Intelligent processing suggestions

## 🔧 API Features

### Core Capabilities
- ✅ **High Performance** - Optimized for 50k-100k record datasets
- ✅ **Pluggable Architecture** - Swap LLM providers without code changes
- ✅ **Expression Support** - Mathematical and string expressions with `{column_name}` syntax
- ✅ **Comprehensive Validation** - Request/response validation with detailed error messages
- ✅ **Multiple Export Formats** - CSV, Excel, JSON download options

### AI-Powered Features
- ✅ **Smart Configuration Generation** - AI generates transformation rules from natural language
- ✅ **Dynamic Conditions** - Complex conditional logic for data transformation
- ✅ **Regex Generation** - AI-powered pattern generation for data extraction
- ✅ **Data Analysis** - Intelligent data structure analysis and suggestions

## 🔑 Authentication

Currently using API key authentication for LLM providers. Set your OpenAI API key in the `.env` file:

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

## 📊 Request/Response Patterns

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

## 🧪 Testing & Validation

### Comprehensive Testing Suite
The **[testing/](./testing/)** directory provides organized testing resources:

- **[Reconciliation Testing](../test_docs/reconciliation/)** - Financial data matching scenarios
- **[Transformation Testing](../test_docs/transformation/)** - Data transformation workflows  
- **[Delta Generation Testing](../test_docs/delta/)** - File comparison and change detection
- **[AI Features Testing](./testing/ai-features/)** - AI-powered automation testing
- **[File Processing Testing](../test_docs/file-processing/)** - Upload and processing validation

Each testing folder includes:
- ✅ Comprehensive test scenarios and expected results
- ✅ Sample data files for different use cases
- ✅ Performance benchmarks and validation criteria
- ✅ Step-by-step testing workflows
- ✅ Troubleshooting guides and best practices

### Interactive Testing
1. **Swagger UI**: http://localhost:8000/docs
   - Interactive API explorer
   - Built-in request/response validation
   - Schema documentation

2. **ReDoc**: http://localhost:8000/redoc
   - Clean, readable documentation
   - Schema references
   - Example payloads

### Testing Tools
1. **Postman Collection**: Import `postman/FTT-ML_API_Collection.json` for:
   - Pre-configured requests for all endpoints
   - Environment variables for easy testing
   - Automated test assertions
   - Response data extraction

2. **Python Client**: Use `examples/python_client.py` for:
   - Complete workflow examples
   - Error handling patterns
   - Batch processing examples
   - Integration patterns

3. **Feature-Specific Testing**: Navigate to `testing/[feature]/` for:
   - Dedicated test data and scenarios
   - Performance benchmarking
   - Error condition validation
   - End-to-end workflow testing

## 📈 Performance Guidelines

### File Size Limits
- **Default**: 500MB per file
- **Large file threshold**: 100,000 rows
- **Recommended batch size**: 1000 records

### Optimization Tips
1. **Use pagination** for large result sets
2. **Enable preview mode** for testing configurations
3. **Batch process** multiple files efficiently
4. **Monitor memory usage** during large operations

### Performance Monitoring
- **Health endpoint**: `/health`
- **Performance metrics**: `/performance/metrics`
- **Debug status**: `/debug/status`

## 🔍 Schema Validation

All API endpoints use Pydantic models for:
- **Request validation** - Automatic validation of incoming data
- **Response serialization** - Consistent response format
- **Type safety** - Runtime type checking
- **Documentation generation** - Auto-generated schema docs

### Common Validation Rules
- File IDs must start with "file_" prefix
- Column names must exist in source data
- Expressions must use valid `{column_name}` syntax
- Numeric values must be valid numbers
- Dates support multiple formats with auto-detection

## 🆘 Troubleshooting

### Common Issues
1. **File not found**: Verify file_id from upload response
2. **Column not found**: Check column names match source data exactly
3. **Expression errors**: Validate mathematical expressions with test data
4. **LLM service errors**: Ensure OpenAI API key is configured
5. **Timeout errors**: Reduce batch size for large datasets

### Debug Endpoints
- **System status**: `GET /debug/status`
- **Performance metrics**: `GET /performance/metrics`
- **Health check**: `GET /health`
- **Transformation health**: `GET /transformation/health`

### Error Codes
| Code | Description | Solution |
|------|-------------|----------|
| 400 | Bad Request | Check request format and required fields |
| 404 | Not Found | Verify file IDs and endpoint URLs |
| 413 | File Too Large | Reduce file size or increase limits |
| 422 | Validation Error | Check data types and field constraints |
| 500 | Internal Server Error | Check logs and system status |

## 📞 Support

### Getting Help
1. **Check the documentation** in this directory
2. **Use interactive docs** at `/docs` and `/redoc`
3. **Monitor health endpoints** for system status
4. **Review example workflows** for common patterns

### Contributing
1. Update documentation when adding new features
2. Include examples for new endpoints
3. Update Postman collection with new requests
4. Add schema validation for new models

## 📄 Version Information

- **API Version**: 4.1.0
- **Documentation Version**: 4.1.0
- **Last Updated**: January 2024

### Recent Changes
- Added pluggable LLM service architecture
- Enhanced expression evaluation system
- Improved comprehensive API documentation
- Added request/response schema validation
- Enhanced error handling and debugging

## 📚 Additional Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Pydantic Documentation**: https://pydantic-docs.helpmanual.io/
- **OpenAPI Specification**: https://swagger.io/specification/
- **Postman Documentation**: https://learning.postman.com/docs/