# File Processing Testing Guide

This folder contains comprehensive testing resources for File Upload, Processing, and Management features.

## 📁 Folder Contents

### Documentation Files
- **README.md** - This guide
- **file_formats_guide.md** - Supported file formats and specifications
- **upload_performance_guide.md** - Performance testing for file uploads
- **error_handling_guide.md** - Error scenarios and handling

### Test Data Files
- **fileA.csv** - Basic CSV file for upload testing
- **fileB.csv** - Secondary CSV file for multi-file operations
- **sample_trades.csv** - Financial trading data sample
- **large_file_test.csv** - Large file for performance testing (10K+ records)
- **malformed_data_test.csv** - File with data quality issues
- **unicode_test.csv** - File with special characters and Unicode
- **empty_file.csv** - Empty file for edge case testing

## 🚀 Quick Start Testing

### 1. Basic File Upload Test
```bash
# Start the backend server
cd /path/to/backend
python app/main.py

# Test file upload via API
curl -X POST "http://localhost:8000/upload" \
  -F "file=@docs/testing/file-processing/fileA.csv" \
  -F "label=Test File A"
```

### 2. File Processing Features Covered

#### File Upload
- **Single File Upload**: Standard CSV file upload
- **Multiple File Upload**: Batch file processing
- **Large File Handling**: Files > 100MB
- **Format Validation**: CSV, Excel (xlsx, xls) support
- **Error Handling**: Invalid formats, corrupted files

#### File Management
- **File Metadata**: Storage of file information and statistics
- **File Preview**: Sample data display and column detection
- **File Validation**: Data quality checks and format verification
- **File Cleanup**: Temporary file management

## 🧪 File Processing Workflows

### Manual Testing Steps

1. **Basic File Upload**
   - Navigate to file upload interface
   - Select fileA.csv from test folder
   - Verify successful upload and metadata display
   - Check file preview functionality

2. **Multi-File Upload**
   - Upload multiple files simultaneously
   - Verify individual file processing
   - Test file labeling and organization
   - Validate batch processing capabilities

3. **Large File Processing**
   - Upload large_file_test.csv (10K+ records)
   - Monitor memory usage and processing time
   - Verify streaming upload functionality
   - Test progress tracking

4. **Error Handling**
   - Upload malformed_data_test.csv
   - Test unsupported file formats
   - Verify error messages and recovery
   - Test empty file handling

### API Testing

```bash
# Basic file upload
curl -X POST "http://localhost:8000/upload" \
  -F "file=@docs/testing/file-processing/fileA.csv" \
  -F "label=Basic Test File"

# Get file information
curl -X GET "http://localhost:8000/files/{file_id}/info"

# Get file preview
curl -X GET "http://localhost:8000/files/{file_id}/preview?limit=10"

# Delete file
curl -X DELETE "http://localhost:8000/files/{file_id}"

# Upload multiple files
curl -X POST "http://localhost:8000/upload-multiple" \
  -F "files=@docs/testing/file-processing/fileA.csv" \
  -F "files=@docs/testing/file-processing/fileB.csv"
```

## 📊 Test File Descriptions

### fileA.csv
- **Records**: 100 sample transactions
- **Columns**: id, date, amount, description, category
- **Use Case**: Basic upload and processing testing
- **Size**: ~5KB
- **Encoding**: UTF-8

### fileB.csv
- **Records**: 150 sample transactions  
- **Columns**: transaction_id, timestamp, value, notes, type
- **Use Case**: Multi-file operations and column mapping
- **Size**: ~7KB
- **Encoding**: UTF-8

### sample_trades.csv
- **Records**: 1,000 trading transactions
- **Columns**: trade_id, symbol, quantity, price, currency, trade_date
- **Use Case**: Financial data processing
- **Size**: ~50KB
- **Encoding**: UTF-8

### large_file_test.csv
- **Records**: 50,000 transactions
- **Columns**: Multiple financial data columns
- **Use Case**: Performance and memory testing
- **Size**: ~15MB
- **Encoding**: UTF-8

## 🎯 File Processing Test Scenarios

### 1. Standard File Upload
```json
{
  "scenario": "Basic CSV file upload and processing",
  "test_file": "fileA.csv",
  "expected_results": {
    "upload_status": "success",
    "records_detected": 100,
    "columns_detected": 5,
    "file_size": "~5KB",
    "processing_time": "< 2 seconds"
  },
  "validations": [
    "File metadata correctly stored",
    "Column names properly detected",
    "Data types inferred correctly",
    "Preview data displays properly"
  ]
}
```

### 2. Large File Performance
```json
{
  "scenario": "Large file upload and processing",
  "test_file": "large_file_test.csv",
  "performance_targets": {
    "upload_time": "< 30 seconds",
    "memory_usage": "< 200MB peak",
    "processing_time": "< 60 seconds",
    "response_time": "< 5 seconds for metadata"
  },
  "features_tested": [
    "Streaming upload",
    "Progressive processing",
    "Memory management",
    "Progress tracking"
  ]
}
```

### 3. Error Handling
```json
{
  "scenario": "File processing error handling",
  "test_cases": [
    {
      "file": "malformed_data_test.csv",
      "expected_error": "Data format validation error",
      "error_handling": "Graceful error message with details"
    },
    {
      "file": "empty_file.csv",
      "expected_error": "Empty file error",
      "error_handling": "Clear error message and suggested actions"
    },
    {
      "file": "unsupported_format.txt",
      "expected_error": "Unsupported file format",
      "error_handling": "List of supported formats provided"
    }
  ]
}
```

## 📈 Performance Benchmarks

### Upload Performance
- **Small Files** (< 1MB): < 2 seconds
- **Medium Files** (1-10MB): < 10 seconds
- **Large Files** (10-100MB): < 60 seconds
- **Very Large Files** (100MB+): < 5 minutes

### Processing Performance
- **Column Detection**: < 1 second
- **Data Type Inference**: < 2 seconds
- **Preview Generation**: < 3 seconds
- **Metadata Storage**: < 1 second

### Memory Usage
- **Baseline**: ~25MB
- **Small File Processing**: ~50MB
- **Large File Processing**: ~200MB peak
- **Multiple Files**: +50MB per additional file

## 🔍 Testing Checklist

### File Upload Features
- [ ] Single file upload
- [ ] Multiple file upload
- [ ] Drag and drop functionality
- [ ] Progress tracking
- [ ] File size validation
- [ ] Format validation
- [ ] Error handling

### File Processing Features
- [ ] Column detection
- [ ] Data type inference
- [ ] Record counting
- [ ] Data preview generation
- [ ] Metadata extraction
- [ ] Encoding detection
- [ ] Duplicate detection

### File Management Features
- [ ] File listing
- [ ] File information retrieval
- [ ] File deletion
- [ ] File labeling
- [ ] Storage cleanup
- [ ] Temporary file management
- [ ] Concurrent access handling

### Performance Features
- [ ] Large file handling
- [ ] Memory optimization
- [ ] Streaming processing
- [ ] Background processing
- [ ] Progress reporting
- [ ] Cancellation support
- [ ] Resource monitoring

## 🚦 Success Criteria

### Functional Requirements
- ✅ Support CSV and Excel formats
- ✅ Handle files up to 500MB
- ✅ Automatic column and data type detection
- ✅ Comprehensive error handling
- ✅ File preview and metadata

### Performance Requirements
- ✅ Upload 100MB files within 60 seconds
- ✅ Process 100K records within 2 minutes
- ✅ Memory usage under 500MB for large files
- ✅ Support 10 concurrent uploads
- ✅ 99.9% upload success rate

### Quality Requirements
- ✅ 100% accuracy for column detection
- ✅ 95% accuracy for data type inference
- ✅ Proper handling of special characters
- ✅ Graceful error recovery
- ✅ Data integrity preservation

## 🔧 Advanced File Processing Testing

### Stress Testing
```bash
# Upload multiple large files simultaneously
python scripts/stress_test_upload.py \
  --files docs/testing/file-processing/large_file_test.csv \
  --concurrent 5 \
  --iterations 10
```

### Memory Leak Testing
```bash
# Monitor memory usage during repeated uploads
python scripts/memory_test.py \
  --test_file docs/testing/file-processing/large_file_test.csv \
  --iterations 100 \
  --monitor_memory
```

### Format Compatibility Testing
```bash
# Test various file formats and encodings
python scripts/format_test.py \
  --input_dir docs/testing/file-processing/ \
  --test_encodings utf-8,latin-1,cp1252 \
  --test_formats csv,xlsx,xls
```

## 🔍 Troubleshooting File Processing

### Common Issues

1. **Upload Failures**
   - Check file size limits (default: 500MB)
   - Verify supported file formats
   - Ensure stable network connection
   - Monitor server disk space

2. **Processing Errors**
   - Validate file encoding (UTF-8 recommended)
   - Check for malformed CSV data
   - Verify column header format
   - Review data type consistency

3. **Performance Issues**
   - Monitor server memory usage
   - Check for concurrent upload limits
   - Verify file storage configuration
   - Review processing batch sizes

4. **Memory Problems**
   - Implement streaming processing
   - Adjust chunk sizes
   - Enable garbage collection
   - Monitor memory leaks

### Debug Configuration
```bash
# Enable file processing debugging
export FILE_DEBUG=true
export LOG_LEVEL=DEBUG
export MAX_FILE_SIZE=500  # MB
export TEMP_DIR=/tmp/ftt-ml
python app/main.py
```

## 📋 File Processing Quality Assurance

### Automated Testing
```bash
# Run comprehensive file processing tests
python -m pytest tests/file_processing/ -v --cov=app.routes.file_routes

# Test specific scenarios
python -m pytest tests/file_processing/test_large_files.py
python -m pytest tests/file_processing/test_error_handling.py
```

### Data Validation Testing
```bash
# Validate processed file data
python scripts/validate_file_processing.py \
  --input docs/testing/file-processing/fileA.csv \
  --expected_columns 5 \
  --expected_records 100
```

### Integration Testing
```bash
# Test file processing with other features
python scripts/integration_test.py \
  --upload_file docs/testing/file-processing/sample_trades.csv \
  --test_reconciliation \
  --test_transformation \
  --test_delta
```

## 📊 File Processing Metrics

### Success Metrics
```json
{
  "upload_success_rate": 0.999,
  "processing_accuracy": 0.995,
  "avg_upload_time": 15.2,
  "avg_processing_time": 8.7,
  "memory_efficiency": 0.92,
  "user_satisfaction": 4.3
}
```

### Error Metrics
```json
{
  "common_errors": {
    "file_too_large": 0.02,
    "unsupported_format": 0.005,
    "malformed_data": 0.003,
    "network_timeout": 0.001
  },
  "error_recovery_rate": 0.98,
  "avg_error_resolution_time": 12.5
}
```

## 📞 File Processing Support

### Resources
- File Upload API: `/docs/API_DOCUMENTATION.md#file-upload`
- Supported Formats: `/docs/file_formats_guide.md`
- Performance Tuning: `/docs/performance/file_processing.md`

### Contact Information
- File Processing Issues: file-support@company.com
- Performance Problems: performance-team@company.com
- API Questions: api-support@company.com

---

**Last Updated**: December 2024  
**Version**: 2.0.0  
**Supported Formats**: CSV, XLSX, XLS  
**Max File Size**: 500MB  
**Maintainer**: FTT-ML File Processing Team