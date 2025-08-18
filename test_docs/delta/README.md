# Delta Generation Testing Guide

This folder contains comprehensive testing resources for the Delta Generation feature - comparing file versions to identify changes.

## 📁 Folder Contents

### Documentation Files
- **README.md** - This guide
- **delta_scenarios.md** - Detailed delta generation test scenarios
- **comparison_algorithms.md** - Documentation of comparison algorithms

### Test Data Files
- **delta_fileA_old.csv** - Original version of the dataset
- **delta_fileB_new.csv** - Updated version of the dataset
- **large_dataset_v1.csv** - Large dataset original version (10K+ records)
- **large_dataset_v2.csv** - Large dataset updated version
- **financial_positions_old.csv** - Financial positions baseline
- **financial_positions_new.csv** - Financial positions updated

## 🚀 Quick Start Testing

### 1. Basic Delta Generation Test
```bash
# Start the backend server
cd /path/to/backend
python app/main.py

# Upload both versions via API or frontend
# Use delta_fileA_old.csv as "Old Version"
# Use delta_fileB_new.csv as "New Version"
```

### 2. Test Scenarios Covered

#### Change Detection Types
- **New Records**: Records present in new version only
- **Deleted Records**: Records present in old version only  
- **Modified Records**: Records with changed values
- **Unchanged Records**: Records with identical values

#### Comparison Methods
- **Exact Matching**: Byte-for-byte comparison
- **Key-based Comparison**: Compare using primary keys
- **Tolerance Matching**: Allow minor differences (amounts, dates)
- **Semantic Comparison**: Intelligent field-level analysis

## 🧪 Testing Workflows

### Manual Testing Steps

1. **Upload File Versions**
   - Navigate to delta generation interface
   - Upload delta_fileA_old.csv as "Original Version"
   - Upload delta_fileB_new.csv as "Updated Version"

2. **Configure Comparison Settings**
   - Select primary key columns (e.g., transaction_id)
   - Set tolerance levels for numeric fields
   - Choose comparison algorithm
   - Enable/disable semantic analysis

3. **Execute Delta Generation**
   - Run comparison process
   - Review change summary statistics
   - Examine detailed change records

4. **Analyze Results**
   - Validate detected changes
   - Review change categories
   - Export delta report

### API Testing

```bash
# Upload original version
curl -X POST "http://localhost:8000/upload" \
  -F "file=@docs/testing/delta/delta_fileA_old.csv" \
  -F "label=Original Version"

# Upload updated version  
curl -X POST "http://localhost:8000/upload" \
  -F "file=@docs/testing/delta/delta_fileB_new.csv" \
  -F "label=Updated Version"

# Generate delta
curl -X POST "http://localhost:8000/delta/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "old_file_id": "old_file_id",
    "new_file_id": "new_file_id",
    "primary_keys": ["transaction_id"],
    "tolerance": {
      "amount": 0.01,
      "percentage": 0.001
    }
  }'
```

## 📊 Sample Data Descriptions

### delta_fileA_old.csv (Original Version)
- **Records**: 1,000 financial transactions
- **Columns**: transaction_id, amount, currency, status, date, counterparty
- **Baseline Data**: Original state for comparison
- **Key Field**: transaction_id

### delta_fileB_new.csv (Updated Version)
- **Records**: 1,050 financial transactions  
- **Changes**: 50 new records, 25 modified, 10 deleted
- **Modifications**: Status updates, amount corrections, date changes
- **Key Field**: transaction_id

### Expected Delta Results
```json
{
  "summary": {
    "total_old_records": 1000,
    "total_new_records": 1050,
    "unchanged_records": 915,
    "modified_records": 25,
    "new_records": 50,
    "deleted_records": 10
  },
  "change_types": {
    "status_changes": 15,
    "amount_corrections": 8,
    "date_adjustments": 2
  }
}
```

## 🎯 Common Delta Scenarios

### 1. Financial Position Changes
```json
{
  "scenario": "Daily position file comparison",
  "old_file": "financial_positions_old.csv",
  "new_file": "financial_positions_new.csv",
  "primary_keys": ["account_id", "security_id"],
  "expected_changes": {
    "new_positions": 25,
    "closed_positions": 12,
    "quantity_changes": 150,
    "price_updates": 800
  }
}
```

### 2. Trade Settlement Updates
```json
{
  "scenario": "Trade settlement status tracking",
  "comparison_focus": "status_field_changes",
  "tolerance_settings": {
    "settlement_amount": 0.01,
    "ignore_fields": ["timestamp", "update_user"]
  },
  "expected_patterns": {
    "pending_to_settled": 75,
    "failed_settlements": 3,
    "amount_corrections": 5
  }
}
```

### 3. Large Dataset Performance
```json
{
  "scenario": "Performance test with large datasets",
  "old_file": "large_dataset_v1.csv",
  "new_file": "large_dataset_v2.csv",
  "records": 50000,
  "performance_targets": {
    "processing_time": "< 120 seconds",
    "memory_usage": "< 500MB",
    "accuracy": "100%"
  }
}
```

## 📈 Performance Benchmarks

### Expected Performance
- **Small Dataset** (< 1K records): < 5 seconds
- **Medium Dataset** (1K-10K records): < 30 seconds
- **Large Dataset** (10K-50K records): < 2 minutes
- **Enterprise Dataset** (50K+ records): < 5 minutes

### Memory Usage
- **Baseline**: ~100MB
- **10K records**: ~200MB
- **50K records**: ~600MB
- **100K records**: ~1GB

### Accuracy Benchmarks
- **Exact Matches**: 100% accuracy
- **Tolerance Matches**: 99.9% accuracy
- **False Positives**: < 0.1%
- **False Negatives**: < 0.05%

## 🔍 Testing Checklist

### Core Functionality
- [ ] File upload and validation
- [ ] Primary key configuration
- [ ] Change detection accuracy
- [ ] Tolerance settings
- [ ] Report generation
- [ ] Export functionality

### Change Detection Types
- [ ] New record identification
- [ ] Deleted record identification
- [ ] Modified record detection
- [ ] Field-level change tracking
- [ ] Percentage-based changes
- [ ] Date/time comparisons

### Performance Testing
- [ ] Large file processing
- [ ] Memory efficiency
- [ ] Processing speed
- [ ] Concurrent operations
- [ ] Error handling

### Data Quality
- [ ] Comparison accuracy
- [ ] Tolerance handling
- [ ] Data type compatibility
- [ ] Special character handling
- [ ] Null value processing

## 🚦 Success Criteria

### Functional Requirements
- ✅ 100% accuracy for exact change detection
- ✅ Configurable tolerance levels
- ✅ Primary key flexibility
- ✅ Comprehensive change reporting
- ✅ Multiple export formats

### Performance Requirements
- ✅ Process 50K records within 2 minutes
- ✅ Memory usage under 600MB for standard comparisons
- ✅ Support for multiple concurrent delta operations
- ✅ Real-time progress tracking

## 🔧 Advanced Testing Features

### Custom Comparison Rules
```json
{
  "custom_rules": [
    {
      "field": "amount",
      "comparison": "tolerance",
      "tolerance": 0.01
    },
    {
      "field": "status",
      "comparison": "exact"
    },
    {
      "field": "timestamp",
      "comparison": "ignore"
    }
  ]
}
```

### Batch Delta Processing
```bash
# Process multiple file pairs
python scripts/batch_delta.py \
  --input_dir docs/testing/delta/batch_input/ \
  --output_dir delta_results/ \
  --config delta_rules.json
```

### Historical Analysis
```bash
# Compare multiple versions over time
python scripts/historical_delta.py \
  --versions file_v1.csv,file_v2.csv,file_v3.csv \
  --timeline daily \
  --output change_timeline.json
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Overflow with Large Files**
   - Implement streaming comparison
   - Reduce chunk size
   - Use indexed comparison

2. **Slow Performance**
   - Optimize primary key indexing
   - Use parallel processing
   - Enable compression

3. **Incorrect Change Detection**
   - Verify primary key configuration
   - Check data type compatibility
   - Review tolerance settings

4. **Export Failures**
   - Check output format configuration
   - Verify file permissions
   - Monitor disk space

### Debug Mode
```bash
# Enable detailed delta logging
export DELTA_DEBUG=true
export LOG_LEVEL=DEBUG
python app/main.py
```

### Validation Commands
```bash
# Validate delta results
python scripts/validate_delta.py \
  --result delta_output.json \
  --expected expected_changes.json
```

## 📋 Quality Assurance

### Test Data Validation
```bash
# Verify test data integrity
python scripts/validate_test_data.py \
  --folder docs/testing/delta/ \
  --check_format --check_keys --check_changes
```

### Automated Testing
```bash
# Run comprehensive delta test suite
python -m pytest tests/delta/ -v --cov=app.routes.delta_routes
```

## 📞 Support and Resources

### Additional Resources
- API Documentation: `/docs/API_DOCUMENTATION.md`
- Performance Tuning: `/docs/performance/delta_optimization.md`
- Integration Guide: `/docs/examples/delta_integration.md`

### Contact Information
- Development Team: dev-team@company.com
- Technical Support: support@company.com
- Documentation Issues: docs@company.com

---

**Last Updated**: December 2024  
**Version**: 2.0.0  
**Maintainer**: FTT-ML Development Team