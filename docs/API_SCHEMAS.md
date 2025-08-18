# API Request/Response Schemas

## 📋 Overview

This document provides detailed schema documentation for all API endpoints in the FTT-ML platform. All schemas are automatically validated using Pydantic models and are available in the interactive documentation at `/docs`.

## 🔧 Common Data Types

### Basic Types
- `string`: Text data
- `integer`: Whole numbers
- `number`: Decimal numbers
- `boolean`: true/false values
- `array`: List of items
- `object`: JSON object with key-value pairs

### Custom Enums

#### MappingType
```json
{
  "type": "string",
  "enum": ["direct", "static", "dynamic"],
  "description": "Type of column mapping"
}
```

#### ConditionOperator
```json
{
  "type": "string", 
  "enum": ["==", "!=", ">", "<", ">=", "<=", "contains", "startsWith", "endsWith"],
  "description": "Comparison operator for conditions"
}
```

## 📁 File Management Schemas

### File Upload Response
```json
{
  "type": "object",
  "properties": {
    "success": {"type": "boolean"},
    "message": {"type": "string"},
    "data": {
      "type": "object", 
      "properties": {
        "file_id": {"type": "string", "example": "file_abc123"},
        "filename": {"type": "string", "example": "customer_data.csv"},
        "file_size_mb": {"type": "number", "example": 2.5},
        "total_rows": {"type": "integer", "example": 1000},
        "columns": {
          "type": "array",
          "items": {"type": "string"},
          "example": ["customer_id", "first_name", "last_name", "email"]
        },
        "file_type": {"type": "string", "example": "csv"}
      }
    }
  }
}
```

### File Preview Response
```json
{
  "type": "object",
  "properties": {
    "success": {"type": "boolean"},
    "data": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": true
      },
      "example": [
        {
          "customer_id": "CUST001",
          "first_name": "John", 
          "last_name": "Doe",
          "email": "john.doe@email.com"
        }
      ]
    }
  }
}
```

## 🔄 Transformation Schemas

### SourceFile Schema
```json
{
  "type": "object",
  "required": ["file_id", "alias"],
  "properties": {
    "file_id": {
      "type": "string",
      "description": "Unique identifier for the uploaded file",
      "example": "file_abc123"
    },
    "alias": {
      "type": "string", 
      "description": "Alias name for referencing this file",
      "example": "customers"
    },
    "purpose": {
      "type": "string",
      "description": "Description of the file's purpose",
      "example": "Primary customer data source"
    }
  }
}
```

### DynamicCondition Schema
```json
{
  "type": "object",
  "required": ["id", "condition_column", "condition_value", "output_value"],
  "properties": {
    "id": {
      "type": "string",
      "description": "Unique identifier for this condition",
      "example": "cond_001"
    },
    "condition_column": {
      "type": "string",
      "description": "Source column to evaluate", 
      "example": "amount"
    },
    "operator": {
      "$ref": "#/components/schemas/ConditionOperator",
      "default": "==",
      "example": ">="
    },
    "condition_value": {
      "type": "string",
      "description": "Value to compare against",
      "example": "1000"
    },
    "output_value": {
      "type": "string", 
      "description": "Value to output when condition is met (supports expressions)",
      "example": "Premium"
    }
  }
}
```

### RuleOutputColumn Schema
```json
{
  "type": "object",
  "required": ["id", "name"],
  "properties": {
    "id": {
      "type": "string",
      "description": "Unique identifier for this column mapping",
      "example": "col_001"
    },
    "name": {
      "type": "string",
      "description": "Output column name",
      "example": "customer_tier"
    },
    "mapping_type": {
      "$ref": "#/components/schemas/MappingType",
      "default": "direct"
    },
    "source_column": {
      "type": "string",
      "description": "Source column for direct mapping",
      "example": "customer_id"
    },
    "static_value": {
      "type": "string",
      "description": "Static value or expression for all rows",
      "example": "{first_name} {last_name}"
    },
    "dynamic_conditions": {
      "type": "array",
      "items": {"$ref": "#/components/schemas/DynamicCondition"},
      "description": "List of conditions for dynamic mapping"
    },
    "default_value": {
      "type": "string",
      "description": "Default value when no dynamic conditions match",
      "example": "Standard"
    }
  }
}
```

### TransformationRule Schema
```json
{
  "type": "object",
  "required": ["id", "name"],
  "properties": {
    "id": {
      "type": "string",
      "description": "Unique identifier for this rule",
      "example": "rule_001"
    },
    "name": {
      "type": "string",
      "description": "Rule display name",
      "example": "Customer Summary Rule"
    },
    "enabled": {
      "type": "boolean",
      "default": true,
      "description": "Whether this rule is active"
    },
    "priority": {
      "type": "integer",
      "default": 0,
      "description": "Execution priority (lower = higher priority)"
    },
    "condition": {
      "type": "string",
      "description": "Condition expression for when this rule applies",
      "example": "amount > 0"
    },
    "output_columns": {
      "type": "array",
      "items": {"$ref": "#/components/schemas/RuleOutputColumn"},
      "description": "List of output columns this rule generates"
    }
  }
}
```

### TransformationRequest Schema
```json
{
  "type": "object",
  "required": ["process_name", "source_files", "transformation_config"],
  "properties": {
    "process_name": {
      "type": "string",
      "description": "Name of the transformation process", 
      "example": "Customer Data Standardization"
    },
    "description": {
      "type": "string",
      "description": "Description of what this transformation does",
      "example": "Standardize customer data and calculate totals"
    },
    "source_files": {
      "type": "array",
      "items": {"$ref": "#/components/schemas/SourceFile"},
      "description": "List of source files to transform"
    },
    "transformation_config": {
      "type": "object",
      "description": "Transformation configuration with rules and mappings",
      "properties": {
        "name": {"type": "string", "example": "Customer Summary Transformation"},
        "description": {"type": "string"},
        "source_files": {
          "type": "array",
          "items": {"$ref": "#/components/schemas/SourceFile"}
        },
        "row_generation_rules": {
          "type": "array", 
          "items": {"$ref": "#/components/schemas/TransformationRule"}
        },
        "merge_datasets": {
          "type": "boolean",
          "default": true,
          "description": "Merge all rule outputs into single dataset"
        }
      }
    },
    "preview_only": {
      "type": "boolean",
      "default": false,
      "description": "If true, only process a small sample for preview"
    },
    "row_limit": {
      "type": "integer",
      "description": "Maximum number of rows to process (for preview mode)",
      "example": 10
    }
  }
}
```

### TransformationResult Schema
```json
{
  "type": "object",
  "required": ["success", "transformation_id", "total_input_rows", "total_output_rows", "processing_time_seconds", "validation_summary"],
  "properties": {
    "success": {
      "type": "boolean",
      "description": "Whether the transformation completed successfully"
    },
    "transformation_id": {
      "type": "string",
      "description": "Unique identifier for this transformation",
      "example": "transform_abc123"
    },
    "total_input_rows": {
      "type": "integer", 
      "description": "Total number of input rows processed",
      "example": 1000
    },
    "total_output_rows": {
      "type": "integer",
      "description": "Total number of output rows generated", 
      "example": 1000
    },
    "processing_time_seconds": {
      "type": "number",
      "description": "Time taken to process the transformation",
      "example": 2.456
    },
    "validation_summary": {
      "type": "object",
      "description": "Summary of validation results and processing statistics",
      "properties": {
        "input_row_count": {"type": "integer"},
        "output_row_count": {"type": "integer"},
        "processing_time": {"type": "number"},
        "rules_processed": {"type": "integer"},
        "datasets_generated": {"type": "integer"}
      }
    },
    "warnings": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Non-critical warnings encountered during processing"
    },
    "errors": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Critical errors encountered during processing"
    },
    "preview_data": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": true
      },
      "description": "Sample of output data (only included in preview mode)"
    }
  }
}
```

## 🤖 AI Configuration Generation Schemas

### AI Configuration Request
```json
{
  "type": "object",
  "required": ["requirements", "source_files"],
  "properties": {
    "requirements": {
      "type": "string",
      "description": "Natural language description of desired transformation",
      "example": "Transform customer data to include full name and calculate total with tax"
    },
    "source_files": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "file_id": {"type": "string", "example": "file_abc123"},
          "filename": {"type": "string", "example": "customers.csv"},
          "columns": {
            "type": "array",
            "items": {"type": "string"},
            "example": ["customer_id", "first_name", "last_name", "amount", "tax_rate"]
          },
          "totalRows": {"type": "integer", "example": 1000}
        }
      }
    }
  }
}
```

### AI Configuration Response
```json
{
  "type": "object",
  "properties": {
    "success": {"type": "boolean"},
    "message": {"type": "string"},
    "data": {
      "type": "object",
      "description": "Generated transformation configuration",
      "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "source_files": {
          "type": "array",
          "items": {"$ref": "#/components/schemas/SourceFile"}
        },
        "row_generation_rules": {
          "type": "array",
          "items": {"$ref": "#/components/schemas/TransformationRule"}
        }
      }
    }
  }
}
```

## 🔄 Reconciliation Schemas

### ReconciliationRequest Schema
```json
{
  "type": "object",
  "required": ["file_a_id", "file_b_id", "reconciliation_config"],
  "properties": {
    "file_a_id": {
      "type": "string",
      "description": "ID of the first file to reconcile",
      "example": "file_abc123"
    },
    "file_b_id": {
      "type": "string",
      "description": "ID of the second file to reconcile", 
      "example": "file_def456"
    },
    "reconciliation_config": {
      "type": "object",
      "properties": {
        "name": {"type": "string", "example": "Bank vs Internal Reconciliation"},
        "description": {"type": "string"},
        "matching_criteria": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "field_a": {"type": "string", "example": "transaction_id"},
              "field_b": {"type": "string", "example": "reference_id"},
              "match_type": {
                "type": "string",
                "enum": ["exact", "tolerance", "fuzzy", "date"],
                "example": "exact"
              },
              "tolerance": {"type": "number", "example": 0.01},
              "weight": {"type": "number", "example": 0.4}
            }
          }
        },
        "match_threshold": {
          "type": "number",
          "description": "Minimum confidence score for automatic matching",
          "example": 0.8
        },
        "auto_match": {
          "type": "boolean",
          "description": "Whether to automatically match records",
          "default": true
        }
      }
    }
  }
}
```

### ReconciliationResult Schema
```json
{
  "type": "object",
  "properties": {
    "success": {"type": "boolean"},
    "reconciliation_id": {"type": "string", "example": "recon_xyz789"},
    "total_records_a": {"type": "integer", "example": 1500},
    "total_records_b": {"type": "integer", "example": 1480},
    "matched_pairs": {"type": "integer", "example": 1420},
    "unmatched_a": {"type": "integer", "example": 80},
    "unmatched_b": {"type": "integer", "example": 60},
    "match_rate": {"type": "number", "example": 94.67},
    "processing_time_seconds": {"type": "number", "example": 3.456},
    "summary": {
      "type": "object",
      "properties": {
        "perfect_matches": {"type": "integer", "example": 1380},
        "partial_matches": {"type": "integer", "example": 40},
        "potential_duplicates": {"type": "integer", "example": 5}
      }
    }
  }
}
```

## 📊 Delta Generation Schemas

### DeltaRequest Schema
```json
{
  "type": "object", 
  "required": ["file_a_id", "file_b_id", "delta_config"],
  "properties": {
    "file_a_id": {
      "type": "string",
      "description": "ID of the older file",
      "example": "file_old_123"
    },
    "file_b_id": {
      "type": "string", 
      "description": "ID of the newer file",
      "example": "file_new_456"
    },
    "delta_config": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "key_columns": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Columns to use as unique identifiers",
          "example": ["transaction_id", "account"]
        },
        "compare_columns": {
          "type": "array", 
          "items": {"type": "string"},
          "description": "Columns to compare for changes",
          "example": ["amount", "status", "description"]
        },
        "ignore_columns": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Columns to ignore in comparison"
        }
      }
    }
  }
}
```

## ❌ Error Response Schema

### Standard Error Response
```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean",
      "example": false
    },
    "message": {
      "type": "string",
      "description": "Human-readable error message",
      "example": "Validation error occurred"
    },
    "detail": {
      "type": "string",
      "description": "Detailed error information",
      "example": "Field 'customer_id' is required but missing from source data"
    },
    "error_code": {
      "type": "string",
      "description": "Machine-readable error code",
      "example": "VALIDATION_ERROR"
    },
    "errors": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "field": {"type": "string"},
          "message": {"type": "string"},
          "code": {"type": "string"}
        }
      },
      "description": "Detailed field-level errors"
    }
  }
}
```

### HTTP Status Codes

| Code | Description | Usage |
|------|-------------|-------|
| 200 | OK | Successful operation |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request format or parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 413 | Payload Too Large | File size exceeds limits |
| 422 | Unprocessable Entity | Validation errors |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | Service temporarily unavailable |

## 🔍 Validation Rules

### Common Validation Patterns

1. **File IDs**: Must start with "file_" followed by alphanumeric characters
2. **Email**: Valid email format required
3. **Dates**: ISO 8601 format (YYYY-MM-DD) or auto-detected formats
4. **Numbers**: Must be valid numeric values, decimals allowed
5. **Column Names**: Must exist in source data, case-sensitive
6. **Expressions**: Must use valid `{column_name}` syntax

### Expression Validation

#### Valid Expression Examples
```json
{
  "mathematical": "{quantity} * {unit_price}",
  "string_concatenation": "{first_name} {last_name}",
  "complex_calculation": "{amount} * (1 + {tax_rate}/100)",
  "conditional": "{status} == 'active' ? {amount} : 0"
}
```

#### Invalid Expression Examples
```json
{
  "missing_braces": "quantity * unit_price",
  "nonexistent_column": "{total_sales}",
  "invalid_syntax": "{amount} * (1 + {tax_rate/100}",
  "unsafe_function": "exec('malicious_code')"
}
```

## 📚 Schema Validation Tools

### Using Python Requests
```python
import requests
from pydantic import ValidationError

# Validate request data before sending
try:
    request_data = {
        "process_name": "Customer Transformation",
        "source_files": [{"file_id": "file_123", "alias": "customers"}],
        "transformation_config": {...}
    }
    
    response = requests.post("http://localhost:8000/transformation/process/", json=request_data)
    response.raise_for_status()
    
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
    print(f"Response: {e.response.json()}")
```

### Using cURL for Schema Testing
```bash
# Test transformation request
curl -X POST "http://localhost:8000/transformation/process/" \
  -H "Content-Type: application/json" \
  -d '{
    "process_name": "Test Transformation",
    "source_files": [{"file_id": "file_123", "alias": "test"}],
    "transformation_config": {...},
    "preview_only": true
  }'
```

This comprehensive schema documentation ensures that developers can easily understand and integrate with the FTT-ML API while maintaining data consistency and validation across all endpoints.