# Comprehensive Transformation Testing Guide

## Overview
This guide provides a complete framework for testing all transformation scenarios in the financial data processing platform. It includes test data, API calls, expected results, and validation procedures.

## Prerequisites

### 1. Start the Backend Server
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Verify Server Health
```bash
curl -X GET "http://localhost:8000/health"
```
Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-31T10:00:00Z"
}
```

## Test Data Files

The following test files are available in `/backend/test_data/`:
- `customers_test.csv` - 10 customer records with varied account types and balances
- `transactions_test.csv` - 12 transaction records including purchases and returns
- `products_test.csv` - 11 product records with pricing and specifications

### Customer Data Structure
- **High Balance Customers**: CUST004 ($25k), CUST010 ($22k), CUST007 ($18.5k), CUST001 ($15k)
- **Medium Balance**: CUST005 ($5k), CUST008 ($3.2k), CUST002 ($2.5k)
- **Low Balance**: CUST006 ($1.2k), CUST009 ($950), CUST003 ($750)
- **Account Types**: Premium (4), Standard (3), Basic (3)
- **Status**: Active (8), Suspended (1), Inactive (1)

---

# Test Scenarios

## 🎯 **Level 1: Basic Direct Mapping**

### Step 1: Upload Customer File
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_data/customers_test.csv" \
  -F "file_type=csv"
```

**Expected Response:**
```json
{
  "success": true,
  "file_id": "file_abc123",
  "filename": "customers_test.csv",
  "rows": 10,
  "columns": 15
}
```

### Step 2: Basic Direct Mapping Transformation
```bash
curl -X POST "http://localhost:8000/transformation/process" \
  -H "Content-Type: application/json" \
  -d '{
    "source_files": [
      {
        "file_id": "file_abc123",
        "alias": "customers",
        "purpose": "Main customer data source"
      }
    ],
    "transformation_config": {
      "name": "Basic Customer Mapping",
      "description": "Simple direct mapping of customer fields",
      "row_generation_rules": [
        {
          "id": "rule_001",
          "name": "Direct Customer Mapping",
          "enabled": true,
          "priority": 0,
          "condition": "",
          "output_columns": [
            {
              "id": "col_001",
              "name": "customer_id",
              "mapping_type": "direct",
              "source_column": "Customer_ID"
            },
            {
              "id": "col_002",
              "name": "full_name",
              "mapping_type": "static",
              "static_value": "{First_Name} {Last_Name}"
            },
            {
              "id": "col_003",
              "name": "email",
              "mapping_type": "direct",
              "source_column": "Email"
            },
            {
              "id": "col_004",
              "name": "balance",
              "mapping_type": "direct",
              "source_column": "Balance"
            }
          ]
        }
      ]
    }
  }'
```

**Expected Results:**
- Input Rows: 10 customers
- Output Rows: 10 records
- Output Columns: customer_id, full_name, email, balance
- Sample Output:
```
customer_id,full_name,email,balance
CUST001,John Smith,john.smith@email.com,15000.5
CUST002,Mary Johnson,mary.j@email.com,2500.75
...
```

---

## 🎯 **Level 2: Static Value Assignment & Expression Evaluation**

### Step 1: Customer Enrichment with Static Values
```bash
curl -X POST "http://localhost:8000/transformation/process" \
  -H "Content-Type: application/json" \
  -d '{
    "source_files": [
      {
        "file_id": "file_abc123",
        "alias": "customers",
        "purpose": "Customer data with static enrichment"
      }
    ],
    "transformation_config": {
      "name": "Customer Enrichment with Static Values",
      "description": "Add static computed fields to customer data",
      "row_generation_rules": [
        {
          "id": "rule_002",
          "name": "Customer Enrichment",
          "enabled": true,
          "priority": 0,
          "condition": "",
          "output_columns": [
            {
              "id": "col_001",
              "name": "customer_id",
              "mapping_type": "direct",
              "source_column": "Customer_ID"
            },
            {
              "id": "col_002",
              "name": "full_name",
              "mapping_type": "static",
              "static_value": "{First_Name} {Last_Name}"
            },
            {
              "id": "col_003",
              "name": "data_source",
              "mapping_type": "static",
              "static_value": "Customer Master File"
            },
            {
              "id": "col_004",
              "name": "processing_date",
              "mapping_type": "static",
              "static_value": "2024-01-31"
            },
            {
              "id": "col_005",
              "name": "account_summary",
              "mapping_type": "static",
              "static_value": "{Account_Type} account with balance ${Balance}"
            }
          ]
        }
      ]
    }
  }'
```

**Expected Results:**
- Input Rows: 10 customers
- Output Rows: 10 records
- Sample account_summary values:
  - "Premium account with balance $15000.5"
  - "Standard account with balance $2500.75"
  - "Basic account with balance $750.0"

**Validation Points:**
1. ✅ All rows should have data_source = "Customer Master File"
2. ✅ All rows should have processing_date = "2024-01-31"
3. ✅ account_summary should NOT be empty
4. ✅ account_summary should combine Account_Type and Balance correctly

---

## 🎯 **Level 3: Dynamic Conditional Logic**

### Step 1: Customer Tier Classification
```bash
curl -X POST "http://localhost:8000/transformation/process" \
  -H "Content-Type: application/json" \
  -d '{
    "source_files": [
      {
        "file_id": "file_abc123",
        "alias": "customers",
        "purpose": "Customer classification"
      }
    ],
    "transformation_config": {
      "name": "Customer Tier Classification",
      "description": "Classify customers based on balance and account type",
      "row_generation_rules": [
        {
          "id": "rule_003",
          "name": "Customer Classification",
          "enabled": true,
          "priority": 0,
          "condition": "",
          "output_columns": [
            {
              "id": "col_001",
              "name": "customer_id",
              "mapping_type": "direct",
              "source_column": "Customer_ID"
            },
            {
              "id": "col_002",
              "name": "customer_tier",
              "mapping_type": "dynamic",
              "dynamic_conditions": [
                {
                  "id": "cond_001",
                  "condition_column": "Balance",
                  "operator": ">=",
                  "condition_value": "20000",
                  "output_value": "VIP"
                },
                {
                  "id": "cond_002",
                  "condition_column": "Balance",
                  "operator": ">=",
                  "condition_value": "10000",
                  "output_value": "Premium"
                },
                {
                  "id": "cond_003",
                  "condition_column": "Balance",
                  "operator": ">=",
                  "condition_value": "1000",
                  "output_value": "Standard"
                }
              ],
              "default_value": "Basic"
            },
            {
              "id": "col_003",
              "name": "status_description",
              "mapping_type": "dynamic",
              "dynamic_conditions": [
                {
                  "id": "cond_004",
                  "condition_column": "Status",
                  "operator": "==",
                  "condition_value": "Active",
                  "output_value": "Account is active and in good standing"
                },
                {
                  "id": "cond_005",
                  "condition_column": "Status",
                  "operator": "==",
                  "condition_value": "Suspended",
                  "output_value": "Account is temporarily suspended"
                }
              ],
              "default_value": "Account status requires review"
            }
          ]
        }
      ]
    }
  }'
```

**Expected Results:**
- **VIP Customers**: CUST004 ($25k), CUST010 ($22k)
- **Premium Customers**: CUST001 ($15k), CUST007 ($18.5k)
- **Standard Customers**: CUST005 ($5k), CUST008 ($3.2k), CUST002 ($2.5k), CUST006 ($1.2k)
- **Basic Customers**: CUST009 ($950), CUST003 ($750)

---

## 🎯 **Level 4: Multiple Rules with Conditions**

### Step 1: Upload All Files
```bash
# Upload customers
curl -X POST "http://localhost:8000/upload" \
  -F "file=@test_data/customers_test.csv" \
  -F "file_type=csv"

# Upload transactions  
curl -X POST "http://localhost:8000/upload" \
  -F "file=@test_data/transactions_test.csv" \
  -F "file_type=csv"

# Upload products
curl -X POST "http://localhost:8000/upload" \
  -F "file=@test_data/products_test.csv" \
  -F "file_type=csv"
```

### Step 2: Multi-File Join Transformation
```bash
curl -X POST "http://localhost:8000/transformation/process" \
  -H "Content-Type: application/json" \
  -d '{
    "source_files": [
      {
        "file_id": "customers_file_id",
        "alias": "customers",
        "purpose": "Customer master data"
      },
      {
        "file_id": "transactions_file_id",
        "alias": "transactions",
        "purpose": "Transaction history"
      },
      {
        "file_id": "products_file_id",
        "alias": "products",
        "purpose": "Product catalog"
      }
    ],
    "transformation_config": {
      "name": "Complete Sales Analysis",
      "description": "Comprehensive transformation using all source files",
      "row_generation_rules": [
        {
          "id": "rule_008",
          "name": "Sales Analysis Report",
          "enabled": true,
          "priority": 0,
          "condition": "",
          "output_columns": [
            {
              "id": "col_001",
              "name": "transaction_id",
              "mapping_type": "direct",
              "source_column": "Transaction_ID"
            },
            {
              "id": "col_002",
              "name": "customer_name",
              "mapping_type": "static",
              "static_value": "{First_Name} {Last_Name}"
            },
            {
              "id": "col_003",
              "name": "customer_tier",
              "mapping_type": "dynamic",
              "dynamic_conditions": [
                {
                  "id": "cond_001",
                  "condition_column": "Balance",
                  "operator": ">=",
                  "condition_value": "15000",
                  "output_value": "VIP"
                },
                {
                  "id": "cond_002",
                  "condition_column": "Balance",
                  "operator": ">=",
                  "condition_value": "5000",
                  "output_value": "Premium"
                }
              ],
              "default_value": "Standard"
            },
            {
              "id": "col_004",
              "name": "product_name",
              "mapping_type": "direct",
              "source_column": "Product_Name"
            },
            {
              "id": "col_005",
              "name": "category",
              "mapping_type": "direct",
              "source_column": "Category"
            },
            {
              "id": "col_006",
              "name": "sale_amount",
              "mapping_type": "direct",
              "source_column": "Amount"
            },
            {
              "id": "col_007",
              "name": "transaction_summary",
              "mapping_type": "static",
              "static_value": "{Customer_ID} purchased {Product_Name} for ${Amount}"
            }
          ]
        }
      ]
    }
  }'
```

---

## 🎯 **Level 5: Advanced Expression & Mathematical Calculations**

### Step 1: Advanced Customer Metrics
```bash
curl -X POST "http://localhost:8000/transformation/process" \
  -H "Content-Type: application/json" \
  -d '{
    "source_files": [
      {
        "file_id": "customers_file_id",
        "alias": "customers",
        "purpose": "Customer data for advanced metrics"
      }
    ],
    "transformation_config": {
      "name": "Advanced Customer Metrics",
      "description": "Complex calculations and advanced expressions",
      "row_generation_rules": [
        {
          "id": "rule_009",
          "name": "Advanced Metrics Calculation",
          "enabled": true,
          "priority": 0,
          "condition": "",
          "output_columns": [
            {
              "id": "col_001",
              "name": "customer_id",
              "mapping_type": "direct",
              "source_column": "Customer_ID"
            },
            {
              "id": "col_002",
              "name": "balance_category",
              "mapping_type": "dynamic",
              "dynamic_conditions": [
                {
                  "id": "cond_001",
                  "condition_column": "Balance",
                  "operator": ">=",
                  "condition_value": "20000",
                  "output_value": "High Balance (≥$20k)"
                },
                {
                  "id": "cond_002",
                  "condition_column": "Balance",
                  "operator": ">=",
                  "condition_value": "5000",
                  "output_value": "Medium Balance ($5k-$20k)"
                },
                {
                  "id": "cond_003",
                  "condition_column": "Balance",
                  "operator": ">=",
                  "condition_value": "1000",
                  "output_value": "Low Balance ($1k-$5k)"
                }
              ],
              "default_value": "Very Low Balance (<$1k)"
            },
            {
              "id": "col_003",
              "name": "account_profile",
              "mapping_type": "static",
              "static_value": "{First_Name} {Last_Name} - {Account_Type} ({Status})"
            },
            {
              "id": "col_004",
              "name": "balance_formatted",
              "mapping_type": "static",
              "static_value": "Balance: ${Balance}"
            },
            {
              "id": "col_005",
              "name": "risk_assessment",
              "mapping_type": "dynamic",
              "dynamic_conditions": [
                {
                  "id": "cond_004",
                  "condition_column": "Status",
                  "operator": "==",
                  "condition_value": "Suspended",
                  "output_value": "High Risk - Account Suspended"
                },
                {
                  "id": "cond_005",
                  "condition_column": "Status",
                  "operator": "==",
                  "condition_value": "Inactive",
                  "output_value": "Medium Risk - Account Inactive"
                }
              ],
              "default_value": "Low Risk - Account Active"
            }
          ]
        }
      ]
    }
  }'
```

---

# Validation & Testing Scripts

## 🔍 **Test Result Validation Script**

Create this validation script to verify your transformation results:

```python
#!/usr/bin/env python3
"""
Validation script for transformation testing results
"""

import requests
import json
import csv
from typing import Dict, List, Any

class TransformationValidator:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def validate_basic_mapping(self, result_data: List[Dict]) -> Dict[str, Any]:
        """Validate basic direct mapping results"""
        validation = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "summary": {}
        }
        
        if not result_data:
            validation["passed"] = False
            validation["errors"].append("No result data found")
            return validation
            
        # Check required columns
        required_columns = ["customer_id", "full_name", "email", "balance"]
        first_row = result_data[0]
        
        for col in required_columns:
            if col not in first_row:
                validation["passed"] = False
                validation["errors"].append(f"Missing required column: {col}")
        
        # Validate data types and content
        for i, row in enumerate(result_data):
            # Check customer_id format
            if not row.get("customer_id", "").startswith("CUST"):
                validation["warnings"].append(f"Row {i}: customer_id should start with CUST")
            
            # Check full_name has space (combination of first and last name)
            if " " not in row.get("full_name", ""):
                validation["errors"].append(f"Row {i}: full_name should contain space")
                validation["passed"] = False
            
            # Check email format
            if "@" not in row.get("email", ""):
                validation["warnings"].append(f"Row {i}: email format may be invalid")
            
            # Check balance is numeric
            try:
                float(row.get("balance", 0))
            except (ValueError, TypeError):
                validation["errors"].append(f"Row {i}: balance should be numeric")
                validation["passed"] = False
        
        validation["summary"] = {
            "total_rows": len(result_data),
            "columns_found": list(first_row.keys()) if result_data else [],
            "sample_data": result_data[:3] if result_data else []
        }
        
        return validation
    
    def validate_static_expressions(self, result_data: List[Dict]) -> Dict[str, Any]:
        """Validate static value and expression results"""
        validation = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "summary": {}
        }
        
        if not result_data:
            validation["passed"] = False
            validation["errors"].append("No result data found")
            return validation
        
        account_summary_empty_count = 0
        account_summary_samples = []
        
        for i, row in enumerate(result_data):
            # Check data_source
            if row.get("data_source") != "Customer Master File":
                validation["errors"].append(f"Row {i}: data_source should be 'Customer Master File'")
                validation["passed"] = False
            
            # Check processing_date
            if row.get("processing_date") != "2024-01-31":
                validation["errors"].append(f"Row {i}: processing_date should be '2024-01-31'")
                validation["passed"] = False
            
            # Check account_summary - THIS IS THE KEY TEST
            account_summary = row.get("account_summary", "")
            if not account_summary or account_summary.strip() == "":
                account_summary_empty_count += 1
                validation["errors"].append(f"Row {i}: account_summary is empty")
                validation["passed"] = False
            else:
                account_summary_samples.append(account_summary)
                # Validate format: should contain "account with balance $"
                if "account with balance $" not in account_summary:
                    validation["warnings"].append(f"Row {i}: account_summary format may be incorrect")
        
        validation["summary"] = {
            "total_rows": len(result_data),
            "account_summary_empty_count": account_summary_empty_count,
            "account_summary_samples": account_summary_samples[:5],
            "data_source_check": "Passed" if all(r.get("data_source") == "Customer Master File" for r in result_data) else "Failed",
            "processing_date_check": "Passed" if all(r.get("processing_date") == "2024-01-31" for r in result_data) else "Failed"
        }
        
        return validation
    
    def validate_dynamic_conditions(self, result_data: List[Dict]) -> Dict[str, Any]:
        """Validate dynamic conditional logic results"""
        validation = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "summary": {}
        }
        
        tier_counts = {"VIP": 0, "Premium": 0, "Standard": 0, "Basic": 0}
        
        for i, row in enumerate(result_data):
            tier = row.get("customer_tier", "")
            if tier in tier_counts:
                tier_counts[tier] += 1
            else:
                validation["warnings"].append(f"Row {i}: unexpected customer_tier value: {tier}")
            
            # Validate status_description
            status_desc = row.get("status_description", "")
            if not status_desc:
                validation["errors"].append(f"Row {i}: status_description is empty")
                validation["passed"] = False
        
        validation["summary"] = {
            "total_rows": len(result_data),
            "tier_distribution": tier_counts,
            "expected_vip": "Should be 2 (CUST004, CUST010)",
            "expected_premium": "Should be 2 (CUST001, CUST007)"
        }
        
        return validation

def run_comprehensive_test():
    """Run comprehensive transformation testing"""
    validator = TransformationValidator()
    
    print("🧪 Starting Comprehensive Transformation Testing")
    print("=" * 60)
    
    # Test results would be passed here from your API calls
    # This is a template for how to structure your validation
    
    print("✅ Test framework ready!")
    print("📝 Use the API calls above to get transformation results")
    print("🔍 Then pass the results to these validation functions")

if __name__ == "__main__":
    run_comprehensive_test()
```

---

# Common Issues & Troubleshooting

## 🚨 **Issue: account_summary Coming as Empty**

### Debugging Steps:

1. **Check File Upload:**
```bash
curl -X GET "http://localhost:8000/files/{file_id}/preview"
```

2. **Verify Column Names:**
Ensure the CSV has exact column names: `Account_Type`, `Balance`

3. **Test Expression Evaluation:**
```bash
curl -X POST "http://localhost:8000/transformation/preview" \
  -H "Content-Type: application/json" \
  -d '{
    "source_files": [...],
    "transformation_config": {...},
    "row_limit": 3
  }'
```

4. **Check Server Logs:**
Look for error messages in the server console output.

### Expected vs Actual Debug:

**Expected account_summary values:**
- `"Premium account with balance $15000.5"`
- `"Standard account with balance $2500.75"`
- `"Basic account with balance $750.0"`

**If getting empty values, check:**
- Column name casing (Account_Type vs account_type)
- Expression syntax ({Account_Type} not {account_type})
- Data type handling (Balance as number vs string)

---

# Quick Test Commands

## 🚀 **One-Click Test Sequence**

```bash
#!/bin/bash
# Quick test sequence

echo "🏁 Starting Quick Test Sequence"

# 1. Health Check  
echo "1️⃣ Health Check"
curl -s "http://localhost:8000/health" | jq

# 2. Upload File
echo "2️⃣ Uploading Customer File"
UPLOAD_RESPONSE=$(curl -s -X POST "http://localhost:8000/upload" \
  -F "file=@test_data/customers_test.csv" \
  -F "file_type=csv")
echo $UPLOAD_RESPONSE | jq

# Extract file_id (you'll need to parse this)
FILE_ID=$(echo $UPLOAD_RESPONSE | jq -r '.file_id')
echo "📁 File ID: $FILE_ID"

# 3. Test Basic Transformation
echo "3️⃣ Testing Basic Transformation"
curl -s -X POST "http://localhost:8000/transformation/process" \
  -H "Content-Type: application/json" \
  -d "{
    \"source_files\": [{
      \"file_id\": \"$FILE_ID\",
      \"alias\": \"customers\",
      \"purpose\": \"Test data\"
    }],
    \"transformation_config\": {
      \"row_generation_rules\": [{
        \"id\": \"rule_001\",
        \"name\": \"Test Rule\",
        \"enabled\": true,
        \"priority\": 0,
        \"condition\": \"\",
        \"output_columns\": [{
          \"id\": \"col_001\",
          \"name\": \"account_summary\",
          \"mapping_type\": \"static\",
          \"static_value\": \"{Account_Type} account with balance \\${Balance}\"
        }]
      }]
    }
  }" | jq

echo "✅ Quick test completed!"
```

This comprehensive guide provides everything needed to test all transformation scenarios systematically. Start with Level 1 and progress through each level to verify all functionality works correctly.