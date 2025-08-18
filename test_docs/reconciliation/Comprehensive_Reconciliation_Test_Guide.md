# Comprehensive Reconciliation Testing Guide

## Overview
This guide provides incremental test scenarios to validate all reconciliation features using the provided test files:
- `comprehensive_test_file_a.csv` (25 records)
- `comprehensive_test_file_b.csv` (29 records)

## Test Files Structure

### File A - Transaction Data
- **Records**: 25 transactions (TXN001-TXN025)
- **Key Features**: Mixed case status values, duplicate references, various priorities
- **Special Cases**: Negative amounts, case variations, duplicate entry

### File B - Statement Data  
- **Records**: 29 statements (STMT001-STMT029)
- **Key Features**: Different date format, uppercase status values, unmatched records
- **Special Cases**: Date format variations, tolerance test amounts

---

## Test Scenarios (Incremental)

### 🎯 **Level 1: Basic Reference Matching**
**Goal**: Test simple exact matching without filters

#### Configuration:
```json
{
  "Files": [
    {
      "Name": "FileA",
      "Extract": [],
      "Filter": []
    },
    {
      "Name": "FileB",
      "Extract": [],
      "Filter": []
    }
  ],
  "ReconciliationRules": [
    {
      "LeftFileColumn": "Reference",
      "RightFileColumn": "Ref_Number",
      "MatchType": "equals",
      "ToleranceValue": 0
    }
  ]
}
```

#### AI Prompt:
```
"Please create a basic reconciliation configuration that matches transactions by reference number only. Use exact matching between the Reference field in File A and Ref_Number field in File B. No filters or extractions needed."
```

#### Expected Results:
- **Matched**: 20 records (REF123-REF142)
- **Unmatched File A**: 5 records (REF143-REF146, plus duplicates)
- **Unmatched File B**: 9 records (including REF999, REF888, etc.)
- **Match Rate**: 80% (20/25)

---

### 🎯 **Level 2: Case Insensitive Status Filtering**
**Goal**: Test filter functionality with case insensitive matching

#### Configuration:
```json
{
  "Files": [
    {
      "Name": "FileA", 
      "Extract": [],
      "Filter": [
        {
          "ColumnName": "Status",
          "MatchType": "equals",
          "Value": "Settled"
        }
      ]
    },
    {
      "Name": "FileB",
      "Extract": [],
      "Filter": [
        {
          "ColumnName": "Settlement_Status", 
          "MatchType": "equals",
          "Value": "SETTLED"
        }
      ]
    }
  ],
  "ReconciliationRules": [
    {
      "LeftFileColumn": "Reference",
      "RightFileColumn": "Ref_Number",
      "MatchType": "equals",
      "ToleranceValue": 0
    }
  ]
}
```

#### AI Prompt:
```
"Create a reconciliation configuration that only processes 'Settled' transactions. Filter File A for Status='Settled' and File B for Settlement_Status='SETTLED'. Then match by reference number."
```

#### Expected Results:
- **File A After Filter**: ~12 records (Settled status)
- **File B After Filter**: ~12 records (SETTLED status)  
- **Matched**: ~10 records
- **Tests**: Case insensitive filtering

---

### 🎯 **Level 3: Multiple Matching Rules**
**Goal**: Test multi-column exact matching

#### Configuration:
```json
{
  "Files": [
    {
      "Name": "FileA",
      "Extract": [],
      "Filter": []
    },
    {
      "Name": "FileB",
      "Extract": [],
      "Filter": []
    }
  ],
  "ReconciliationRules": [
    {
      "LeftFileColumn": "Reference",
      "RightFileColumn": "Ref_Number", 
      "MatchType": "equals",
      "ToleranceValue": 0
    },
    {
      "LeftFileColumn": "Account",
      "RightFileColumn": "Account_Number",
      "MatchType": "equals", 
      "ToleranceValue": 0
    },
    {
      "LeftFileColumn": "Customer_ID",
      "RightFileColumn": "Client_Code",
      "MatchType": "equals",
      "ToleranceValue": 0
    }
  ]
}
```

#### AI Prompt:
```
"Create a comprehensive reconciliation that matches on three fields: Reference/Ref_Number, Account/Account_Number, and Customer_ID/Client_Code. All must match exactly for a successful reconciliation."
```

#### Expected Results:
- **Matched**: ~20 records (all rules must pass)
- **Tests**: Multi-rule validation, AND logic

---

### 🎯 **Level 4: Amount Tolerance Matching**
**Goal**: Test numeric tolerance matching

#### Configuration:
```json
{
  "Files": [
    {
      "Name": "FileA",
      "Extract": [],
      "Filter": []
    },
    {
      "Name": "FileB",
      "Extract": [],
      "Filter": []
    }
  ],
  "ReconciliationRules": [
    {
      "LeftFileColumn": "Reference",
      "RightFileColumn": "Ref_Number",
      "MatchType": "equals",
      "ToleranceValue": 0
    },
    {
      "LeftFileColumn": "Amount",
      "RightFileColumn": "Net_Amount", 
      "MatchType": "tolerance",
      "ToleranceValue": 0.01
    }
  ]
}
```

#### AI Prompt:
```
"Create a reconciliation that matches by reference number exactly AND amount within 0.01 tolerance. This should catch small rounding differences in financial amounts."
```

#### Expected Results:
- **Matched**: ~19 records (exact matches)
- **Tolerance Matches**: 1 record (TXN025/STMT026: 500.01 vs 500.00)
- **Tests**: Numeric tolerance logic

---

### 🎯 **Level 5: Text Extraction**
**Goal**: Test pattern extraction from text fields

#### Configuration:
```json
{
  "Files": [
    {
      "Name": "FileA",
      "Extract": [
        {
          "ResultColumnName": "Extracted_Amount",
          "SourceColumn": "Amount_Text",
          "MatchType": "regex",
          "Patterns": ["[0-9]+\\.[0-9]{2}"]
        }
      ],
      "Filter": []
    },
    {
      "Name": "FileB",
      "Extract": [
        {
          "ResultColumnName": "Extracted_Amount",
          "SourceColumn": "Net_Amount_Text", 
          "MatchType": "regex",
          "Patterns": ["[0-9]+\\.[0-9]{2}"]
        }
      ],
      "Filter": []
    }
  ],
  "ReconciliationRules": [
    {
      "LeftFileColumn": "Reference",
      "RightFileColumn": "Ref_Number",
      "MatchType": "equals",
      "ToleranceValue": 0
    },
    {
      "LeftFileColumn": "Extracted_Amount",
      "RightFileColumn": "Extracted_Amount",
      "MatchType": "equals",
      "ToleranceValue": 0
    }
  ]
}
```

#### AI Prompt:
```
"Extract numeric amounts from the Amount_Text and Net_Amount_Text fields using regex pattern, then reconcile using both reference numbers and the extracted amounts."
```

#### Expected Results:
- **Extracted Values**: All amount values from text fields
- **Matched**: Records where both reference and extracted amount match
- **Tests**: Regex extraction functionality

---

### 🎯 **Level 6: Date Format Handling**
**Goal**: Test date matching with different formats

#### Configuration:
```json
{
  "Files": [
    {
      "Name": "FileA",
      "Extract": [],
      "Filter": []
    },
    {
      "Name": "FileB",
      "Extract": [],
      "Filter": []
    }
  ],
  "ReconciliationRules": [
    {
      "LeftFileColumn": "Reference", 
      "RightFileColumn": "Ref_Number",
      "MatchType": "equals",
      "ToleranceValue": 0
    },
    {
      "LeftFileColumn": "Date",
      "RightFileColumn": "Process_Date",
      "MatchType": "date_equals",
      "ToleranceValue": 0
    }
  ]
}
```

#### AI Prompt:
```
"Create reconciliation that matches reference numbers and dates, handling the different date formats between files (YYYY-MM-DD vs DD/MM/YYYY)."
```

#### Expected Results:
- **Matched**: Records where dates match after format conversion
- **Tests**: Date format normalization

---

### 🎯 **Level 7: Priority and Category Filtering**
**Goal**: Test complex filtering with multiple conditions

#### Configuration:
```json
{
  "Files": [
    {
      "Name": "FileA",
      "Extract": [],
      "Filter": [
        {
          "ColumnName": "Priority",
          "MatchType": "equals",
          "Value": "High"
        },
        {
          "ColumnName": "Category",
          "MatchType": "contains",
          "Value": "Payment"
        }
      ]
    },
    {
      "Name": "FileB", 
      "Extract": [],
      "Filter": [
        {
          "ColumnName": "Level",
          "MatchType": "equals",
          "Value": "HIGH"
        },
        {
          "ColumnName": "Type",
          "MatchType": "contains", 
          "Value": "PAYMENT"
        }
      ]
    }
  ],
  "ReconciliationRules": [
    {
      "LeftFileColumn": "Reference",
      "RightFileColumn": "Ref_Number",
      "MatchType": "equals",
      "ToleranceValue": 0
    }
  ]
}
```

#### AI Prompt:
```  
"Filter for high priority payment transactions only. File A should include Priority='High' AND Category containing 'Payment'. File B should include Level='HIGH' AND Type containing 'PAYMENT'."
```

#### Expected Results:
- **File A After Filter**: ~3 records (High priority payments)
- **File B After Filter**: ~3 records (HIGH level payments)
- **Matched**: ~2 records
- **Tests**: Multiple filter conditions, case insensitive contains

---

### 🎯 **Level 8: Amount Range Filtering**  
**Goal**: Test numeric range filtering

#### Configuration:
```json
{
  "Files": [
    {
      "Name": "FileA",
      "Extract": [],
      "Filter": [
        {
          "ColumnName": "Amount",
          "MatchType": "greater_than",
          "Value": 1000.00
        }
      ]
    },
    {
      "Name": "FileB",
      "Extract": [],
      "Filter": [
        {
          "ColumnName": "Net_Amount",
          "MatchType": "greater_than", 
          "Value": 1000.00
        }
      ]
    }
  ],
  "ReconciliationRules": [
    {
      "LeftFileColumn": "Reference",
      "RightFileColumn": "Ref_Number",
      "MatchType": "equals",
      "ToleranceValue": 0
    }
  ]
}
```

#### AI Prompt:
```
"Reconcile only high-value transactions over $1000. Filter both files for amounts greater than 1000.00, then match by reference."
```

#### Expected Results:
- **File A After Filter**: ~8 records (amounts > 1000)
- **File B After Filter**: ~8 records (amounts > 1000)
- **Matched**: ~7 records
- **Tests**: Numeric comparison filtering

---

### 🎯 **Level 9: Fuzzy Text Matching**
**Goal**: Test fuzzy matching for similar but not identical text

#### Configuration:
```json
{
  "Files": [
    {
      "Name": "FileA",
      "Extract": [],
      "Filter": []
    },
    {
      "Name": "FileB",
      "Extract": [],
      "Filter": []
    }
  ],
  "ReconciliationRules": [
    {
      "LeftFileColumn": "Description",
      "RightFileColumn": "Transaction_Desc",
      "MatchType": "fuzzy",
      "ToleranceValue": 0.7
    },
    {
      "LeftFileColumn": "Amount",
      "RightFileColumn": "Net_Amount",
      "MatchType": "tolerance", 
      "ToleranceValue": 0.01
    }
  ]
}
```

#### AI Prompt:
```
"Match transactions using fuzzy text matching on descriptions (70% similarity) combined with amount tolerance matching. This catches transactions with slightly different description wording."
```

#### Expected Results:
- **Matched**: Records with similar descriptions and matching amounts
- **Tests**: Fuzzy string matching algorithm

---

### 🎯 **Level 10: Complex Multi-Rule Scenario**
**Goal**: Test most advanced scenario with all features

#### Configuration:
```json
{
  "Files": [
    {
      "Name": "FileA",
      "Extract": [
        {
          "ResultColumnName": "Clean_Amount",
          "SourceColumn": "Amount_Text", 
          "MatchType": "regex",
          "Patterns": ["-?[0-9]+\\.[0-9]{2}"]
        }
      ],
      "Filter": [
        {
          "ColumnName": "Status",
          "MatchType": "in",
          "Value": "Settled,Completed"
        },
        {
          "ColumnName": "Priority",
          "MatchType": "equals",
          "Value": "High"
        }
      ]
    },
    {
      "Name": "FileB",
      "Extract": [
        {
          "ResultColumnName": "Clean_Amount",
          "SourceColumn": "Net_Amount_Text",
          "MatchType": "regex", 
          "Patterns": ["-?[0-9]+\\.[0-9]{2}"]
        }
      ],
      "Filter": [
        {
          "ColumnName": "Settlement_Status",
          "MatchType": "in",
          "Value": "SETTLED,COMPLETE"
        },
        {
          "ColumnName": "Level",
          "MatchType": "equals",
          "Value": "HIGH"
        }
      ]
    }
  ],
  "ReconciliationRules": [
    {
      "LeftFileColumn": "Reference",
      "RightFileColumn": "Ref_Number",
      "MatchType": "equals",
      "ToleranceValue": 0
    },
    {
      "LeftFileColumn": "Clean_Amount", 
      "RightFileColumn": "Clean_Amount",
      "MatchType": "tolerance",
      "ToleranceValue": 0.01
    },
    {
      "LeftFileColumn": "Date",
      "RightFileColumn": "Process_Date",
      "MatchType": "date_equals",
      "ToleranceValue": 0
    },
    {
      "LeftFileColumn": "Account",
      "RightFileColumn": "Account_Number",
      "MatchType": "equals",
      "ToleranceValue": 0
    }
  ]
}
```

#### AI Prompt:
```
"Create the most comprehensive reconciliation configuration that includes: 1) Extract amounts from text fields, 2) Filter for completed high-priority transactions only, 3) Match on reference, amount (with tolerance), date, and account. All conditions must be met for a successful match."
```

#### Expected Results:
- **Complex Processing**: Extraction + Multiple filters + Multi-rule matching
- **Matched**: Only records passing all criteria  
- **Tests**: All features working together

---

## Test Validation Checklist

### ✅ **For Each Test Level:**
1. **File Upload**: Both files upload successfully
2. **Processing**: No errors during reconciliation
3. **Results Count**: Match expected matched/unmatched counts
4. **Data Quality**: Spot check matched records for accuracy
5. **Performance**: Reasonable processing time
6. **Error Handling**: Graceful handling of edge cases

### ✅ **Feature Coverage:**
- [ ] Basic exact matching
- [ ] Case insensitive filtering  
- [ ] Multi-column matching
- [ ] Tolerance matching
- [ ] Text extraction (regex)
- [ ] Date format handling
- [ ] Complex filtering
- [ ] Numeric range filtering
- [ ] Fuzzy text matching
- [ ] Combined advanced features

### ✅ **Edge Cases Covered:**
- [ ] Negative amounts
- [ ] Duplicate references  
- [ ] Case variations
- [ ] Unmatched records
- [ ] Empty/null values
- [ ] Large numbers
- [ ] Date format differences
- [ ] Special characters

---

## Quick Reference

### **Column Mappings:**
| File A | File B | Purpose |
|--------|--------|---------|
| Reference | Ref_Number | Transaction reference |
| Amount | Net_Amount | Transaction amount |
| Date | Process_Date | Transaction date |
| Account | Account_Number | Account identifier |
| Customer_ID | Client_Code | Customer identifier |
| Status | Settlement_Status | Transaction status |
| Priority | Level | Priority level |
| Category | Type | Transaction type |

### **Test Data Highlights:**
- **REF123**: Appears twice in File A (duplicate test)
- **REF143-REF146**: Case sensitivity tests  
- **REF999, REF888**: Unmatched records in File B
- **TXN025/STMT026**: Amount tolerance test (500.01 vs 500.00)
- **Various statuses**: Mixed case testing (Settled/SETTLED/settled)

This comprehensive guide provides everything needed to test all reconciliation features incrementally! 🎯