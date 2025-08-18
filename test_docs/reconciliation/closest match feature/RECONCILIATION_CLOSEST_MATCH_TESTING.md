# Reconciliation with Closest Match Testing Guide

## Overview
This guide demonstrates how to test the new **Composite Similarity Closest Match** functionality in the reconciliation system. The feature uses multiple algorithms (rapidfuzz) to find the closest matches for unmatched records.

## Test Data Description

### File A: `recon_test_file_a.csv`
- **10 records** with transaction data
- Columns: `transaction_id`, `customer_name`, `amount`, `date`, `account_number`, `description`
- Contains exact matches and near-matches for testing

### File B: `recon_test_file_b.csv` 
- **12 records** with similar transaction data
- Columns: `ref_id`, `client_name`, `value`, `transaction_date`, `acc_no`, `notes`
- Contains exact matches, near-matches, and 2 extra records

## Expected Reconciliation Results

### Exact Matches (6 records):
- TXN001: John Smith, $1000.50
- TXN003: Bob Johnson/Robert Johnson (name variation), $750.25
- INV004: Alice Brown, $1200.00 
- CON005: Charlie Davis/Charles Davis (name variation), $3000.75
- UTL006: Eva Wilson, $500.00
- TXN009: Henry Clark, $2200.00

### Unmatched File A (4 records):
- TXN002: Jane Doe (different ref_id: REF002)
- TXN007: Frank Miller (different ref_id: EQP007)
- TXN008: Grace Taylor (different ref_id: CST008)
- TXN010: Ivy Rodriguez (different ref_id: TRV010)

### Unmatched File B (6 records):
- REF002, EQP007, CST008, TRV010 (different IDs but same data)
- TXN011: Mike Anderson (new customer)
- TXN012: Sarah Wilson (new subscription)

## Step-by-Step Testing Instructions

### Step 1: Start the Backend Server
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 2: Start the Frontend Server
```bash
cd frontend
npm run dev
```

### Step 3: Upload Test Files

1. **Navigate to:** `http://localhost:5174`
2. **Upload File A:**
   - Click "Upload Files" 
   - Select `backend/test_data/recon_test_file_a.csv`
   - Note the file ID (e.g., `file_abc123`)

3. **Upload File B:**
   - Upload `backend/test_data/recon_test_file_b.csv`
   - Note the file ID (e.g., `file_def456`)

### Step 4: Test Basic Reconciliation (Without Closest Match)

#### Option A: Using Frontend UI
1. Go to **Reconciliation** tab
2. Select both uploaded files
3. Configure reconciliation rules:
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
         "LeftFileColumn": "transaction_id",
         "RightFileColumn": "ref_id",
         "MatchType": "equals",
         "ToleranceValue": 0
       },
       {
         "LeftFileColumn": "amount",
         "RightFileColumn": "value", 
         "MatchType": "tolerance",
         "ToleranceValue": 0.01
       }
     ]
   }
   ```
4. **DO NOT** check the "Find Closest Matches" option yet
5. Run reconciliation
6. **Expected Results:** ~6 matches, ~4 unmatched A, ~6 unmatched B

#### Option B: Using API Directly
```bash
curl -X POST "http://localhost:8000/reconciliation/process/" \
  -H "Content-Type: application/json" \
  -d '{
    "process_type": "reconciliation",
    "process_name": "Test Basic Reconciliation",
    "user_requirements": "Match transactions by ID and amount",
    "find_closest_matches": false,
    "files": [
      {"file_id": "YOUR_FILE_A_ID", "role": "file_0", "label": "File A"},
      {"file_id": "YOUR_FILE_B_ID", "role": "file_1", "label": "File B"}
    ],
    "reconciliation_config": {
      "Files": [
        {"Name": "FileA", "Extract": [], "Filter": []},
        {"Name": "FileB", "Extract": [], "Filter": []}
      ],
      "ReconciliationRules": [
        {
          "LeftFileColumn": "transaction_id",
          "RightFileColumn": "ref_id",
          "MatchType": "equals",
          "ToleranceValue": 0
        },
        {
          "LeftFileColumn": "amount", 
          "RightFileColumn": "value",
          "MatchType": "tolerance",
          "ToleranceValue": 0.01
        }
      ]
    }
  }'
```

### Step 5: Test Reconciliation WITH Closest Match

#### Option A: Using Frontend UI
1. Use the same configuration as Step 4
2. **CHECK** the "Find Closest Matches" option ✅
3. Run reconciliation
4. **Expected Results:** Same matches, but unmatched records now have 3 additional columns:
   - `closest_match_record`: Details of the closest match
   - `closest_match_score`: Similarity score (0-100)
   - `closest_match_details`: Shows only mismatched columns with current and suggested values

#### Option B: Using API Directly
```bash
curl -X POST "http://localhost:8000/reconciliation/process/" \
  -H "Content-Type: application/json" \
  -d '{
    "process_type": "reconciliation",
    "process_name": "Test Closest Match Reconciliation", 
    "user_requirements": "Match transactions and find closest matches for unmatched records",
    "find_closest_matches": true,
    "files": [
      {"file_id": "YOUR_FILE_A_ID", "role": "file_0", "label": "File A"},
      {"file_id": "YOUR_FILE_B_ID", "role": "file_1", "label": "File B"}
    ],
    "reconciliation_config": {
      "Files": [
        {"Name": "FileA", "Extract": [], "Filter": []},
        {"Name": "FileB", "Extract": [], "Filter": []}
      ],
      "ReconciliationRules": [
        {
          "LeftFileColumn": "transaction_id",
          "RightFileColumn": "ref_id", 
          "MatchType": "equals",
          "ToleranceValue": 0
        },
        {
          "LeftFileColumn": "amount",
          "RightFileColumn": "value",
          "MatchType": "tolerance", 
          "ToleranceValue": 0.01
        }
      ]
    }
  }'
```

### Step 6: Analyze Closest Match Results

#### Understanding the Simplified Closest Match Details Format

The `closest_match_details` column now shows **only the columns that don't match exactly**, in this simplified format:
```json
{
  "column_name": {
    "current_value": "value_in_source_file", 
    "suggested_value": "value_in_target_file"
  }
}
```

**Example**: If a record has `transaction_id: "TXN002"` but the closest match has `ref_id: "REF002"`, you'll see:
```
transaction_id: 'TXN002' → 'REF002'
```

**Benefits of this format**:
- ✅ Shows only the mismatched columns (ignores perfect matches)
- ✅ Clear "what you have" vs "what it could be" format  
- ✅ Easier to understand which changes would create a match
- ✅ Focuses on actionable information

#### Expected Closest Matches for Unmatched File A Records:

1. **TXN002 (Jane Doe)** should match **REF002 (Jane Doe)**
   - Score: ~95-100 (same customer name, amount, date)
   - Only difference: transaction_id vs ref_id

2. **TXN007 (Frank Miller)** should match **EQP007 (Frank Miller)**
   - Score: ~95-100 (same customer, amount, date)
   - Only difference: transaction_id vs ref_id

3. **TXN008 (Grace Taylor)** should match **CST008 (Grace Taylor)**
   - Score: ~95-100 (same customer, amount, date)

4. **TXN010 (Ivy Rodriguez)** should match **TRV010 (Ivy Rodriguez)**
   - Score: ~95-100 (same customer, amount, date)

#### View Results:
1. **In Frontend:** Click "View Unmatched A Results" or "View Unmatched B Results"
2. **Via API:** 
   ```bash
   curl "http://localhost:8000/reconciliation/results/YOUR_RECON_ID?result_type=unmatched_a"
   ```

### Step 7: Verify Similarity Algorithms

The system uses these algorithms with weights:
- **Text Fields (names):**
  - Basic ratio (30%)
  - Partial ratio (20%) 
  - Token sort ratio (25%)
  - Token set ratio (25%)

- **Numeric Fields (amounts):**
  - Percentage difference calculation
  - 0% diff = 100% similarity

- **Date Fields:**
  - Exact date = 100%
  - 1 day difference = 95%
  - 7+ days = decreasing similarity

- **Identifier Fields (IDs):**
  - More strict matching
  - Higher weight on exact similarity

### Step 8: Performance Testing

For larger datasets:
1. Create files with 1000+ records
2. Enable closest matching
3. Monitor console logs for performance metrics
4. Expected processing time: <10 seconds for 1000x1000 comparisons

## API Response Structure

### Without Closest Match:
```json
{
  "matched": [...],
  "unmatched_file_a": [
    {
      "transaction_id": "TXN002",
      "customer_name": "Jane Doe",
      "amount": 2500.00,
      ...
    }
  ],
  "unmatched_file_b": [...]
}
```

### With Closest Match:
```json
{
  "unmatched_file_a": [
    {
      "transaction_id": "TXN002",
      "customer_name": "Jane Doe", 
      "amount": 2500.00,
      "closest_match_record": "{\"ref_id\":\"REF002\",\"client_name\":\"Jane Doe\",...}",
      "closest_match_score": 98.5,
      "closest_match_details": "transaction_id: 'TXN002' → 'REF002'"
    }
  ]
}
```

## Troubleshooting

### Common Issues:

1. **No closest matches found:**
   - Check if unmatched records exist in both files
   - Verify reconciliation rules are correct
   - Ensure column names match the rules

2. **Low similarity scores:**
   - Expected for truly different records
   - Check data quality and column types

3. **Performance issues:**
   - Reduce dataset size for testing
   - Monitor server logs for memory usage

### Debugging Tips:

1. **Check server logs:**
   ```bash
   # Look for these messages:
   "Finding closest matches for X unmatched A records..."
   "Comparing N column pairs for closest matches"
   "Added closest match information to X records"
   ```

2. **Verify API response:**
   - Check if closest_match_* columns are present
   - Validate similarity scores are reasonable (0-100)

3. **Test with simpler data:**
   - Start with 2-3 records per file
   - Use obvious near-matches for validation

## Success Criteria

✅ **Basic reconciliation works** (exact matches found)  
✅ **Closest match columns added** to unmatched records  
✅ **Similarity scores reasonable** (>90 for near-matches)  
✅ **Performance acceptable** (<10s for moderate datasets)  
✅ **No errors in server logs**  
✅ **API responses well-formatted**  

## Advanced Testing Scenarios

### Scenario 1: Mixed Data Types
Test with various column types:
- Dates in different formats
- Amounts with different precision
- Names with spelling variations

### Scenario 2: Large Datasets
- 1000+ records in each file
- Monitor memory usage and performance

### Scenario 3: Edge Cases  
- Files with no unmatched records
- Files with all unmatched records
- Empty datasets

This comprehensive testing ensures the closest match functionality works correctly across different scenarios and data types.