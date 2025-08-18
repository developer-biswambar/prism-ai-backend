# Closest Match Feature Testing Guide

## Overview
The closest match feature helps identify potential matches when records don't match exactly but are similar enough to warrant investigation. This is particularly useful for financial reconciliation where data formats, names, or amounts might have slight variations.

## Test Files

### File A: `closest_match_file_a.csv`
- **Source**: Internal transaction system
- **15 records** with transaction details
- **Key columns**: transaction_id, customer_name, amount, date, reference

### File B: `closest_match_file_b.csv`  
- **Source**: Bank/payment processor system
- **15 records** (13 matching + 2 new records)
- **Key columns**: ref_number, client_name, total_amount, payment_date, transaction_ref

## Data Variations Designed for Testing

### 1. **Exact Matches** (Should match normally)
- REF001/TXN001: John Smith, $1500.00
- REF002/TXN002: Sarah Johnson → S. Johnson, $2350.75
- REF013/TXN013: Christopher Lee → Chris Lee, $990.00

### 2. **Name Variations** (Closest match candidates)
| File A | File B | Variation Type |
|--------|--------|----------------|
| John Smith | John Smith Jr | Middle name/suffix |
| Sarah Johnson | S. Johnson | First name abbreviation |
| Michael Brown | Mike Brown | Nickname |
| Emily Davis | Emily J Davis | Middle initial |
| David Wilson | D. Wilson | First name abbreviation |
| Christopher Lee | Chris Lee | Nickname |
| Maria Rodriguez | Maria R Rodriguez | Middle initial |
| Thomas Anderson | Tom Anderson | Nickname |

### 3. **Amount Variations** (Small differences)
| File A | File B | Difference | Reason |
|--------|--------|------------|---------|
| $899.99 | $899.95 | $0.04 | Rounding/fees |
| $1750.25 | $1750.30 | $0.05 | Processing fee |
| $2800.30 | $2800.25 | $0.05 | Exchange rate |
| $1200.00 | $1200.05 | $0.05 | Interest |
| $720.50 | $720.55 | $0.05 | Tax adjustment |
| $1850.25 | $1850.20 | $0.05 | Discount |
| $3750.80 | $3750.75 | $0.05 | Fee difference |

### 4. **Date Format Differences**
- **File A**: YYYY-MM-DD (2024-01-15)
- **File B**: DD/MM/YYYY (15/01/2024)

### 5. **Missing Records** (Unmatched)
- **File A only**: REF006 (Lisa Garcia), REF015 (Daniel Martinez)
- **File B only**: REF016 (New Client), REF017 (Another Client)

## Testing Scenarios

### Scenario 1: Basic Reconciliation Rules
```json
{
  "toleranceMatching": true,
  "toleranceAmount": 0.10,
  "ReconciliationRules": [
    {
      "FileAColumn": "reference",
      "FileBColumn": "ref_number", 
      "MatchType": "equals"
    },
    {
      "FileAColumn": "amount",
      "FileBColumn": "total_amount",
      "MatchType": "numeric_tolerance",
      "Tolerance": 0.10
    }
  ]
}
```
**Expected Results**:
- 11 exact matches
- 2 unmatched in File A (REF006, REF015)
- 4 unmatched in File B (REF016, REF017, plus 2 others)

### Scenario 2: Enable Closest Match
```json
{
  "toleranceMatching": true,
  "toleranceAmount": 0.10,
  "findClosestMatches": true,
  "closestMatchConfig": {
    "enabled": true,
    "maxCandidates": 3,
    "minimumScore": 0.6,
    "considerAmount": true,
    "considerName": true,
    "considerDate": true,
    "amountTolerance": 0.10,
    "dateTolerance": 5
  },
  "ReconciliationRules": [...]
}
```
**Expected Results**:
- 11 exact matches  
- 2-4 closest match candidates for unmatched records
- Higher confidence scores for name similarities

### Scenario 3: Strict Closest Match
```json
{
  "closestMatchConfig": {
    "enabled": true,
    "maxCandidates": 2,
    "minimumScore": 0.8,
    "considerAmount": true,
    "considerName": true,
    "considerDate": true,
    "amountTolerance": 0.05,
    "dateTolerance": 2
  }
}
```
**Expected Results**:
- Only high-confidence matches shown
- Fewer candidates per unmatched record

## Key Test Points

### 1. **Name Similarity Scoring**
Test how the system scores these name pairs:
- "John Smith" vs "John Smith Jr" (high score)
- "Michael Brown" vs "Mike Brown" (high score)  
- "Sarah Johnson" vs "S. Johnson" (medium score)
- "David Wilson" vs "D. Wilson" (medium score)

### 2. **Amount Tolerance**
Verify that small amount differences are handled:
- $0.04-$0.05 differences should be caught with 0.10 tolerance
- Should not match with 0.01 tolerance

### 3. **Date Format Handling**
Ensure different date formats are normalized:
- "2024-01-15" should match "15/01/2024"

### 4. **Performance with Multiple Candidates**
Test with maxCandidates = 1, 3, 5 to see:
- Response time differences
- Quality of top candidates

### 5. **Minimum Score Thresholds**
Test with minimumScore = 0.5, 0.7, 0.9 to verify:
- Low scores filter out poor matches
- High scores only show confident matches

## Expected Match Pairs for Closest Match

| File A Record | File B Candidate | Match Reason | Expected Score |
|---------------|------------------|--------------|----------------|
| REF006 (Lisa Garcia) | No good candidates | Name too different | < 0.6 |
| REF015 (Daniel Martinez) | No good candidates | Name too different | < 0.6 |
| Missing exact matches | Various candidates | Name/amount similarity | 0.6-0.9 |

## Step-by-Step Testing Instructions

### Step 1: File Upload
1. Navigate to the application's file upload section
2. Upload `closest_match_file_a.csv` as File A
3. Upload `closest_match_file_b.csv` as File B
4. **Expected**: Both files should upload successfully and show 15 records each

### Step 2: Basic Reconciliation (Without Closest Match)
1. Navigate to Reconciliation section
2. Select the uploaded files
3. Configure basic reconciliation rules:
   ```json
   {
     "toleranceMatching": true,
     "toleranceAmount": 0.10,
     "findClosestMatches": false,
     "ReconciliationRules": [
       {
         "FileAColumn": "reference",
         "FileBColumn": "ref_number", 
         "MatchType": "equals"
       },
       {
         "FileAColumn": "amount",
         "FileBColumn": "total_amount",
         "MatchType": "numeric_tolerance",
         "Tolerance": 0.10
       }
     ]
   }
   ```
4. Run reconciliation
5. **Expected Results**:
   - **11 exact matches** (REF001-REF005, REF007-REF012, REF014)
   - **2 unmatched in File A**: REF006 (Lisa Garcia), REF015 (Daniel Martinez)
   - **4 unmatched in File B**: REF016, REF017, plus 2 others with slight variations

### Step 3: Enable Closest Match Feature
1. Use the same file selection
2. Enable closest match with this configuration:
   ```json
   {
     "toleranceMatching": true,
     "toleranceAmount": 0.10,
     "findClosestMatches": true,
     "closestMatchConfig": {
       "enabled": true,
       "maxCandidates": 3,
       "minimumScore": 0.6,
       "considerAmount": true,
       "considerName": true,
       "considerDate": true,
       "amountTolerance": 0.10,
       "dateTolerance": 5
     },
     "ReconciliationRules": [
       {
         "FileAColumn": "reference",
         "FileBColumn": "ref_number", 
         "MatchType": "equals"
       },
       {
         "FileAColumn": "amount",
         "FileBColumn": "total_amount",
         "MatchType": "numeric_tolerance",
         "Tolerance": 0.10
       }
     ]
   }
   ```
3. Run reconciliation
4. **Expected Results**:
   - **Same 11 exact matches** as before
   - **Closest match candidates** should appear for unmatched records
   - Look for records with similar names/amounts but different enough to not exact match

### Step 4: Verify Closest Match Quality
Check that closest match suggestions include:

1. **High Confidence Matches (Score > 0.8)**:
   - Records with minor name variations (John Smith vs John Smith Jr)
   - Records with nickname differences (Michael vs Mike)
   - Records with amount differences within tolerance

2. **Medium Confidence Matches (Score 0.6-0.8)**:
   - Records with abbreviated names (Sarah Johnson vs S. Johnson)
   - Records with minor formatting differences

3. **Filtered Out (Score < 0.6)**:
   - REF006 (Lisa Garcia) should have no good candidates
   - REF015 (Daniel Martinez) should have no good candidates

### Step 5: Test Different Tolerance Settings
1. **Strict Configuration** (Higher quality, fewer candidates):
   ```json
   "closestMatchConfig": {
     "enabled": true,
     "maxCandidates": 2,
     "minimumScore": 0.8,
     "amountTolerance": 0.05,
     "dateTolerance": 2
   }
   ```
   **Expected**: Only very high confidence matches shown

2. **Lenient Configuration** (More candidates, lower quality):
   ```json
   "closestMatchConfig": {
     "enabled": true,
     "maxCandidates": 5,
     "minimumScore": 0.4,
     "amountTolerance": 0.20,
     "dateTolerance": 10
   }
   ```
   **Expected**: More match candidates, including some lower quality ones

### Step 6: Performance Validation
1. Measure processing time with closest match enabled vs disabled
2. Test with different `maxCandidates` values (1, 3, 5)
3. **Expected**: Small increase in processing time, proportional to maxCandidates

### Step 7: Result Validation
Verify the results contain:

1. **Match Summary**:
   - Total records in each file
   - Number of exact matches
   - Number of unmatched records
   - Number of closest match candidates found

2. **Closest Match Details**:
   - Candidate records with similarity scores
   - Reason for match suggestion (name similarity, amount proximity, etc.)
   - Confidence level for each suggestion

3. **Export Functionality**:
   - Results can be exported to CSV/Excel
   - Closest match suggestions are clearly marked
   - Scores and match reasons are included

## Expected Specific Results

### File A Records That Should Find Closest Matches:
- **REF006 (Lisa Garcia, $899.99)**: Should find no good candidates (score < 0.6)
- **REF015 (Daniel Martinez, $3750.80)**: Should find no good candidates (score < 0.6)

### File B Records That Should Be Closest Match Candidates:
- Records with name variations of unmatched File A records
- Records with similar amounts but different names
- New records (REF016, REF017) that don't match anything in File A

## API Testing Commands

```bash
# Upload files
curl -X POST http://localhost:8000/files/upload \
  -F "file=@closest_match_file_a.csv"

curl -X POST http://localhost:8000/files/upload \
  -F "file=@closest_match_file_b.csv"

# Test reconciliation with closest match
curl -X POST http://localhost:8000/reconciliation/process \
  -H "Content-Type: application/json" \
  -d @closest_match_config.json
```

## Success Criteria

✅ **Closest match feature correctly identifies potential matches**  
✅ **Similarity scores are reasonable and consistent**  
✅ **Poor matches are filtered out by minimum score threshold**  
✅ **Amount and date tolerances work as configured**  
✅ **Performance impact is acceptable (< 2x processing time)**  
✅ **Results are clearly presented with match confidence levels**  
✅ **Export functionality includes closest match data**

This comprehensive test setup will help you validate that the closest match feature correctly identifies potential matches while filtering out poor candidates.