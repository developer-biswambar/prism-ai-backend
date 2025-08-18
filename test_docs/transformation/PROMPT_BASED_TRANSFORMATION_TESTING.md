# 🤖 Prompt-Based Transformation Testing Guide

## Overview
This guide demonstrates how to test the transformation functionality using AI prompts with the existing test files in the `/docs/testing/transformation` folder. Test via the web interface at `http://localhost:8000` using natural language prompts.

## 📁 Available Test Files
- `customers_test.csv` - 10 customer records with account details
- `transactions_test.csv` - 12 transaction records 
- `products_test.csv` - 11 product records

## 🚀 Prerequisites

### 1. Start the Backend Server
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Access the Web Interface
Open your browser and go to: **`http://localhost:8000`**

### 3. Ensure OpenAI API Key is Set
Verify your `.env` file contains:
```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo
```

---

# 🧪 Test Scenarios Using Prompts

## 🎯 **Test 1: Basic Static Expression (Critical Test)**
**This tests the account_summary issue you reported**

### Step 1: Upload Customer File
1. Go to **`http://localhost:8000`**
2. Navigate to the **Upload** section
3. Upload the file: **`docs/testing/transformation/customers_test.csv`**
4. Note the returned **file_id**

### Step 2: Generate Configuration Using AI Prompt
1. Go to the **Transformation** section
2. Use the **AI Assistance** feature
3. Enter this prompt:

**🤖 PROMPT:**
```
Transform customer data by adding static fields: data_source='Customer Master File', processing_date='2024-01-31', and account_summary that combines account type and balance in a readable format.
```

**📁 FILE TO USE:** `customers_test.csv` (the one you just uploaded)

### Step 3: Execute the Transformation
1. Use the AI-generated configuration 
2. Click **"Process Transformation"**
3. Wait for processing to complete

### ✅ **Expected Results:**
You should see output data with these columns:
- `customer_id`: CUST001, CUST002, etc.
- `data_source`: "Customer Master File" 
- `processing_date`: "2024-01-31"
- `account_summary`: **"Premium account with balance $15000.5"**, **"Standard account with balance $2500.75"**, etc.

### 🚨 **Critical Validation:**
**Check if `account_summary` fields are populated correctly**
- ❌ **FAIL**: If account_summary is empty (`""` or blank)
- ✅ **PASS**: If account_summary shows "Premium account with balance $15000.5" format

---

## 🎯 **Test 2: Dynamic Conditional Logic**

### Steps:
1. Use the same **`customers_test.csv`** file
2. Go to **AI Assistance** 
3. Enter this prompt:

**🤖 PROMPT:**
```
Create customer tiers based on balance: VIP (≥$20k), Premium (≥$10k), Standard (≥$1k), Basic (others). Also add status descriptions for Active, Suspended, and other statuses.
```

**📁 FILE TO USE:** `customers_test.csv`

### ✅ **Expected Results:**
- **VIP Customers**: CUST004 ($25k), CUST010 ($22k) - should show 2 customers
- **Premium Customers**: CUST001 ($15k), CUST007 ($18.5k) - should show 2 customers  
- **Standard Customers**: Should show 4 customers with balance ≥ $1k
- **Basic Customers**: Should show 2 customers with balance < $1k

---

## 🎯 **Test 5: Edge Cases and Error Handling**

### Test 5A: Missing Field Handling
1. Use **`customers_test.csv`** file
2. Enter this prompt:

**🤖 PROMPT:**
```
Create a transformation that references both existing fields (Customer_ID, Balance) and non-existing fields (NonExistentField, MissingColumn). Handle missing fields gracefully with default values.
```

### Test 5B: Special Characters
1. Use **`customers_test.csv`** file  
2. Enter this prompt:

**🤖 PROMPT:**
```
Create expressions with special characters: apostrophes, quotes, symbols (@#$%), and unicode characters. Test expressions like "John's Account" and symbols in output.
```

### ✅ **Expected Results:**
- **Missing Fields**: Should show defaults like "N/A" or "Unknown" instead of errors
- **Special Characters**: Should properly handle apostrophes, quotes, and symbols in names

---

## 🎯 **Test 6: Product Analysis with Mathematical Calculations**

### Steps:
1. Use **`products_test.csv`** file
2. Go to **AI Assistance**
3. Enter this prompt:

**🤖 PROMPT:**
```
Transform product data to calculate profit margins, markup percentages, and inventory status. Create calculated fields: profit_margin as (Retail_Price - Cost_Price), markup_percentage as ((Retail_Price - Cost_Price) / Cost_Price * 100), and inventory_status based on stock levels.
```

**📁 FILE TO USE:** `products_test.csv`

### ✅ **Expected Results:**
- **profit_margin**: $300.00, $99.51, $30.00, $249.01, etc.
- **markup_percentage**: 30.00%, 49.76%, 50.03%, 38.29%, etc. 
- **inventory_status**: "Low Stock" if Stock_Quantity ≤ Reorder_Level, otherwise "In Stock"
- Should show 11 product records with calculations

---

## 🎯 **Test 7: Transaction Analysis with Discounts and Totals**

### Steps:
1. Use **`transactions_test.csv`** file
2. Go to **AI Assistance** 
3. Enter this prompt:

**🤖 PROMPT:**
```
Analyze transaction data by calculating final amounts after discounts, savings amounts, and creating transaction summaries. Calculate: discounted_amount as (Amount * (1 - Discount_Percent/100)), savings_amount as (Amount - discounted_amount), and transaction_summary combining customer, product, and payment details.
```

**📁 FILE TO USE:** `transactions_test.csv`

### ✅ **Expected Results:**
- **discounted_amount**: $1299.99, $269.55, $85.49, $899.00, etc.
- **savings_amount**: $0.00, $29.95, $4.50, $0.00, etc.
- **transaction_summary**: "CUST001 bought Laptop Pro for $1299.99 via Credit Card"
- Should show 12 transaction records with calculations

---

## 🎯 **Test 8: Product Inventory Categories with Complex Logic**

### Steps:
1. Use **`products_test.csv`** file
2. Go to **AI Assistance**
3. Enter this prompt:

**🤖 PROMPT:**
```
Categorize products by inventory status, price tiers, and create detailed product descriptions. Create: inventory_category (Critical/Low/Good/Excellent based on stock vs reorder levels), price_tier (Budget/Standard/Premium/Luxury based on retail price), and full_description combining brand, name, category, and specifications.
```

**📁 FILE TO USE:** `products_test.csv`

### ✅ **Expected Results:**
- **inventory_category**: 
  - "Critical" if Stock_Quantity ≤ Reorder_Level
  - "Low" if Stock_Quantity ≤ Reorder_Level * 2
  - "Good" if Stock_Quantity ≤ Reorder_Level * 4
  - "Excellent" otherwise
- **price_tier**: Budget (<$100), Standard ($100-$500), Premium ($500-$1000), Luxury (>$1000)
- **full_description**: "TechBrand Laptop Pro (Electronics) - Silver Aluminum, 24 month warranty"

---

## 🎯 **Test 9: Transaction Channel and Payment Analysis**

### Steps:
1. Use **`transactions_test.csv`** file
2. Go to **AI Assistance**
3. Enter this prompt:

**🤖 PROMPT:**
```
Analyze transactions by channel performance and payment preferences. Create: channel_type (Online/Physical based on channel), payment_category (Card/Digital/Cash based on payment method), transaction_size (Small/Medium/Large based on amount), and sales_summary with rep performance indicators.
```

**📁 FILE TO USE:** `transactions_test.csv`

### ✅ **Expected Results:**
- **channel_type**: "Online" or "Physical" (Store)
- **payment_category**: "Card" (Credit/Debit), "Digital" (PayPal), "Cash"
- **transaction_size**: Small (<$100), Medium ($100-$500), Large (>$500)
- **sales_summary**: "REP001 processed 3 transactions totaling $3048.98"

---

## 🎯 **Test 10: Multi-Category Product Portfolio Analysis**

### Steps:
1. Use **`products_test.csv`** file
2. Go to **AI Assistance**
3. Enter this prompt:

**🤖 PROMPT:**
```
Create a comprehensive product portfolio analysis with profitability rankings, inventory efficiency scores, and market positioning. Calculate: roi_percentage as (profit_margin / Cost_Price * 100), inventory_turnover_indicator based on stock levels, and product_positioning combining category performance with pricing strategy.
```

**📁 FILE TO USE:** `products_test.csv`

### ✅ **Expected Results:**
- **roi_percentage**: 30.00%, 49.76%, 50.03%, 38.29%, etc.
- **inventory_turnover_indicator**: "Fast" (low stock), "Slow" (high stock)
- **product_positioning**: "Premium Electronics Leader" or "Budget Furniture Option"

---

## 🎯 **Test 11: Transaction Type and Return Analysis**

### Steps:
1. Use **`transactions_test.csv`** file
2. Go to **AI Assistance**
3. Enter this prompt:

**🤖 PROMPT:**
```
Analyze different transaction types including purchases and returns. Create: transaction_impact (Positive for purchases, Negative for returns), net_amount considering transaction type, customer_interaction_type, and return_indicator for tracking customer satisfaction.
```

**📁 FILE TO USE:** `transactions_test.csv`

### ✅ **Expected Results:**
- **transaction_impact**: "Positive" for Purchase, "Negative" for Return
- **net_amount**: Positive amounts for purchases, negative for returns
- **customer_interaction_type**: "Sale", "Return Processing", "Exchange"
- Should identify TXN011 as a return transaction

---

## 🎯 **Test 12: Product Weight and Shipping Analysis**

### Steps:
1. Use **`products_test.csv`** file
2. Go to **AI Assistance**
3. Enter this prompt:

**🤖 PROMPT:**
```
Analyze products for shipping and logistics planning. Calculate: shipping_category based on weight and dimensions, handling_requirements based on material and fragility, estimated_shipping_cost using weight-based calculations, and packaging_needs based on product specifications.
```

**📁 FILE TO USE:** `products_test.csv`

### ✅ **Expected Results:**
- **shipping_category**: "Light" (<1kg), "Medium" (1-5kg), "Heavy" (>5kg)
- **handling_requirements**: "Fragile" (Glass), "Standard" (Plastic/Metal), "Delicate" (Electronics)
- **estimated_shipping_cost**: Based on weight * shipping rate calculations
- **packaging_needs**: "Small Box", "Medium Box", "Large Box", "Special Handling"

---

## 🎯 **Test 13: String Manipulation and Formatting**

### Steps:
1. Use **`customers_test.csv`** file
2. Go to **AI Assistance**
3. Enter this prompt:

**🤖 PROMPT:**
```
Create various string formatting and manipulation examples. Generate: initials from first and last names, formatted_phone with standard format, email_username (part before @), region_code using first 2 letters of region, and formatted_address combining city and region with proper formatting.
```

**📁 FILE TO USE:** `customers_test.csv`

### ✅ **Expected Results:**
- **initials**: "J.S.", "M.J.", "R.B.", etc.
- **formatted_phone**: "(555) 010-1", "(555) 010-2", etc.
- **email_username**: "john.smith", "mary.j", "r.brown", etc.
- **region_code**: "NO", "SO", "EA", "WE", "CE"
- **formatted_address**: "New York, North Region", "Atlanta, South Region"

---

## 🎯 **Test 14: Numerical Analysis and Statistical Calculations**

### Steps:
1. Use **`transactions_test.csv`** file
2. Go to **AI Assistance**
3. Enter this prompt:

**🤖 PROMPT:**
```
Perform numerical analysis on transaction amounts and quantities. Calculate: amount_rounded to nearest dollar, discount_amount in absolute values, quantity_category (Single/Multiple based on quantity), amount_tier using mathematical ranges, and value_efficiency as amount per quantity.
```

**📁 FILE TO USE:** `transactions_test.csv`

### ✅ **Expected Results:**
- **amount_rounded**: $1300, $300, $90, $899, etc.
- **discount_amount**: $0.00, $29.95, $4.50, $0.00, etc.
- **quantity_category**: "Single" (qty=1), "Multiple" (qty>1)
- **amount_tier**: Tier1 (<$100), Tier2 ($100-$500), Tier3 (>$500)
- **value_efficiency**: Amount/Quantity calculations

---

## 🎯 **Test 15: Conditional Logic with Multiple Criteria**

### Steps:
1. Use **`products_test.csv`** file
2. Go to **AI Assistance**
3. Enter this prompt:

**🤖 PROMPT:**
```
Create complex conditional logic based on multiple product criteria. Generate: reorder_priority based on stock levels and category importance, warranty_category based on warranty months, supplier_rating based on supplier performance indicators, and business_priority combining profitability with inventory status.
```

**📁 FILE TO USE:** `products_test.csv`

### ✅ **Expected Results:**
- **reorder_priority**: "Urgent", "High", "Medium", "Low" based on stock vs reorder levels
- **warranty_category**: "Short" (<12 months), "Standard" (12 months), "Extended" (>12 months)
- **supplier_rating**: "A", "B", "C" ratings based on supplier performance
- **business_priority**: "Critical", "Important", "Standard", "Optional"

---

# 🌐 **Browser-Based Testing Instructions**

## Quick Test Sequence

### 1. **Open Your Browser**
- Go to: **`http://localhost:8000`**
- Ensure the server is running first

### 2. **Critical Test: account_summary Issue**
1. **Upload File**: Upload `docs/testing/transformation/customers_test.csv`
2. **AI Prompt**: Use the exact prompt from Test 1 above
3. **Execute**: Run the transformation
4. **Check Results**: Look specifically at the `account_summary` column
   - ❌ **ISSUE EXISTS**: If values are empty/blank
   - ✅ **ISSUE FIXED**: If values show "Premium account with balance $15000.5" format

### 3. **Test All Scenarios**
Work through Tests 1-5 systematically using the web interface

### 4. **Document Results**
For each test, note:
- ✅ **PASS**: Expected results achieved
- ❌ **FAIL**: Issues encountered
- ⚠️ **PARTIAL**: Some issues but mostly working

---

# 🧪 **Validation Checklist**

## ✅ **For Each Prompt Test:**

### 1. **AI Configuration Generation**
- [ ] Prompt generates valid JSON configuration
- [ ] All required fields are present
- [ ] Configuration matches prompt intent
- [ ] Source columns are correctly referenced

### 2. **Transformation Execution**
- [ ] Transformation completes without errors
- [ ] Row counts match expectations
- [ ] All output columns are generated
- [ ] Data types are preserved correctly

### 3. **Result Validation**
- [ ] **account_summary fields are NOT empty** (Critical - Test 1)
- [ ] Static values are applied correctly
- [ ] Dynamic conditions work as expected
- [ ] Mathematical expressions calculate correctly (Tests 6, 7, 10, 12, 14)
- [ ] String expressions concatenate properly (Tests 1, 7, 8, 13)
- [ ] Conditional logic produces correct categories (Tests 2, 8, 9, 11, 15)

### 4. **Mathematical Calculations Validation**
- [ ] Profit margin calculations are accurate (Test 6)
- [ ] Discount calculations work correctly (Test 7)
- [ ] Percentage calculations are proper (Tests 6, 10, 14)
- [ ] Division and multiplication operations succeed
- [ ] Rounding functions work as expected

### 5. **String Manipulation Validation**
- [ ] Text concatenation works properly
- [ ] Special characters are handled correctly (Test 5B)
- [ ] String formatting produces expected results (Test 13)
- [ ] Email parsing and manipulation succeeds

### 6. **Conditional Logic Validation**
- [ ] Single condition logic works (Tests 2, 8)
- [ ] Multiple criteria conditions work (Test 15)
- [ ] Numeric range conditions succeed (Tests 2, 8, 9, 12, 14)
- [ ] String equality conditions work (Tests 9, 11)
- [ ] Default values are applied when conditions fail

### 7. **Error Handling**
- [ ] Missing columns handled gracefully (Test 5A)
- [ ] Invalid expressions return defaults
- [ ] Special characters processed correctly (Test 5B)
- [ ] Large datasets process without timeout

---

# 🚨 **Troubleshooting Guide**

## **Issue: AI Config Generation Fails**
```bash
# Check AI service health
curl -X GET "http://localhost:8000/ai-assistance/health"

# Verify OpenAI API key
curl -X POST "http://localhost:8000/ai-assistance/test-connection"
```

## **Issue: account_summary Still Empty**
1. **Check AI-generated configuration:**
   - Look for `"static_value": "{Account_Type} account with balance ${Balance}"`
   - Verify column names match source data exactly

2. **Test expression evaluation:**
   ```bash
   # Use the simple expression test
   python docs/testing/transformation/simple_expression_test.py
   ```

3. **Debug transformation service:**
   - Check server logs for expression evaluation errors
   - Verify the `evaluate_expression` function is working

## **Issue: Transformation Timeouts**
```bash
# Check system resources
curl -X GET "http://localhost:8000/debug/status"

# Use smaller test files
head -5 customers_test.csv > small_customers.csv
```

---

# 📊 **Expected Test Results Summary**

| Test Scenario | Expected Result | Critical? |
|---------------|----------------|-----------|
| **account_summary generation** | "Premium account with balance $15000.5" | ✅ YES |
| **Customer tier classification** | VIP: 2, Premium: 2, Standard: 4, Basic: 2 | No |
| **Multi-file joins** | ~25 records with customer+transaction data | No |
| **Mathematical expressions** | Calculated fields with correct values | No |
| **Error handling** | Graceful defaults for missing fields | No |

## 🎯 **Success Criteria**
- **Primary Goal**: account_summary fields must NOT be empty
- **Secondary Goal**: All generated configurations execute successfully  
- **Tertiary Goal**: Results match prompt intentions

---

---

# 📊 **Test Summary Matrix**

| Test | Focus Area | File Used | Key Features Tested | Expected Records |
|------|------------|-----------|-------------------|------------------|
| **Test 1** | 🔥 Critical account_summary | customers_test.csv | Static expressions, string interpolation | 10 |
| **Test 2** | Dynamic conditions | customers_test.csv | Conditional logic, numeric comparisons | 10 |
| **Test 3** | Multi-file joins | All 3 CSV files | File relationships, complex queries | ~11 |
| **Test 4** | Complex expressions | customers_test.csv | ~~Date functions~~ Mathematical operations | 10 |
| **Test 5A** | Error handling | customers_test.csv | Missing field handling | 10 |
| **Test 5B** | Special characters | customers_test.csv | String parsing, special chars | 10 |
| **Test 6** | Mathematical calculations | products_test.csv | Profit margins, percentages | 11 |
| **Test 7** | Discount calculations | transactions_test.csv | Financial calculations | 12 |
| **Test 8** | Inventory categorization | products_test.csv | Multi-level conditions | 11 |
| **Test 9** | Channel analysis | transactions_test.csv | String categorization | 12 |
| **Test 10** | Portfolio analysis | products_test.csv | ROI calculations, ratios | 11 |
| **Test 11** | Transaction types | transactions_test.csv | Purchase/Return logic | 12 |
| **Test 12** | Shipping analysis | products_test.csv | Weight-based calculations | 11 |
| **Test 13** | String manipulation | customers_test.csv | Text formatting, parsing | 10 |
| **Test 14** | Statistical analysis | transactions_test.csv | Numerical operations | 12 |
| **Test 15** | Complex conditionals | products_test.csv | Multi-criteria logic | 11 |

## 🎯 **Feature Coverage**

### ✅ **Supported Features (Current Code)**
- **Static Value Assignment**: Tests 1, 3, 6, 7, 8, 13
- **String Concatenation**: Tests 1, 7, 8, 13  
- **Mathematical Operations**: Tests 6, 7, 10, 12, 14
- **Conditional Logic**: Tests 2, 8, 9, 11, 15
- **Dynamic Mapping**: Tests 2, 8, 9, 11, 15
- **Multi-File Processing**: Test 3
- **Numeric Functions**: round(), abs(), min(), max()
- **String Functions**: Basic concatenation and formatting
- **Error Handling**: Tests 5A, 5B

### ❌ **NOT Supported (Avoid These)**
- **Date Functions**: DATEDIFF(), NOW(), TODAY()
- **Advanced String Functions**: SUBSTRING(), CHARINDEX()
- **SQL-style Functions**: CASE WHEN, CONCAT()
- **Aggregation Functions**: SUM(), AVG(), COUNT() (across rows)

## 🔥 **Priority Testing Order**

1. **🚨 CRITICAL**: Test 1 (account_summary issue)
2. **🏗️ FOUNDATION**: Tests 2, 5A, 5B (basic functionality)
3. **🧮 CALCULATIONS**: Tests 6, 7, 14 (mathematical operations)
4. **🔄 LOGIC**: Tests 8, 9, 11, 15 (conditional logic)
5. **🎨 FORMATTING**: Tests 10, 12, 13 (advanced formatting)
6. **🔗 INTEGRATION**: Test 3 (multi-file processing)

**📝 Note**: This guide focuses on testing transformation functionality through AI prompts using features that are actually supported by the current code. All prompts avoid unsupported date functions and focus on mathematical operations, string manipulation, and conditional logic that the system can handle.