# 👥 Customer Data Transformation Testing Guide

## Overview
This guide focuses specifically on testing transformation scenarios using the **`customers_test.csv`** file. All tests are designed for browser-based testing at `http://localhost:8000`.

## 📁 Test File
- **File**: `customers_test.csv`
- **Records**: 10 customer records
- **Columns**: Customer_ID, First_Name, Last_Name, Email, Phone, Date_Joined, Account_Type, Balance, Status, Region, City, Age, Gender, Membership_Level, Last_Login

## 🚀 Prerequisites
1. Start backend server: `python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
2. Open browser: `http://localhost:8000`
3. Upload `customers_test.csv` file

---

# 🧪 Customer Transformation Tests

## 🎯 **Test C1: Critical Account Summary Generation**
**🔥 Priority: CRITICAL - Tests the reported account_summary bug**

### Steps:
1. Upload **`customers_test.csv`**
2. Go to **AI Assistance**
3. Enter this prompt:

**🤖 PROMPT:**
```
Transform customer data by adding static fields: data_source='Customer Master File', processing_date='2024-01-31', and account_summary that combines account type and balance in a readable format.
```

### ✅ **Expected Results:**
- **account_summary**: "Premium account with balance $15000.5", "Standard account with balance $2500.75", etc.
- **data_source**: "Customer Master File" for all records
- **processing_date**: "2024-01-31" for all records
- **Total Records**: 10

### 🚨 **Critical Validation:**
- ❌ **FAIL**: If account_summary is empty or blank
- ✅ **PASS**: If account_summary contains proper format like "Premium account with balance $15000.5"

---

## 🎯 **Test C2: Customer Tier Classification**

### Steps:
1. Use **`customers_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Create customer tiers based on balance: VIP (≥$20k), Premium (≥$10k), Standard (≥$1k), Basic (others). Also add status descriptions for Active, Suspended, and other statuses.
```

### ✅ **Expected Results:**
- **VIP Customers**: 2 customers (CUST004: $25k, CUST010: $22k)
- **Premium Customers**: 2 customers (CUST001: $15k, CUST007: $18.5k)
- **Standard Customers**: 4 customers (balance $1k-$10k)
- **Basic Customers**: 2 customers (balance <$1k)
- **Status Descriptions**: Active = "Account is active and in good standing"

---

## 🎯 **Test C3: Personal Information Formatting**

### Steps:
1. Use **`customers_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Create formatted customer profiles with full names, contact summaries, and location details. Generate: full_name combining first and last names, contact_info with email and phone, location_summary with city and region, and customer_profile with age and gender information.
```

### ✅ **Expected Results:**
- **full_name**: "John Smith", "Mary Johnson", "Robert Brown", etc.
- **contact_info**: "john.smith@email.com | 555-0101"
- **location_summary**: "New York, North region"
- **customer_profile**: "35 year old Male" or "28 year old Female"

---

## 🎯 **Test C4: Account Status and Risk Analysis**

### Steps:
1. Use **`customers_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Analyze customer accounts for risk assessment and status monitoring. Create: risk_level based on status and balance, account_health combining multiple factors, balance_category (High/Medium/Low based on ranges), and alert_flags for accounts needing attention.
```

### ✅ **Expected Results:**
- **risk_level**: "High Risk" for Suspended accounts, "Low Risk" for Active with high balance
- **account_health**: "Excellent", "Good", "Moderate", "Poor"
- **balance_category**: "High" (>$15k), "Medium" ($5k-$15k), "Low" (<$5k)
- **alert_flags**: "SUSPENDED" for inactive accounts, "LOW_BALANCE" for balances <$1k

---

## 🎯 **Test C5: String Manipulation and Parsing**

### Steps:
1. Use **`customers_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Create various string formatting examples from customer data. Generate: initials from names, email_domain extracted from email addresses, phone_formatted in standard format, region_code using first 2 letters, and membership_display combining level and status.
```

### ✅ **Expected Results:**
- **initials**: "J.S.", "M.J.", "R.B.", "L.D.", etc.
- **email_domain**: "email.com" extracted from addresses
- **phone_formatted**: "(555) 010-1", "(555) 010-2", etc.
- **region_code**: "NO", "SO", "EA", "WE", "CE"
- **membership_display**: "Gold Member (Active)", "Silver Member (Active)"

---

## 🎯 **Test C6: Age and Demographics Analysis**

### Steps:
1. Use **`customers_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Analyze customer demographics and age groups. Create: age_category (Young/Adult/Senior based on age ranges), generation_group (Gen Z/Millennial/Gen X based on age), gender_display with proper formatting, and demographic_summary combining age and gender insights.
```

### ✅ **Expected Results:**
- **age_category**: "Young" (<30), "Adult" (30-50), "Senior" (>50)
- **generation_group**: "Gen Z" (<25), "Millennial" (25-40), "Gen X" (40-55)
- **gender_display**: "Male Customer", "Female Customer"
- **demographic_summary**: "35-year-old Male Millennial", "28-year-old Female Millennial"

---

## 🎯 **Test C7: Membership and Loyalty Analysis**

### Steps:
1. Use **`customers_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Analyze customer membership levels and loyalty indicators. Create: membership_value based on level ranking, loyalty_score combining membership and balance, upgrade_eligibility for next membership tier, and customer_segment for marketing purposes.
```

### ✅ **Expected Results:**
- **membership_value**: "Platinum" = 4, "Gold" = 3, "Silver" = 2, "Bronze" = 1
- **loyalty_score**: Calculated score combining membership level and balance
- **upgrade_eligibility**: "Eligible for Gold", "Eligible for Platinum", "Maximum Level"
- **customer_segment**: "High Value VIP", "Growth Potential", "Standard Customer"

---

## 🎯 **Test C8: Contact and Communication Preferences**

### Steps:
1. Use **`customers_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Analyze customer contact information and create communication profiles. Generate: preferred_contact combining email and phone preferences, contact_quality based on information completeness, communication_channel recommendations, and outreach_priority based on customer value.
```

### ✅ **Expected Results:**
- **preferred_contact**: "Email Primary (john.smith@email.com)", "Phone Available (555-0101)"
- **contact_quality**: "Complete" (both email and phone), "Partial" (missing info)
- **communication_channel**: "Email Marketing", "Phone Outreach", "Digital Marketing"
- **outreach_priority**: "High Priority" (VIP customers), "Medium Priority", "Low Priority"

---

## 🎯 **Test C9: Error Handling and Edge Cases**

### Steps:
1. Use **`customers_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Test error handling by referencing both existing fields (Customer_ID, Balance) and non-existing fields (NonExistentField, MissingColumn). Create: safe_customer_id from existing field, safe_missing_field with default handling, combined_data mixing valid and invalid references, and error_resistant_summary.
```

### ✅ **Expected Results:**
- **safe_customer_id**: Should populate correctly (CUST001, CUST002, etc.)
- **safe_missing_field**: Should show defaults like "N/A" or "Unknown"
- **combined_data**: Should handle mix of valid and invalid fields gracefully
- **error_resistant_summary**: Should not break the transformation

---

## 🎯 **Test C10: Special Characters and Formatting**

### Steps:
1. Use **`customers_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Test special character handling and advanced formatting. Create: formatted_name with apostrophes and titles, special_symbols using various characters (@#$%), quoted_text with embedded quotes, and unicode_safe_display ensuring proper character handling.
```

### ✅ **Expected Results:**
- **formatted_name**: "Mr. John Smith's Account", "Ms. Mary Johnson's Profile"
- **special_symbols**: "Customer #CUST001 @email.com $15000.50"
- **quoted_text**: Proper handling of quotes in strings
- **unicode_safe_display**: No character encoding issues

---

# 📊 **Customer Test Summary**

| Test | Focus Area | Expected Records | Key Validation |
|------|------------|------------------|----------------|
| **C1** | 🔥 account_summary (Critical) | 10 | Must NOT be empty |
| **C2** | Conditional logic | 10 | Correct tier classification |
| **C3** | String concatenation | 10 | Proper name/contact formatting |
| **C4** | Risk analysis | 10 | Logical risk categories |
| **C5** | String parsing | 10 | Correct extraction/formatting |
| **C6** | Demographics | 10 | Age/generation categorization |
| **C7** | Loyalty analysis | 10 | Membership calculations |
| **C8** | Contact preferences | 10 | Communication logic |
| **C9** | Error handling | 10 | Graceful failure handling |
| **C10** | Special characters | 10 | Character encoding safety |

## 🎯 **Testing Priority**
1. **🚨 CRITICAL**: Test C1 (account_summary bug)
2. **🏗️ FOUNDATION**: Tests C2, C3 (basic functionality)
3. **🧮 LOGIC**: Tests C4, C6, C7 (conditional operations)
4. **🎨 FORMATTING**: Tests C5, C8 (string manipulation)
5. **🛡️ SAFETY**: Tests C9, C10 (error handling)

## ✅ **Expected Customer Data Insights**
- **High-Value Customers**: CUST004 ($25k), CUST001 ($15k), CUST007 ($18.5k), CUST010 ($22k)
- **Low-Balance Customers**: CUST003 ($750), CUST009 ($950)
- **Demographics**: Mix of ages 26-42, balanced gender distribution
- **Regions**: North, South, East, West, Central representation
- **Membership Levels**: Platinum, Gold, Silver, Bronze distribution

This focused guide allows comprehensive testing of all customer-related transformation scenarios!