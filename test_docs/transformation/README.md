# 🧪 Transformation Testing Suite - Complete Guide

## Overview
This directory contains comprehensive transformation testing guides organized by data source and complexity level. All tests are designed for browser-based testing at `http://localhost:8000` using AI prompts.

## 📁 Testing Guide Structure

### **Individual File Testing**
| Guide File | Focus | Tests | Priority |
|------------|-------|-------|----------|
| **[CUSTOMERS_TRANSFORMATION_TESTS.md](CUSTOMERS_TRANSFORMATION_TESTS.md)** | 👥 Customer Data | 10 tests (C1-C10) | 🔥 **CRITICAL** |
| **[PRODUCTS_TRANSFORMATION_TESTS.md](PRODUCTS_TRANSFORMATION_TESTS.md)** | 📦 Product Data | 10 tests (P1-P10) | 💰 High |
| **[TRANSACTIONS_TRANSFORMATION_TESTS.md](TRANSACTIONS_TRANSFORMATION_TESTS.md)** | 💳 Transaction Data | 10 tests (T1-T10) | 📊 High |

### **Integration Testing**
| Guide File | Focus | Tests | Priority |
|------------|-------|-------|----------|
| **[MULTI_FILE_TRANSFORMATION_TESTS.md](MULTI_FILE_TRANSFORMATION_TESTS.md)** | 🔗 Multi-File Integration | 10 tests (M1-M10) | 🏢 Business |

### **Original Comprehensive Guide**
| Guide File | Focus | Tests | Priority |
|------------|-------|-------|----------|
| **[PROMPT_BASED_TRANSFORMATION_TESTING.md](PROMPT_BASED_TRANSFORMATION_TESTING.md)** | 🤖 Complete Suite | 15 tests | 📖 Reference |

---

# 🎯 **Quick Start Testing Sequence**

## 🚨 **Critical Path (Must Test First)**
1. **Test C1** - account_summary bug validation (CRITICAL)
2. **Test P1** - Mathematical calculations verification
3. **Test T1** - Financial discount calculations
4. **Test M1** - Multi-file integration proof

## 🏗️ **Foundation Testing (Core Functionality)**
- **Customer Tests**: C2, C3, C9, C10 (basic transformations)
- **Product Tests**: P2, P3 (categorization logic)
- **Transaction Tests**: T2, T4 (payment and type analysis)

## 🧮 **Advanced Testing (Business Logic)**
- **Customer Tests**: C4, C6, C7 (risk and demographics)
- **Product Tests**: P4, P7, P10 (business intelligence)
- **Transaction Tests**: T3, T5, T9 (performance analytics)
- **Multi-File Tests**: M2, M3, M4 (comprehensive analysis)

---

# 📊 **Test Coverage Matrix**

## **Features Tested Across All Guides**

### ✅ **Mathematical Operations**
- **Addition/Subtraction**: Profit margins (P1), Discount calculations (T1)
- **Multiplication/Division**: Markup percentages (P1), ROI calculations (P7)
- **Percentage Calculations**: Discounts (T1), Performance metrics (P7)

### ✅ **String Manipulation**
- **Concatenation**: Full names (C3), Product descriptions (P8)
- **Parsing**: Email domains (C5), Contact formatting (C8)
- **Formatting**: Names (C5), Addresses (C3), Phone numbers (C5)

### ✅ **Conditional Logic**
- **Single Conditions**: Customer tiers (C2), Stock status (P2)
- **Multiple Conditions**: Risk analysis (C4), Business priority (P3)
- **Numeric Ranges**: Price tiers (P3), Transaction sizes (T2)
- **String Equality**: Transaction types (T4), Payment methods (T2)

### ✅ **Data Integration**
- **File Relationships**: Customer-Transaction links (M1-M10)
- **Complex Joins**: 3-file analysis (M1), Business intelligence (M10)
- **Data Enrichment**: Adding calculated fields across files

### ✅ **Error Handling**
- **Missing Fields**: Graceful defaults (C9, Test 5A in original)
- **Special Characters**: Text parsing safety (C10, Test 5B in original)
- **Data Validation**: Type checking, range validation

---

# 🔧 **Setup Instructions**

## **Prerequisites**
1. **Start Backend Server**:
   ```bash
   cd backend
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Open Browser**: Navigate to `http://localhost:8000`

3. **Verify Environment**: Ensure OpenAI API key is configured in `.env`

## **Test Data Files**
Located in `/docs/testing/transformation/`:
- **customers_test.csv** (10 records) - Customer master data
- **products_test.csv** (11 records) - Product catalog  
- **transactions_test.csv** (12 records) - Transaction history

---

# 🎯 **Testing Methodology**

## **Individual File Testing**
1. **Upload single CSV file**
2. **Use AI Assistance with provided prompts**
3. **Validate expected results**
4. **Check for data accuracy and completeness**

## **Multi-File Integration Testing**
1. **Upload all required CSV files**
2. **Select multiple files in AI interface**
3. **Use integration prompts**
4. **Validate data relationships and joins**

## **Critical Bug Testing**
- **Focus on Test C1** (account_summary field generation)
- **Verify mathematical calculations** (Tests P1, T1)
- **Check string concatenation** (Tests C3, C5)
- **Validate conditional logic** (Tests C2, P2, T4)

---

# 📈 **Expected Results Summary**

## **Critical Success Metrics**
- ✅ **account_summary fields populated** (Not empty)
- ✅ **Mathematical calculations accurate** (Profit margins, discounts)
- ✅ **String concatenation working** (Names, descriptions)
- ✅ **Conditional logic functioning** (Tiers, categories, statuses)
- ✅ **Multi-file joins succeeding** (Proper data relationships)

## **Business Intelligence Validation**
- **Customer Insights**: Tier classification, risk analysis, demographics
- **Product Analytics**: Profitability, inventory status, market positioning
- **Transaction Intelligence**: Payment patterns, channel analysis, rep performance
- **Integration Analysis**: Comprehensive business dashboards

---

# 🚨 **Troubleshooting Quick Reference**

## **Common Issues & Solutions**

### **account_summary Empty (Critical)**
- **Check**: Static value expression syntax
- **Verify**: Column names match exactly (Account_Type, Balance)
- **Test**: Use Test C1 specifically for this issue

### **Mathematical Calculations Wrong**
- **Check**: Expression syntax for operations
- **Verify**: Data types (numeric vs string)
- **Test**: Use Tests P1, T1 for calculation validation

### **Multi-File Joins Failing**
- **Check**: All files uploaded successfully
- **Verify**: Common key fields exist (Customer_ID, Product_Code)
- **Test**: Start with Test M1 for basic integration

### **AI Configuration Issues**
- **Check**: OpenAI API key configuration
- **Verify**: Prompts are clear and specific
- **Test**: Start with simple prompts before complex ones

---

# 📝 **Testing Best Practices**

## **Test Execution Order**
1. **Start with individual file tests** before integration
2. **Test critical functionality first** (C1, P1, T1)
3. **Validate basic operations** before complex scenarios
4. **Use integration tests** to verify relationships

## **Result Validation**
- **Check record counts** match expectations
- **Verify data accuracy** with sample calculations
- **Validate business logic** makes sense
- **Test error handling** with edge cases

## **Documentation**
- **Record test results** for each scenario
- **Note any failures** with specific error details
- **Track performance** for large datasets
- **Document** any discovered issues or limitations

---

This comprehensive testing suite provides **40+ individual tests** covering all aspects of transformation functionality, from basic string operations to complex multi-file business intelligence scenarios!