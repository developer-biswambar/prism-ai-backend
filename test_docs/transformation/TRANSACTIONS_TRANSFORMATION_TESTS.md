# 💳 Transaction Data Transformation Testing Guide

## Overview
This guide focuses specifically on testing transformation scenarios using the **`transactions_test.csv`** file. All tests are designed for browser-based testing at `http://localhost:8000`.

## 📁 Test File
- **File**: `transactions_test.csv`
- **Records**: 12 transaction records
- **Columns**: Transaction_ID, Customer_ID, Product_Code, Product_Name, Category, Transaction_Date, Amount, Quantity, Unit_Price, Discount_Percent, Payment_Method, Channel, Sales_Rep, Transaction_Type, Currency

## 🚀 Prerequisites
1. Start backend server: `python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
2. Open browser: `http://localhost:8000`
3. Upload `transactions_test.csv` file

---

# 🧪 Transaction Transformation Tests

## 🎯 **Test T1: Discount and Financial Calculations**

### Steps:
1. Upload **`transactions_test.csv`**
2. Go to **AI Assistance**
3. Enter this prompt:

**🤖 PROMPT:**
```
Analyze transaction data by calculating final amounts after discounts, savings amounts, and financial summaries. Calculate: discounted_amount as (Amount * (1 - Discount_Percent/100)), savings_amount as (Amount - discounted_amount), and total_value assessment for each transaction.
```

### ✅ **Expected Results:**
- **discounted_amount**: $1299.99 (0% discount), $269.55 (10% discount), $85.49 (5% discount), etc.
- **savings_amount**: $0.00, $29.95, $4.50, $0.00, etc.
- **total_value**: High value (>$500), Medium value ($100-$500), Low value (<$100)
- **Total Records**: 12

---

## 🎯 **Test T2: Payment Method and Channel Analysis**

### Steps:
1. Use **`transactions_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Analyze payment preferences and sales channels. Create: payment_category grouping similar payment methods, channel_type distinguishing online vs physical, transaction_convenience_score based on payment and channel, and customer_experience_rating.
```

### ✅ **Expected Results:**
- **payment_category**: "Card Payment" (Credit/Debit), "Digital Payment" (PayPal), "Cash Payment"
- **channel_type**: "Online Channel", "Physical Store"
- **transaction_convenience_score**: Higher scores for online card payments
- **customer_experience_rating**: "Excellent", "Good", "Standard"

---

## 🎯 **Test T3: Sales Representative Performance**

### Steps:
1. Use **`transactions_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Analyze sales representative performance and productivity. Generate: rep_transaction_count for each representative, rep_total_sales summing amounts by rep, rep_performance_category based on sales volumes, and rep_efficiency_score considering transaction values.
```

### ✅ **Expected Results:**
- **rep_transaction_count**: REP001: 4 transactions, REP002: 2 transactions, etc.
- **rep_total_sales**: REP001: $3048.98, REP002: $299.50, etc.
- **rep_performance_category**: "Top Performer", "Above Average", "Average"
- **rep_efficiency_score**: Calculated based on transaction value and count

---

## 🎯 **Test T4: Transaction Type and Return Analysis**

### Steps:
1. Use **`transactions_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Analyze different transaction types including purchases and returns. Create: transaction_impact (Positive for purchases, Negative for returns), net_revenue considering transaction type, return_indicator for tracking, and customer_satisfaction_flag based on transaction patterns.
```

### ✅ **Expected Results:**
- **transaction_impact**: "Revenue Positive" for Purchase, "Revenue Negative" for Return
- **net_revenue**: Positive amounts for purchases, negative for returns
- **return_indicator**: "RETURN" for TXN011 (Office Chair return)
- **customer_satisfaction_flag**: "POTENTIAL_ISSUE" for returns, "SATISFIED" for purchases

---

## 🎯 **Test T5: Product Category and Sales Analysis**

### Steps:
1. Use **`transactions_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Analyze product category performance and sales patterns. Generate: category_performance based on transaction amounts, product_popularity using transaction frequency, category_average_sale calculating mean amounts per category, and market_segment_analysis.
```

### ✅ **Expected Results:**
- **category_performance**: "High Performing" (Electronics), "Medium Performing" (Furniture)
- **product_popularity**: "Best Seller", "Popular", "Standard", "Slow Moving"
- **category_average_sale**: Electronics: ~$750, Furniture: ~$300, etc.
- **market_segment_analysis**: "Premium Electronics", "Home Furnishing", "Daily Essentials"

---

## 🎯 **Test T6: Customer Transaction Behavior**

### Steps:
1. Use **`transactions_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Analyze customer transaction patterns and purchasing behavior. Create: customer_spending_pattern based on transaction amounts, purchase_frequency_indicator, preferred_payment_method by customer, and customer_loyalty_indicator based on transaction history.
```

### ✅ **Expected Results:**
- **customer_spending_pattern**: "High Spender" (>$1000), "Medium Spender" ($100-$1000), "Low Spender" (<$100)
- **purchase_frequency_indicator**: "Frequent Buyer", "Occasional Buyer", "One-time Buyer"
- **preferred_payment_method**: Most used payment method per customer
- **customer_loyalty_indicator**: "Loyal Customer", "Regular Customer", "New Customer"

---

## 🎯 **Test T7: Quantity and Unit Price Analysis**

### Steps:
1. Use **`transactions_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Analyze transaction quantities and unit pricing patterns. Generate: quantity_category (Single/Multiple based on quantity), unit_price_verification comparing unit price to total amount, bulk_purchase_indicator for multiple quantities, and pricing_efficiency_score.
```

### ✅ **Expected Results:**
- **quantity_category**: "Single Item" (qty=1), "Multiple Items" (qty>1)
- **unit_price_verification**: "Verified" when Unit_Price × Quantity = Amount
- **bulk_purchase_indicator**: "BULK" for quantity > 1, "SINGLE" for quantity = 1
- **pricing_efficiency_score**: Value assessment based on unit pricing

---

## 🎯 **Test T8: Transaction Timing and Patterns**

### Steps:
1. Use **`transactions_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Analyze transaction timing and temporal patterns. Create: transaction_period based on dates, seasonal_indicator for transaction timing, purchase_urgency based on patterns, and timing_efficiency_score for business insights.
```

### ✅ **Expected Results:**
- **transaction_period**: "Week 3 January", "Week 4 January", etc.
- **seasonal_indicator**: "Peak Season", "Regular Season", "Off Season"
- **purchase_urgency**: "Immediate", "Planned", "Impulse"
- **timing_efficiency_score**: Business timing analysis

---

## 🎯 **Test T9: Revenue and Profitability Analysis**

### Steps:
1. Use **`transactions_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Calculate comprehensive revenue and profitability metrics. Generate: gross_revenue from transaction amounts, net_revenue after discounts, revenue_per_transaction averages, and profitability_tier based on contribution margins.
```

### ✅ **Expected Results:**
- **gross_revenue**: Sum of original amounts before discounts
- **net_revenue**: Sum of amounts after applying discounts
- **revenue_per_transaction**: Average transaction value calculations
- **profitability_tier**: "High Profit", "Medium Profit", "Low Profit"

---

## 🎯 **Test T10: Currency and Regional Analysis**

### Steps:
1. Use **`transactions_test.csv`**
2. Enter this prompt:

**🤖 PROMPT:**
```
Analyze currency handling and regional transaction patterns. Create: currency_consistency checking all transactions, regional_sales_pattern based on customer regions, international_indicator for currency analysis, and market_penetration_score.
```

### ✅ **Expected Results:**
- **currency_consistency**: "USD Standard" (all transactions in USD)
- **regional_sales_pattern**: Sales distribution across regions
- **international_indicator**: "Domestic Only" (all USD transactions)
- **market_penetration_score**: Regional market analysis

---

# 📊 **Transaction Test Summary**

| Test | Focus Area | Expected Records | Key Calculations |
|------|------------|------------------|------------------|
| **T1** | Financial Calculations | 12 | Discounts, savings amounts |
| **T2** | Payment & Channel | 12 | Payment method grouping |
| **T3** | Sales Rep Performance | 12 | Rep productivity metrics |
| **T4** | Transaction Types | 12 | Purchase vs Return analysis |
| **T5** | Category Analysis | 12 | Product category performance |
| **T6** | Customer Behavior | 12 | Spending pattern analysis |
| **T7** | Quantity Analysis | 12 | Unit price verification |
| **T8** | Timing Patterns | 12 | Temporal analysis |
| **T9** | Revenue Analysis | 12 | Profitability calculations |
| **T10** | Currency & Regional | 12 | Market analysis |

## 🎯 **Financial Calculations Validation**

### ✅ **Key Discount Calculations:**
- **TXN001**: $1299.99 × (1 - 0/100) = $1299.99 (No discount)
- **TXN002**: $299.50 × (1 - 10/100) = $269.55 (10% discount, saves $29.95)
- **TXN003**: $89.99 × (1 - 5/100) = $85.49 (5% discount, saves $4.50)
- **TXN004**: $899.00 × (1 - 0/100) = $899.00 (No discount)

### ✅ **Sales Rep Performance:**
- **REP001**: 4 transactions (TXN001, TXN004, TXN007, TXN010) = $3,048.98 total
- **REP002**: 2 transactions (TXN002, TXN011) = $299.50 total
- **REP003**: 1 transaction (TXN003) = $89.99 total

## 📈 **Transaction Types Breakdown**
- **Purchases**: 11 transactions (TXN001-TXN010, TXN012)
- **Returns**: 1 transaction (TXN011 - Office Chair return)
- **Categories**: Electronics (6), Furniture (2), Appliances (1), Accessories (1), Stationery (1), Office (1)

## 💳 **Payment Methods Distribution**
- **Credit Card**: 6 transactions
- **Debit Card**: 3 transactions  
- **Cash**: 2 transactions
- **PayPal**: 1 transaction

## 🏪 **Channel Distribution**
- **Online**: 7 transactions
- **Store**: 5 transactions

## 🎯 **Testing Priority**
1. **💰 FINANCIAL**: Test T1 (discount calculations) - Critical for accuracy
2. **🔄 RETURNS**: Test T4 (transaction types) - Important for business logic
3. **👨‍💼 PERFORMANCE**: Test T3 (sales rep analysis) - Business insights
4. **💳 PAYMENTS**: Test T2 (payment analysis) - Customer experience
5. **📊 ANALYTICS**: Tests T5, T6, T9 (business intelligence)
6. **⏰ PATTERNS**: Tests T7, T8, T10 (operational insights)

This guide provides comprehensive testing of all transaction-related transformation scenarios with focus on financial calculations, business analytics, and operational intelligence!