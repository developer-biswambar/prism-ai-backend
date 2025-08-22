# DynamoDB Rules Table Documentation - Simplified Schema

## Table Overview

This document describes the simplified DynamoDB table structure for storing Delta Rules and Rule Management data using only primary keys with full table scans for queries.

## Table Configuration

- **Table Name**: `Rules` (configurable via `DYNAMODB_RULES_TABLE` environment variable)
- **Billing Mode**: On-Demand (recommended) or Provisioned
- **Region**: Configurable via `AWS_REGION` environment variable
- **Design Philosophy**: Minimal schema with only primary keys, using full scans for flexibility

## Primary Key Structure

### Partition Key (PK)
- **Type**: String
- **Format**: `{RULE_TYPE}#{RULE_ID}`
- **Examples**:
  - `DELTA_RULE#delta_rule_12345`
  - `RECONCILIATION_RULE#rule_67890`
  - `TRANSFORMATION_RULE#tr_rule_54321`

### Sort Key (SK)
- **Type**: String
- **Format**: `{RULE_ID}`
- **Examples**:
  - `delta_rule_12345`
  - `rule_67890`
  - `tr_rule_54321`

## Attributes Schema

### Core Attributes (All Rule Types)

| Attribute | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `PK` | String | Yes | Partition key | `DELTA_RULE#delta_rule_123` |
| `SK` | String | Yes | Sort key | `delta_rule_123` |
| `id` | String | Yes | Unique rule identifier | `delta_rule_123` |
| `rule_type` | String | Yes | Type of rule | `delta`, `reconciliation`, `transformation` |
| `name` | String | Yes | Display name | `Monthly Delta Analysis` |
| `description` | String | No | Optional description | `Compares monthly financial data` |
| `category` | String | Yes | Rule category | `financial`, `delta`, `general` |
| `tags` | StringSet | No | List of tags | `["monthly", "financial", "automated"]` |
| `template_id` | String | No | Associated template ID | `template_xyz` |
| `template_name` | String | No | Template display name | `Financial Template` |
| `created_at` | String | Yes | ISO timestamp | `2024-01-15T10:30:00Z` |
| `updated_at` | String | Yes | ISO timestamp | `2024-01-15T10:30:00Z` |
| `version` | String | Yes | Rule version | `1.0` |
| `rule_config` | Map | Yes | Rule configuration (JSON) | See below |
| `usage_count` | Number | Yes | Times rule has been used | `5` |
| `last_used_at` | String | No | Last usage timestamp | `2024-01-14T15:20:00Z` |

## Rule Configuration Examples

### Delta Rule Configuration
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
  "KeyRules": [
    {
      "LeftFileColumn": "transaction_id",
      "RightFileColumn": "ref_id",
      "MatchType": "equals"
    }
  ],
  "ComparisonRules": [
    {
      "LeftFileColumn": "amount", 
      "RightFileColumn": "value",
      "MatchType": "tolerance",
      "ToleranceValue": 0.01
    }
  ],
  "selected_columns_file_a": ["transaction_id", "amount", "date"],
  "selected_columns_file_b": ["ref_id", "value", "transaction_date"],
  "user_requirements": "Generate delta between older and newer files"
}
```

### Reconciliation Rule Configuration
```json
{
  "Files": [
    {
      "Name": "FileA",
      "Extract": [
        {
          "ResultColumnName": "clean_amount",
          "SourceColumn": "amount_text", 
          "MatchType": "regex",
          "Patterns": ["\\d+\\.\\d{2}"]
        }
      ],
      "Filter": [
        {
          "ColumnName": "status",
          "MatchType": "equals",
          "Value": "active"
        }
      ]
    }
  ],
  "ReconciliationRules": [
    {
      "LeftFileColumn": "transaction_id",
      "RightFileColumn": "ref_number",
      "MatchType": "equals",
      "ToleranceValue": 0
    }
  ],
  "example_columns_file_a": ["transaction_id", "amount_text", "status"],
  "example_columns_file_b": ["ref_number", "amount", "date"],
  "user_requirements": "Match transactions by ID with amount validation"
}
```

### Transformation Rule Configuration
```json
{
  "example_name": "Financial Data Transformation",
  "example_description": "Transform and merge financial datasets",
  "source_files": [
    {
      "file_id": "source_file_0",
      "alias": "source_file_0", 
      "purpose": "Primary financial data"
    }
  ],
  "row_generation_rules": [
    {
      "condition": "amount > 1000",
      "output_columns": {
        "high_value": "true",
        "category": "large_transaction"
      }
    }
  ],
  "merge_datasets": true,
  "validation_rules": [
    {
      "column": "amount",
      "type": "numeric",
      "required": true
    }
  ],
  "user_requirements": "Transform and validate financial data"
}
```

## Access Patterns

### 1. Get Specific Rule
- **Pattern**: Get item by PK + SK
- **Keys**: `PK = DELTA_RULE#{rule_id}`, `SK = {rule_id}`
- **Use Case**: Load specific rule by ID
- **Performance**: O(1) lookup

### 2. List Rules by Type
- **Pattern**: Scan with filter
- **Filter**: `rule_type = 'delta'`
- **Use Case**: List all delta rules
- **Performance**: O(n) scan, acceptable for moderate data sizes

### 3. List Rules by Category
- **Pattern**: Scan with filter
- **Filter**: `rule_type = 'delta' AND category = 'financial'`
- **Use Case**: Get rules by type and category
- **Performance**: O(n) scan, post-filtered by application

### 4. List Rules by Template
- **Pattern**: Scan with filter
- **Filter**: `rule_type = 'delta' AND template_id = 'template_xyz'`
- **Use Case**: Get all rules for a template
- **Performance**: O(n) scan, application-level sorting

### 5. Search Rules with Multiple Criteria
- **Pattern**: Scan with multiple filters
- **Filters**: `rule_type = 'delta' AND category = 'financial'`
- **Use Case**: Complex search with multiple criteria
- **Performance**: O(n) scan, client-side filtering for complex criteria

## DynamoDB Table Creation

### AWS CLI Command (Simplified)
```bash
aws dynamodb create-table \
  --table-name Rules \
  --attribute-definitions \
    AttributeName=PK,AttributeType=S \
    AttributeName=SK,AttributeType=S \
  --key-schema \
    AttributeName=PK,KeyType=HASH \
    AttributeName=SK,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

### Terraform Configuration (Simplified)
```hcl
resource "aws_dynamodb_table" "rules" {
  name           = "Rules"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "PK"
  range_key      = "SK"

  attribute {
    name = "PK"
    type = "S"
  }

  attribute {
    name = "SK"
    type = "S"
  }

  tags = {
    Name        = "Rules"
    Environment = "production"
    Purpose     = "Delta and Rule Management - Simplified"
  }
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DYNAMODB_RULES_TABLE` | `Rules` | DynamoDB table name |
| `AWS_REGION` | `us-east-1` | AWS region for DynamoDB |
| `AWS_ACCESS_KEY_ID` | - | AWS access key (or use IAM roles) |
| `AWS_SECRET_ACCESS_KEY` | - | AWS secret key (or use IAM roles) |

## IAM Permissions Required

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem", 
        "dynamodb:UpdateItem",
        "dynamodb:DeleteItem",
        "dynamodb:Scan",
        "dynamodb:DescribeTable"
      ],
      "Resource": [
        "arn:aws:dynamodb:*:*:table/Rules"
      ]
    }
  ]
}
```

Note: No Query permission needed since we only use Scan operations.

## Performance Considerations

### Advantages of Simplified Schema
1. **No GSI Costs**: Eliminates additional read/write costs for Global Secondary Indexes
2. **Schema Flexibility**: Easy to add new attributes without GSI maintenance
3. **Simpler Operations**: No complex GSI key management or eventual consistency concerns
4. **Lower Complexity**: Reduced operational overhead and easier debugging

### Performance Trade-offs
1. **Scan Operations**: All queries use table scans instead of efficient queries
2. **Higher Read Costs**: Full table scans consume more read capacity
3. **Latency**: Scan operations have higher latency than key-based queries
4. **Filtering**: Application-level filtering required for complex searches

### Optimization Strategies
1. **Client-side Caching**: Cache frequently accessed rules to reduce scans
2. **Batch Operations**: Process multiple rules in single scan operations
3. **Pagination**: Implement proper pagination to limit scan results
4. **Data Size Management**: Keep rule configurations compact to improve scan speed
5. **Application-level Indexing**: Maintain in-memory indexes for frequently queried attributes

## When to Use This Schema

### Ideal Use Cases
- **Small to Medium Data Sets**: Rule count < 10,000 items
- **Infrequent Queries**: Rules accessed occasionally, not real-time
- **Cost Optimization**: When GSI costs outweigh query performance benefits
- **Flexible Schema**: When rule attributes change frequently
- **Development/Testing**: Simplified setup for non-production environments

### Consider GSI Schema When
- **Large Data Sets**: Rule count > 10,000 items
- **High Query Frequency**: Rules accessed frequently in production
- **Performance Critical**: Sub-second query response times required
- **Complex Query Patterns**: Multiple query access patterns with high performance needs

## Monitoring and Metrics

### CloudWatch Metrics to Monitor
- **ConsumedReadCapacityUnits**: Monitor scan operation costs
- **UserErrors**: Track scan failures and throttling
- **SuccessfulRequestLatency**: Monitor scan performance
- **ItemCount**: Track table growth

### Application Metrics
- **Scan Duration**: Track how long rule listing operations take
- **Cache Hit Rate**: Monitor effectiveness of client-side caching
- **Rule Usage Patterns**: Identify most frequently accessed rules
- **Data Growth**: Monitor rule count and configuration size

### Cost Optimization
- **Scan Frequency**: Minimize unnecessary scan operations
- **Result Filtering**: Filter results early to reduce data transfer
- **Caching Strategy**: Implement aggressive caching for read-heavy workloads
- **Archive Old Rules**: Move unused rules to reduce scan overhead