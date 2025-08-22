# DynamoDB Rules Table Documentation

## Table Overview

This document describes the DynamoDB table structure for storing Delta Rules and Rule Management data in a single table design.

## Table Configuration

- **Table Name**: `Rules` (configurable via `DYNAMODB_RULES_TABLE` environment variable)
- **Billing Mode**: On-Demand (recommended) or Provisioned
- **Region**: Configurable via `AWS_REGION` environment variable

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

## Global Secondary Indexes (GSI)

### GSI1 - Template-based Queries
- **Purpose**: Query rules by template ID
- **GSI1PK (Partition Key)**: `template_id`
- **GSI1SK (Sort Key)**: `updated_at` (for sorting by recency)
- **Projection**: All attributes

### GSI2 - Category and Usage Queries
- **Purpose**: Query rules by category and sort by usage/recency
- **GSI2PK (Partition Key)**: `category`
- **GSI2SK (Sort Key)**: `{usage_count:08d}#{updated_at}` (padded for lexicographic sorting)
- **Projection**: All attributes

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

### GSI Attributes

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `GSI1PK` | String | Template ID (only if template_id exists) | `template_xyz` |
| `GSI1SK` | String | Updated timestamp for sorting | `2024-01-15T10:30:00Z` |
| `GSI2PK` | String | Category | `financial` |
| `GSI2SK` | String | Usage count (padded) + timestamp | `00000005#2024-01-15T10:30:00Z` |

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

### 2. List Rules by Type
- **Pattern**: Query by rule_type (requires scan with filter)
- **Filter**: `rule_type = 'delta'`
- **Use Case**: List all delta rules

### 3. List Rules by Template
- **Pattern**: Query GSI1
- **Keys**: `GSI1PK = {template_id}`
- **Sort**: By `GSI1SK` (updated_at) descending
- **Use Case**: Get all rules for a template

### 4. List Rules by Category
- **Pattern**: Query GSI2
- **Keys**: `GSI2PK = {category}`
- **Sort**: By `GSI2SK` (usage + recency) descending
- **Use Case**: Get rules by category, sorted by popularity

### 5. Search Rules
- **Pattern**: Scan with filters
- **Filters**: Multiple attribute filters
- **Use Case**: Complex search with multiple criteria

## DynamoDB Table Creation

### AWS CLI Command
```bash
aws dynamodb create-table \
  --table-name Rules \
  --attribute-definitions \
    AttributeName=PK,AttributeType=S \
    AttributeName=SK,AttributeType=S \
    AttributeName=GSI1PK,AttributeType=S \
    AttributeName=GSI1SK,AttributeType=S \
    AttributeName=GSI2PK,AttributeType=S \
    AttributeName=GSI2SK,AttributeType=S \
  --key-schema \
    AttributeName=PK,KeyType=HASH \
    AttributeName=SK,KeyType=RANGE \
  --global-secondary-indexes \
    '[
      {
        "IndexName": "GSI1",
        "KeySchema": [
          {"AttributeName": "GSI1PK", "KeyType": "HASH"},
          {"AttributeName": "GSI1SK", "KeyType": "RANGE"}
        ],
        "Projection": {"ProjectionType": "ALL"},
        "BillingMode": "PAY_PER_REQUEST"
      },
      {
        "IndexName": "GSI2", 
        "KeySchema": [
          {"AttributeName": "GSI2PK", "KeyType": "HASH"},
          {"AttributeName": "GSI2SK", "KeyType": "RANGE"}
        ],
        "Projection": {"ProjectionType": "ALL"},
        "BillingMode": "PAY_PER_REQUEST"
      }
    ]' \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

### Terraform Configuration
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

  attribute {
    name = "GSI1PK"
    type = "S"
  }

  attribute {
    name = "GSI1SK"
    type = "S"
  }

  attribute {
    name = "GSI2PK"
    type = "S"
  }

  attribute {
    name = "GSI2SK"
    type = "S"
  }

  global_secondary_index {
    name     = "GSI1"
    hash_key = "GSI1PK"
    range_key = "GSI1SK"
    projection_type = "ALL"
  }

  global_secondary_index {
    name     = "GSI2"
    hash_key = "GSI2PK"
    range_key = "GSI2SK"
    projection_type = "ALL"
  }

  tags = {
    Name        = "Rules"
    Environment = "production"
    Purpose     = "Delta and Rule Management"
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
        "dynamodb:Query",
        "dynamodb:Scan",
        "dynamodb:DescribeTable"
      ],
      "Resource": [
        "arn:aws:dynamodb:*:*:table/Rules",
        "arn:aws:dynamodb:*:*:table/Rules/index/*"
      ]
    }
  ]
}
```

## Performance Considerations

1. **Hot Partitions**: Distribute rules across different categories to avoid hot partitions
2. **Query Efficiency**: Use GSIs for common query patterns instead of scans
3. **Item Size**: Rule configurations should stay under 400KB DynamoDB limit
4. **Pagination**: Implement pagination for large result sets
5. **Eventual Consistency**: GSI queries are eventually consistent

## Monitoring and Metrics

- **CloudWatch Metrics**: Monitor read/write capacity, throttling, errors
- **Application Metrics**: Track rule usage patterns, popular categories
- **Cost Monitoring**: Monitor DynamoDB costs, especially for scans
- **Performance**: Track query latency and success rates