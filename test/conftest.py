# test/conftest.py
from typing import Dict, Any

import pytest

# Import your FastAPI routers based on your project structure

# Import routers safely with correct paths
try:
    from app.routes.file_routes import router as file_router

    HAS_FILE_ROUTES = True
except ImportError as e:
    print(f"Warning: Could not import file_routes: {e}")
    file_router = None
    HAS_FILE_ROUTES = False

try:
    from app.routes.viewer_routes import router as viewer_router

    HAS_VIEWER_ROUTES = True
except ImportError as e:
    print(f"Warning: Could not import viewer_routes: {e}")
    viewer_router = None
    HAS_VIEWER_ROUTES = False

try:
    from app.routes.reconciliation_routes import router as reconciliation_router

    HAS_RECONCILIATION_ROUTES = True
except ImportError as e:
    print(f"Warning: Could not import reconciliation_routes: {e}")
    reconciliation_router = None
    HAS_RECONCILIATION_ROUTES = False

# Mock storage for testing
test_uploaded_files: Dict[str, Dict[str, Any]] = {}

# Reconciliation test data constants
RECONCILIATION_TEST_FILE_A = """Transaction_ID,Date,Description,Amount_Text,Status,Account,Reference,Customer_ID,Branch
TXN001,2024-01-15,Payment to Vendor ABC,$1,234.56,Settled,ACC001,REF123,CUST001,BR01
TXN002,2024-01-16,Salary Payment,$2,500.00,Pending,ACC002,REF124,CUST002,BR02
TXN003,2024-01-17,Refund Processing,$-456.78,Completed,ACC003,REF125,CUST003,BR01
TXN004,2024-01-18,Investment Transfer,$10,000.00,Settled,ACC001,REF126,CUST001,BR03
TXN005,2024-01-19,Utility Bill Payment,$234.50,Failed,ACC004,REF127,CUST004,BR02
TXN006,2024-01-20,Customer Deposit,$3,456.78,Settled,ACC002,REF128,CUST002,BR01
TXN007,2024-01-21,Loan Disbursement,$15,000.00,Pending,ACC005,REF129,CUST005,BR03
TXN008,2024-01-22,Card Payment,$89.99,Settled,ACC003,REF130,CUST003,BR02
TXN009,2024-01-23,Wire Transfer,$5,678.90,Completed,ACC001,REF131,CUST006,BR01
TXN010,2024-01-24,ATM Withdrawal,$-200.00,Settled,ACC004,REF132,CUST004,BR02
TXN011,2024-01-25,Online Purchase,$125.75,Settled,ACC002,REF133,CUST002,BR03
TXN012,2024-01-26,Insurance Premium,$567.89,Pending,ACC006,REF134,CUST007,BR01
TXN013,2024-01-27,Dividend Payment,$1,000.00,Settled,ACC001,REF135,CUST001,BR02
TXN014,2024-01-28,Merchant Payment,$345.67,Completed,ACC003,REF136,CUST003,BR03
TXN015,2024-01-29,Subscription Fee,$29.99,Settled,ACC004,REF137,CUST004,BR01
TXN016,2024-01-30,International Transfer,$2,345.67,Pending,ACC005,REF138,CUST008,BR02
TXN017,2024-02-01,Cash Deposit,$500.00,Settled,ACC002,REF139,CUST002,BR01
TXN018,2024-02-02,Bill Payment,$78.45,Completed,ACC006,REF140,CUST007,BR03
TXN019,2024-02-03,Investment Return,$1,567.89,Settled,ACC001,REF141,CUST001,BR02
TXN020,2024-02-04,Service Charge,$-25.00,Settled,ACC003,REF142,CUST003,BR01"""

RECONCILIATION_TEST_FILE_B = """Statement_ID,Process_Date,Transaction_Desc,Net_Amount,Settlement_Status,Account_Number,Ref_Number,Client_Code,Location
STMT001,15/01/2024,Vendor ABC Payment,1234.56,SETTLED,ACC001,REF123,CUST001,BR01
STMT002,16/01/2024,Employee Salary,2500.01,PROCESSING,ACC002,REF124,CUST002,BR02
STMT003,17/01/2024,Customer Refund,-456.78,COMPLETE,ACC003,REF125,CUST003,BR01
STMT004,18/01/2024,Investment Xfer,10000.00,SETTLED,ACC001,REF126,CUST001,BR03
STMT005,19/01/2024,Utility Payment,234.50,REJECTED,ACC004,REF127,CUST004,BR02
STMT006,20/01/2024,Deposit from Customer,3456.78,SETTLED,ACC002,REF128,CUST002,BR01
STMT007,21/01/2024,Loan Payment,15000.00,PROCESSING,ACC005,REF129,CUST005,BR03
STMT008,22/01/2024,Credit Card Transaction,89.99,SETTLED,ACC003,REF130,CUST003,BR02
STMT009,23/01/2024,Wire Transfer Out,5678.90,COMPLETE,ACC001,REF131,CUST006,BR01
STMT010,24/01/2024,ATM Cash Withdrawal,-200.00,SETTLED,ACC004,REF132,CUST004,BR02
STMT011,25/01/2024,E-commerce Purchase,125.75,SETTLED,ACC002,REF133,CUST002,BR03
STMT012,26/01/2024,Insurance Payment,567.89,PROCESSING,ACC006,REF134,CUST007,BR01
STMT013,27/01/2024,Dividend Distribution,1000.00,SETTLED,ACC001,REF135,CUST001,BR02
STMT014,28/01/2024,POS Transaction,345.67,COMPLETE,ACC003,REF136,CUST003,BR03
STMT015,29/01/2024,Monthly Subscription,29.99,SETTLED,ACC004,REF137,CUST004,BR01
STMT016,30/01/2024,Foreign Exchange,2345.68,PROCESSING,ACC005,REF138,CUST008,BR02
STMT017,01/02/2024,Cash Deposit Transaction,500.00,SETTLED,ACC002,REF139,CUST002,BR01
STMT018,02/02/2024,Utility Bill Settlement,78.45,COMPLETE,ACC006,REF140,CUST007,BR03
STMT019,03/02/2024,Investment Profit,1567.89,SETTLED,ACC001,REF141,CUST001,BR02
STMT020,04/02/2024,Bank Service Fee,-25.00,SETTLED,ACC003,REF142,CUST003,BR01
STMT021,05/02/2024,Unmatched Transaction,999.99,SETTLED,ACC007,REF999,CUST999,BR04
STMT022,06/02/2024,Another Unmatched,777.77,COMPLETE,ACC008,REF888,CUST888,BR05"""


@pytest.fixture(scope="session")
def app():
    """Create FastAPI app instance for testing"""
    test_app = FastAPI(title="Test App")

    if HAS_FILE_ROUTES and file_router:
        test_app.include_router(file_router)

    if HAS_VIEWER_ROUTES and viewer_router:
        test_app.include_router(viewer_router)

    if HAS_RECONCILIATION_ROUTES and reconciliation_router:
        test_app.include_router(reconciliation_router)

    return test_app


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_storage():
    """Mock the storage service for all tests"""
    test_uploaded_files.clear()  # Clear storage before each test

    # Patch where uploaded_files is actually used (in the route files)
    patches = []

    try:
        patches.append(patch('app.routes.file_routes.uploaded_files', test_uploaded_files))
    except ImportError:
        pass

    try:
        patches.append(patch('app.routes.viewer_routes.uploaded_files', test_uploaded_files))
    except ImportError:
        pass

    try:
        patches.append(patch('app.services.storage_service.uploaded_files', test_uploaded_files))
    except ImportError:
        pass

    # Apply all patches
    if patches:
        with patches[0] if len(patches) == 1 else contextlib.ExitStack() as stack:
            if len(patches) > 1:
                for patch_obj in patches:
                    stack.enter_context(patch_obj)
            yield test_uploaded_files
    else:
        # Fallback
        yield test_uploaded_files


@pytest.fixture
def sample_csv_content():
    """Create sample CSV content"""
    return """name,age,city,salary
John Doe,25,New York,50000
Jane Smith,30,Los Angeles,60000
Bob Johnson,35,Chicago,55000
Alice Brown,28,Houston,52000
Charlie Wilson,32,Phoenix,58000"""


@pytest.fixture
def sample_csv_file(sample_csv_content):
    """Create sample CSV file object"""
    return io.BytesIO(sample_csv_content.encode('utf-8'))


@pytest.fixture
def large_csv_content():
    """Create large CSV content for testing file size limits"""
    header = "id,name,value,description\n"
    rows = []
    for i in range(10000):  # 10k rows
        rows.append(f"{i},Name_{i},{i * 100},Description for item {i}")
    return header + "\n".join(rows)


@pytest.fixture
def large_csv_file(large_csv_content):
    """Create large CSV file object"""
    return io.BytesIO(large_csv_content.encode('utf-8'))


@pytest.fixture
def sample_excel_file():
    """Create sample Excel file with multiple sheets"""
    # Create Excel file in memory
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Sales data
        sales_data = pd.DataFrame({
            'product': ['Product A', 'Product B', 'Product C', 'Product D'],
            'sales': [1000, 1500, 800, 1200],
            'region': ['North', 'South', 'East', 'West']
        })
        sales_data.to_excel(writer, sheet_name='Sales', index=False)

        # Sheet 2: Employee data
        employee_data = pd.DataFrame({
            'employee_id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'department': ['HR', 'IT', 'Finance', 'IT', 'HR'],
            'salary': [55000, 65000, 60000, 70000, 58000]
        })
        employee_data.to_excel(writer, sheet_name='Employees', index=False)

        # Sheet 3: Empty sheet
        empty_data = pd.DataFrame()
        empty_data.to_excel(writer, sheet_name='Empty', index=False)

    output.seek(0)
    return output


@pytest.fixture
def invalid_file():
    """Create invalid file (not CSV or Excel)"""
    return io.BytesIO(b"This is not a valid CSV or Excel file content")


@pytest.fixture
def corrupted_excel_file():
    """Create corrupted Excel file"""
    return io.BytesIO(b"PK\x03\x04\x14\x00corrupted excel content")


@pytest.fixture
def empty_csv_file():
    """Create empty CSV file"""
    return io.BytesIO(b"")


@pytest.fixture
def csv_with_special_chars():
    """Create CSV with special characters and edge cases"""
    content = """name,description,value
"John, Jr.",Product with "quotes",100.50
Jane's Item,"Line 1
Line 2",200
Special Chars,Café & Résumé,300.75"""
    return io.BytesIO(content.encode('utf-8'))


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    env_vars = {
        'MAX_FILE_SIZE': '10',  # 10MB for testing
        'MAX_SAMPLE_ROWS': '50',
        'MAX_PREVIEW_ROWS': '100',
        'LARGE_FILE_THRESHOLD': '1000'
    }

    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def sample_file_data():
    """Create sample file data for testing responses"""
    return {
        "file_id": "test-file-id-123",
        "filename": "test.csv",
        "custom_name": None,
        "file_type": "csv",
        "file_size_mb": 0.001,
        "total_rows": 5,
        "total_columns": 4,
        "columns": ["name", "age", "city", "salary"],
        "upload_time": "2025-07-15T10:00:00",
        "data_types": {
            "name": "object",
            "age": "int64",
            "city": "object",
            "salary": "int64"
        },
        "sheet_name": None
    }


@pytest.fixture
def uploaded_file_with_data(sample_file_data, sample_csv_content):
    """Create uploaded file entry with data"""
    df = pd.read_csv(io.StringIO(sample_csv_content))
    file_entry = {
        "info": sample_file_data,
        "data": df
    }
    test_uploaded_files[sample_file_data["file_id"]] = file_entry
    return file_entry


# === RECONCILIATION-SPECIFIC FIXTURES ===

@pytest.fixture(scope="session")
def reconciliation_test_data():
    """Provide reconciliation test data for all tests"""
    return {
        "file_a_csv": RECONCILIATION_TEST_FILE_A,
        "file_b_csv": RECONCILIATION_TEST_FILE_B,
        "file_a_df": pd.read_csv(io.StringIO(RECONCILIATION_TEST_FILE_A)),
        "file_b_df": pd.read_csv(io.StringIO(RECONCILIATION_TEST_FILE_B))
    }


@pytest.fixture
def temp_reconciliation_files(reconciliation_test_data):
    """Create temporary CSV files for reconciliation testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_a_path = os.path.join(temp_dir, "financial_transactions.csv")
        file_b_path = os.path.join(temp_dir, "bank_statements.csv")

        with open(file_a_path, 'w') as f:
            f.write(reconciliation_test_data["file_a_csv"])

        with open(file_b_path, 'w') as f:
            f.write(reconciliation_test_data["file_b_csv"])

        yield {
            "file_a_path": file_a_path,
            "file_b_path": file_b_path,
            "temp_dir": temp_dir,
            "file_a_name": "financial_transactions.csv",
            "file_b_name": "bank_statements.csv"
        }


@pytest.fixture
def reconciliation_test_files_as_bytesio(reconciliation_test_data):
    """Create reconciliation test files as BytesIO objects for file uploads"""
    return {
        "file_a": io.BytesIO(reconciliation_test_data["file_a_csv"].encode('utf-8')),
        "file_b": io.BytesIO(reconciliation_test_data["file_b_csv"].encode('utf-8'))
    }


@pytest.fixture
def mock_reconciliation_api():
    """Mock reconciliation API client for testing"""
    client = MagicMock()

    # Mock file upload responses
    client.upload_file.side_effect = lambda x: {
        "file_id": f"test_file_{hash(x) % 1000:03d}",
        "filename": os.path.basename(x),
        "status": "success",
        "total_rows": 20 if "financial" in x else 22,
        "columns": ["Transaction_ID", "Date", "Amount_Text"] if "financial" in x else ["Statement_ID", "Process_Date",
                                                                                       "Net_Amount"]
    }

    # Mock column unique values
    client.get_column_unique_values.return_value = {
        "unique_values": ["Settled", "Pending", "Completed", "Failed"],
        "total_unique": 4,
        "is_date_column": False
    }

    # Mock reconciliation processing
    client.process_reconciliation.return_value = {
        "reconciliation_id": "recon_001",
        "status": "completed",
        "summary": {
            "total_file_a": 20,
            "total_file_b": 22,
            "matched_pairs": 20,
            "unmatched_a_count": 0,
            "unmatched_b_count": 2,
            "match_percentage": 90.9
        },
        "matched_records": [],
        "unmatched_records": []
    }

    return client


@pytest.fixture
def reconciliation_configs():
    """Provide various reconciliation configurations for testing"""
    return {
        "basic_reference_match": {
            "Files": [
                {"Name": "FileA", "Extract": [], "Filter": []},
                {"Name": "FileB", "Extract": [], "Filter": []}
            ],
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Reference",
                    "RightFileColumn": "Ref_Number",
                    "MatchType": "equals"
                }
            ]
        },

        "amount_extraction_with_tolerance": {
            "Files": [
                {
                    "Name": "FileA",
                    "Extract": [
                        {
                            "ResultColumnName": "Extracted_Amount",
                            "SourceColumn": "Amount_Text",
                            "MatchType": "regex",
                            "Patterns": ["\\$([0-9,.-]+)"]
                        }
                    ],
                    "Filter": []
                },
                {"Name": "FileB", "Extract": [], "Filter": []}
            ],
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Reference",
                    "RightFileColumn": "Ref_Number",
                    "MatchType": "equals"
                },
                {
                    "LeftFileColumn": "Extracted_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.01
                }
            ]
        },

        "status_filtered_reconciliation": {
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
                    "MatchType": "equals"
                }
            ]
        },

        "multi_column_complex_match": {
            "Files": [
                {
                    "Name": "FileA",
                    "Extract": [
                        {
                            "ResultColumnName": "Clean_Amount",
                            "SourceColumn": "Amount_Text",
                            "MatchType": "regex",
                            "Patterns": ["\\$([0-9,.-]+)"]
                        }
                    ],
                    "Filter": [
                        {
                            "ColumnName": "Status",
                            "MatchType": "equals",
                            "Value": "Settled"
                        },
                        {
                            "ColumnName": "Status",
                            "MatchType": "equals",
                            "Value": "Completed"
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
                        },
                        {
                            "ColumnName": "Settlement_Status",
                            "MatchType": "equals",
                            "Value": "COMPLETE"
                        }
                    ]
                }
            ],
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Reference",
                    "RightFileColumn": "Ref_Number",
                    "MatchType": "equals"
                },
                {
                    "LeftFileColumn": "Account",
                    "RightFileColumn": "Account_Number",
                    "MatchType": "equals"
                },
                {
                    "LeftFileColumn": "Customer_ID",
                    "RightFileColumn": "Client_Code",
                    "MatchType": "equals"
                },
                {
                    "LeftFileColumn": "Clean_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.05
                }
            ]
        }
    }


@pytest.fixture
def mock_delta_processor():
    """Mock DeltaProcessor for testing reconciliation logic"""
    with patch('app.services.reconciliation_service.DeltaProcessor') as mock_processor_class:
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Setup default mock behaviors
        mock_processor.extract_data.return_value = pd.DataFrame()
        mock_processor.apply_filters.return_value = pd.DataFrame()
        mock_processor.process_reconciliation.return_value = {
            "matched_records": [],
            "unmatched_a": [],
            "unmatched_b": [],
            "summary": {
                "matched_pairs": 0,
                "total_file_a": 0,
                "total_file_b": 0,
                "match_percentage": 0.0
            }
        }
        mock_processor.parse_excel_date.return_value = None

        yield mock_processor


@pytest.fixture
def reconciliation_expected_results():
    """Expected results for reconciliation test scenarios"""
    return {
        "perfect_match_scenario": {
            "summary": {
                "total_file_a": 20,
                "total_file_b": 22,
                "matched_pairs": 20,
                "unmatched_a_count": 0,
                "unmatched_b_count": 2,
                "match_percentage": 90.9
            },
            "matched_count": 20,
            "unmatched_refs": ["REF999", "REF888"]
        },

        "tolerance_match_scenario": {
            "tolerance_matches": [
                {
                    "ref": "REF124",
                    "amount_a": 2500.00,
                    "amount_b": 2500.01,
                    "difference": 0.01,
                    "within_tolerance": True
                },
                {
                    "ref": "REF138",
                    "amount_a": 2345.67,
                    "amount_b": 2345.68,
                    "difference": 0.01,
                    "within_tolerance": True
                }
            ]
        },

        "filtered_scenario": {
            "settled_only": {
                "file_a_count": 12,  # Records with Status = 'Settled'
                "file_b_count": 11,  # Records with Settlement_Status = 'SETTLED'
                "expected_matches": 11
            },
            "completed_only": {
                "file_a_count": 4,  # Records with Status = 'Completed'
                "file_b_count": 7,  # Records with Settlement_Status = 'COMPLETE'
                "expected_matches": 4
            }
        }
    }


@pytest.fixture
def test_file_columns():
    """Column definitions for test files"""
    return {
        "file_a_columns": [
            "Transaction_ID", "Date", "Description", "Amount_Text",
            "Status", "Account", "Reference", "Customer_ID", "Branch"
        ],
        "file_b_columns": [
            "Statement_ID", "Process_Date", "Transaction_Desc", "Net_Amount",
            "Settlement_Status", "Account_Number", "Ref_Number", "Client_Code", "Location"
        ]
    }


@pytest.fixture
def reconciliation_regex_patterns():
    """Common regex patterns used in reconciliation testing"""
    return {
        "amount_extraction": r'\$([0-9,.-]+)',
        "date_validation": r'\d{4}-\d{2}-\d{2}',
        "reference_pattern": r'REF\d{3}',
        "account_pattern": r'ACC\d{3}',
        "customer_pattern": r'CUST\d{3}'
    }


@pytest.fixture(scope="session")
def test_database_setup():
    """Setup test database for integration tests"""
    # Mock database setup - replace with actual setup if needed
    test_db_config = {
        "host": "localhost",
        "port": 5433,
        "database": "test_reconciliation",
        "user": "test_user",
        "password": "test_pass"
    }

    # Setup code would go here
    yield test_db_config

    # Cleanup code would go here


@pytest.fixture
def reconciliation_performance_data():
    """Performance benchmarks for reconciliation testing"""
    return {
        "small_dataset": {
            "rows": 100,
            "max_execution_time": 5.0,  # seconds
            "max_memory_mb": 50
        },
        "medium_dataset": {
            "rows": 1000,
            "max_execution_time": 30.0,  # seconds
            "max_memory_mb": 200
        },
        "large_dataset": {
            "rows": 10000,
            "max_execution_time": 300.0,  # seconds
            "max_memory_mb": 1000
        }
    }


# Session-scoped fixtures for expensive setup
@pytest.fixture(scope="session")
def reconciliation_test_environment():
    """Setup comprehensive test environment for reconciliation"""

    # Initialize test environment
    test_env = {
        "test_files_created": 0,
        "reconciliations_processed": 0,
        "start_time": None
    }

    import time
    test_env["start_time"] = time.time()

    yield test_env

    # Cleanup and reporting
    end_time = time.time()
    duration = end_time - test_env["start_time"]

    print(f"\n=== Reconciliation Test Session Summary ===")
    print(f"Total test files created: {test_env['test_files_created']}")
    print(f"Total reconciliations processed: {test_env['reconciliations_processed']}")
    print(f"Total session duration: {duration:.2f} seconds")


# Auto-use fixtures for common setup
@pytest.fixture(autouse=True)
def reconciliation_test_setup(reconciliation_test_environment):
    """Auto-setup for each reconciliation test"""
    # Pre-test setup
    reconciliation_test_environment["test_files_created"] += 1

    yield

    # Post-test cleanup (if needed)
    pass


# Parametrized fixtures for comprehensive testing
@pytest.fixture(params=[
    {"match_type": "equals", "tolerance": 0},
    {"match_type": "tolerance", "tolerance": 0.01},
    {"match_type": "tolerance", "tolerance": 0.05},
    {"match_type": "percentage", "tolerance": 1.0}
])
def reconciliation_rule_params(request):
    """Parametrized reconciliation rule configurations"""
    return request.param


@pytest.fixture(params=["Settled", "Pending", "Completed", "Failed"])
def status_filter_params(request):
    """Parametrized status filter values"""
    return request.param


# Helper functions available to all tests
def create_test_dataframe(data_type="file_a", num_rows=None):
    """Helper function to create test DataFrames"""
    if data_type == "file_a":
        base_data = RECONCILIATION_TEST_FILE_A
    else:
        base_data = RECONCILIATION_TEST_FILE_B

    df = pd.read_csv(io.StringIO(base_data))

    if num_rows:
        # Replicate data to reach desired number of rows
        multiplier = (num_rows // len(df)) + 1
        df = pd.concat([df] * multiplier, ignore_index=True)
        df = df.head(num_rows)

        # Make IDs unique
        if data_type == "file_a":
            df['Transaction_ID'] = df['Transaction_ID'] + '_' + df.index.astype(str)
        else:
            df['Statement_ID'] = df['Statement_ID'] + '_' + df.index.astype(str)

    return df


def assert_reconciliation_summary(result, expected_matches, expected_total_a, expected_total_b):
    """Helper function to assert reconciliation summary results"""
    assert result["summary"]["matched_pairs"] == expected_matches
    assert result["summary"]["total_file_a"] == expected_total_a
    assert result["summary"]["total_file_b"] == expected_total_b

    expected_percentage = (expected_matches / max(expected_total_a, expected_total_b)) * 100
    assert abs(result["summary"]["match_percentage"] - expected_percentage) < 0.1


def validate_reconciliation_config(config):
    """Helper function to validate reconciliation configuration"""
    required_fields = ["Files", "ReconciliationRules"]

    for field in required_fields:
        assert field in config, f"Missing required field: {field}"

    assert len(config["Files"]) >= 2, "At least 2 files required"
    assert len(config["ReconciliationRules"]) >= 1, "At least 1 reconciliation rule required"

    for rule in config["ReconciliationRules"]:
        assert "LeftFileColumn" in rule, "Missing LeftFileColumn in rule"
        assert "RightFileColumn" in rule, "Missing RightFileColumn in rule"
        assert "MatchType" in rule, "Missing MatchType in rule"


# Make helper functions available to all tests
@pytest.fixture
def reconciliation_helpers():
    """Provide helper functions to all tests"""
    return {
        "create_test_dataframe": create_test_dataframe,
        "assert_reconciliation_summary": assert_reconciliation_summary,
        "validate_reconciliation_config": validate_reconciliation_config
    }


# Pytest markers for organizing tests (combining existing and new reconciliation markers)
pytest_markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow tests that take more time",
    "file_upload: File upload related tests",
    "excel: Excel file specific tests",
    "csv: CSV file specific tests",
    "error: Error handling tests",
    "validation: Validation tests",
    "viewer: File viewer related tests",
    "save_results: Save results related tests",
    "smoke: Smoke tests for critical functionality",
    # Reconciliation-specific markers
    "reconciliation: Reconciliation system tests",
    "extraction: Data extraction tests",
    "filtering: Data filtering tests",
    "matching: Record matching tests",
    "tolerance: Tolerance-based matching tests",
    "performance: Performance and load tests"
]


def pytest_configure(config):
    """Configure pytest markers"""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)


# Add missing import for contextlib
import contextlib  # tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from typing import Dict, Any

# Import your FastAPI routers
from fastapi import FastAPI

# Import routers safely
try:
    from app.routes.file_routes import router as file_router

    HAS_FILE_ROUTES = True
except ImportError as e:
    print(f"Warning: Could not import file_routes: {e}")
    file_router = None
    HAS_FILE_ROUTES = False

try:
    from app.routes.viewer_routes import router as viewer_router

    HAS_VIEWER_ROUTES = True
except ImportError as e:
    print(f"Warning: Could not import viewer_routes: {e}")
    viewer_router = None
    HAS_VIEWER_ROUTES = False

# Mock storage for testing
test_uploaded_files: Dict[str, Dict[str, Any]] = {}

# Reconciliation test data constants
RECONCILIATION_TEST_FILE_A = """Transaction_ID,Date,Description,Amount_Text,Status,Account,Reference,Customer_ID,Branch
TXN001,2024-01-15,Payment to Vendor ABC,$1,234.56,Settled,ACC001,REF123,CUST001,BR01
TXN002,2024-01-16,Salary Payment,$2,500.00,Pending,ACC002,REF124,CUST002,BR02
TXN003,2024-01-17,Refund Processing,$-456.78,Completed,ACC003,REF125,CUST003,BR01
TXN004,2024-01-18,Investment Transfer,$10,000.00,Settled,ACC001,REF126,CUST001,BR03
TXN005,2024-01-19,Utility Bill Payment,$234.50,Failed,ACC004,REF127,CUST004,BR02
TXN006,2024-01-20,Customer Deposit,$3,456.78,Settled,ACC002,REF128,CUST002,BR01
TXN007,2024-01-21,Loan Disbursement,$15,000.00,Pending,ACC005,REF129,CUST005,BR03
TXN008,2024-01-22,Card Payment,$89.99,Settled,ACC003,REF130,CUST003,BR02
TXN009,2024-01-23,Wire Transfer,$5,678.90,Completed,ACC001,REF131,CUST006,BR01
TXN010,2024-01-24,ATM Withdrawal,$-200.00,Settled,ACC004,REF132,CUST004,BR02
TXN011,2024-01-25,Online Purchase,$125.75,Settled,ACC002,REF133,CUST002,BR03
TXN012,2024-01-26,Insurance Premium,$567.89,Pending,ACC006,REF134,CUST007,BR01
TXN013,2024-01-27,Dividend Payment,$1,000.00,Settled,ACC001,REF135,CUST001,BR02
TXN014,2024-01-28,Merchant Payment,$345.67,Completed,ACC003,REF136,CUST003,BR03
TXN015,2024-01-29,Subscription Fee,$29.99,Settled,ACC004,REF137,CUST004,BR01
TXN016,2024-01-30,International Transfer,$2,345.67,Pending,ACC005,REF138,CUST008,BR02
TXN017,2024-02-01,Cash Deposit,$500.00,Settled,ACC002,REF139,CUST002,BR01
TXN018,2024-02-02,Bill Payment,$78.45,Completed,ACC006,REF140,CUST007,BR03
TXN019,2024-02-03,Investment Return,$1,567.89,Settled,ACC001,REF141,CUST001,BR02
TXN020,2024-02-04,Service Charge,$-25.00,Settled,ACC003,REF142,CUST003,BR01"""

RECONCILIATION_TEST_FILE_B = """Statement_ID,Process_Date,Transaction_Desc,Net_Amount,Settlement_Status,Account_Number,Ref_Number,Client_Code,Location
STMT001,15/01/2024,Vendor ABC Payment,1234.56,SETTLED,ACC001,REF123,CUST001,BR01
STMT002,16/01/2024,Employee Salary,2500.01,PROCESSING,ACC002,REF124,CUST002,BR02
STMT003,17/01/2024,Customer Refund,-456.78,COMPLETE,ACC003,REF125,CUST003,BR01
STMT004,18/01/2024,Investment Xfer,10000.00,SETTLED,ACC001,REF126,CUST001,BR03
STMT005,19/01/2024,Utility Payment,234.50,REJECTED,ACC004,REF127,CUST004,BR02
STMT006,20/01/2024,Deposit from Customer,3456.78,SETTLED,ACC002,REF128,CUST002,BR01
STMT007,21/01/2024,Loan Payment,15000.00,PROCESSING,ACC005,REF129,CUST005,BR03
STMT008,22/01/2024,Credit Card Transaction,89.99,SETTLED,ACC003,REF130,CUST003,BR02
STMT009,23/01/2024,Wire Transfer Out,5678.90,COMPLETE,ACC001,REF131,CUST006,BR01
STMT010,24/01/2024,ATM Cash Withdrawal,-200.00,SETTLED,ACC004,REF132,CUST004,BR02
STMT011,25/01/2024,E-commerce Purchase,125.75,SETTLED,ACC002,REF133,CUST002,BR03
STMT012,26/01/2024,Insurance Payment,567.89,PROCESSING,ACC006,REF134,CUST007,BR01
STMT013,27/01/2024,Dividend Distribution,1000.00,SETTLED,ACC001,REF135,CUST001,BR02
STMT014,28/01/2024,POS Transaction,345.67,COMPLETE,ACC003,REF136,CUST003,BR03
STMT015,29/01/2024,Monthly Subscription,29.99,SETTLED,ACC004,REF137,CUST004,BR01
STMT016,30/01/2024,Foreign Exchange,2345.68,PROCESSING,ACC005,REF138,CUST008,BR02
STMT017,01/02/2024,Cash Deposit Transaction,500.00,SETTLED,ACC002,REF139,CUST002,BR01
STMT018,02/02/2024,Utility Bill Settlement,78.45,COMPLETE,ACC006,REF140,CUST007,BR03
STMT019,03/02/2024,Investment Profit,1567.89,SETTLED,ACC001,REF141,CUST001,BR02
STMT020,04/02/2024,Bank Service Fee,-25.00,SETTLED,ACC003,REF142,CUST003,BR01
STMT021,05/02/2024,Unmatched Transaction,999.99,SETTLED,ACC007,REF999,CUST999,BR04
STMT022,06/02/2024,Another Unmatched,777.77,COMPLETE,ACC008,REF888,CUST888,BR05"""


@pytest.fixture(scope="session")
def app():
    """Create FastAPI app instance for testing"""
    test_app = FastAPI(title="Test App")

    if HAS_FILE_ROUTES and file_router:
        test_app.include_router(file_router)

    if HAS_VIEWER_ROUTES and viewer_router:
        test_app.include_router(viewer_router)

    return test_app


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_storage():
    """Mock the storage service for all tests"""
    test_uploaded_files.clear()  # Clear storage before each test

    # Patch where uploaded_files is actually used (in the route files)
    # not where it's defined (in the storage service)
    try:
        with patch('app.routes.file_routes.uploaded_files', test_uploaded_files):
            try:
                with patch('app.routes.viewer_routes.uploaded_files', test_uploaded_files):
                    yield test_uploaded_files
            except ImportError:
                yield test_uploaded_files
    except ImportError:
        # Fallback if route imports fail
        with patch('app.services.storage_service.uploaded_files', test_uploaded_files):
            yield test_uploaded_files


@pytest.fixture
def sample_csv_content():
    """Create sample CSV content"""
    return """name,age,city,salary
John Doe,25,New York,50000
Jane Smith,30,Los Angeles,60000
Bob Johnson,35,Chicago,55000
Alice Brown,28,Houston,52000
Charlie Wilson,32,Phoenix,58000"""


@pytest.fixture
def sample_csv_file(sample_csv_content):
    """Create sample CSV file object"""
    return io.BytesIO(sample_csv_content.encode('utf-8'))


@pytest.fixture
def large_csv_content():
    """Create large CSV content for testing file size limits"""
    header = "id,name,value,description\n"
    rows = []
    for i in range(10000):  # 10k rows
        rows.append(f"{i},Name_{i},{i * 100},Description for item {i}")
    return header + "\n".join(rows)


@pytest.fixture
def large_csv_file(large_csv_content):
    """Create large CSV file object"""
    return io.BytesIO(large_csv_content.encode('utf-8'))


@pytest.fixture
def sample_excel_file():
    """Create sample Excel file with multiple sheets"""
    # Create Excel file in memory
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Sales data
        sales_data = pd.DataFrame({
            'product': ['Product A', 'Product B', 'Product C', 'Product D'],
            'sales': [1000, 1500, 800, 1200],
            'region': ['North', 'South', 'East', 'West']
        })
        sales_data.to_excel(writer, sheet_name='Sales', index=False)

        # Sheet 2: Employee data
        employee_data = pd.DataFrame({
            'employee_id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'department': ['HR', 'IT', 'Finance', 'IT', 'HR'],
            'salary': [55000, 65000, 60000, 70000, 58000]
        })
        employee_data.to_excel(writer, sheet_name='Employees', index=False)

        # Sheet 3: Empty sheet
        empty_data = pd.DataFrame()
        empty_data.to_excel(writer, sheet_name='Empty', index=False)

    output.seek(0)
    return output


@pytest.fixture
def invalid_file():
    """Create invalid file (not CSV or Excel)"""
    return io.BytesIO(b"This is not a valid CSV or Excel file content")


@pytest.fixture
def corrupted_excel_file():
    """Create corrupted Excel file"""
    return io.BytesIO(b"PK\x03\x04\x14\x00corrupted excel content")


@pytest.fixture
def empty_csv_file():
    """Create empty CSV file"""
    return io.BytesIO(b"")


@pytest.fixture
def csv_with_special_chars():
    """Create CSV with special characters and edge cases"""
    content = """name,description,value
"John, Jr.",Product with "quotes",100.50
Jane's Item,"Line 1
Line 2",200
Special Chars,Café & Résumé,300.75"""
    return io.BytesIO(content.encode('utf-8'))


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    env_vars = {
        'MAX_FILE_SIZE': '10',  # 10MB for testing
        'MAX_SAMPLE_ROWS': '50',
        'MAX_PREVIEW_ROWS': '100',
        'LARGE_FILE_THRESHOLD': '1000'
    }

    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def sample_file_data():
    """Create sample file data for testing responses"""
    return {
        "file_id": "test-file-id-123",
        "filename": "test.csv",
        "custom_name": None,
        "file_type": "csv",
        "file_size_mb": 0.001,
        "total_rows": 5,
        "total_columns": 4,
        "columns": ["name", "age", "city", "salary"],
        "upload_time": "2025-07-15T10:00:00",
        "data_types": {
            "name": "object",
            "age": "int64",
            "city": "object",
            "salary": "int64"
        },
        "sheet_name": None
    }


@pytest.fixture
def uploaded_file_with_data(sample_file_data, sample_csv_content):
    """Create uploaded file entry with data"""
    df = pd.read_csv(io.StringIO(sample_csv_content))
    file_entry = {
        "info": sample_file_data,
        "data": df
    }
    test_uploaded_files[sample_file_data["file_id"]] = file_entry
    return file_entry


# === RECONCILIATION-SPECIFIC FIXTURES ===

@pytest.fixture(scope="session")
def reconciliation_test_data():
    """Provide reconciliation test data for all tests"""
    return {
        "file_a_csv": RECONCILIATION_TEST_FILE_A,
        "file_b_csv": RECONCILIATION_TEST_FILE_B,
        "file_a_df": pd.read_csv(io.StringIO(RECONCILIATION_TEST_FILE_A)),
        "file_b_df": pd.read_csv(io.StringIO(RECONCILIATION_TEST_FILE_B))
    }


@pytest.fixture
def temp_reconciliation_files(reconciliation_test_data):
    """Create temporary CSV files for reconciliation testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_a_path = os.path.join(temp_dir, "financial_transactions.csv")
        file_b_path = os.path.join(temp_dir, "bank_statements.csv")

        with open(file_a_path, 'w') as f:
            f.write(reconciliation_test_data["file_a_csv"])

        with open(file_b_path, 'w') as f:
            f.write(reconciliation_test_data["file_b_csv"])

        yield {
            "file_a_path": file_a_path,
            "file_b_path": file_b_path,
            "temp_dir": temp_dir,
            "file_a_name": "financial_transactions.csv",
            "file_b_name": "bank_statements.csv"
        }


@pytest.fixture
def reconciliation_test_files_as_bytesio(reconciliation_test_data):
    """Create reconciliation test files as BytesIO objects for file uploads"""
    return {
        "file_a": io.BytesIO(reconciliation_test_data["file_a_csv"].encode('utf-8')),
        "file_b": io.BytesIO(reconciliation_test_data["file_b_csv"].encode('utf-8'))
    }


@pytest.fixture
def mock_reconciliation_api():
    """Mock reconciliation API client for testing"""
    client = MagicMock()

    # Mock file upload responses
    client.upload_file.side_effect = lambda x: {
        "file_id": f"test_file_{hash(x) % 1000:03d}",
        "filename": os.path.basename(x),
        "status": "success",
        "total_rows": 20 if "financial" in x else 22,
        "columns": ["Transaction_ID", "Date", "Amount_Text"] if "financial" in x else ["Statement_ID", "Process_Date",
                                                                                       "Net_Amount"]
    }

    # Mock column unique values
    client.get_column_unique_values.return_value = {
        "unique_values": ["Settled", "Pending", "Completed", "Failed"],
        "total_unique": 4,
        "is_date_column": False
    }

    # Mock reconciliation processing
    client.process_reconciliation.return_value = {
        "reconciliation_id": "recon_001",
        "status": "completed",
        "summary": {
            "total_file_a": 20,
            "total_file_b": 22,
            "matched_pairs": 20,
            "unmatched_a_count": 0,
            "unmatched_b_count": 2,
            "match_percentage": 90.9
        },
        "matched_records": [],
        "unmatched_records": []
    }

    return client


@pytest.fixture
def reconciliation_configs():
    """Provide various reconciliation configurations for testing"""
    return {
        "basic_reference_match": {
            "Files": [
                {"Name": "FileA", "Extract": [], "Filter": []},
                {"Name": "FileB", "Extract": [], "Filter": []}
            ],
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Reference",
                    "RightFileColumn": "Ref_Number",
                    "MatchType": "equals"
                }
            ]
        },

        "amount_extraction_with_tolerance": {
            "Files": [
                {
                    "Name": "FileA",
                    "Extract": [
                        {
                            "ResultColumnName": "Extracted_Amount",
                            "SourceColumn": "Amount_Text",
                            "MatchType": "regex",
                            "Patterns": ["\\$([0-9,.-]+)"]
                        }
                    ],
                    "Filter": []
                },
                {"Name": "FileB", "Extract": [], "Filter": []}
            ],
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Reference",
                    "RightFileColumn": "Ref_Number",
                    "MatchType": "equals"
                },
                {
                    "LeftFileColumn": "Extracted_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.01
                }
            ]
        },

        "status_filtered_reconciliation": {
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
                    "MatchType": "equals"
                }
            ]
        },

        "multi_column_complex_match": {
            "Files": [
                {
                    "Name": "FileA",
                    "Extract": [
                        {
                            "ResultColumnName": "Clean_Amount",
                            "SourceColumn": "Amount_Text",
                            "MatchType": "regex",
                            "Patterns": ["\\$([0-9,.-]+)"]
                        }
                    ],
                    "Filter": [
                        {
                            "ColumnName": "Status",
                            "MatchType": "equals",
                            "Value": "Settled"
                        },
                        {
                            "ColumnName": "Status",
                            "MatchType": "equals",
                            "Value": "Completed"
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
                        },
                        {
                            "ColumnName": "Settlement_Status",
                            "MatchType": "equals",
                            "Value": "COMPLETE"
                        }
                    ]
                }
            ],
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Reference",
                    "RightFileColumn": "Ref_Number",
                    "MatchType": "equals"
                },
                {
                    "LeftFileColumn": "Account",
                    "RightFileColumn": "Account_Number",
                    "MatchType": "equals"
                },
                {
                    "LeftFileColumn": "Customer_ID",
                    "RightFileColumn": "Client_Code",
                    "MatchType": "equals"
                },
                {
                    "LeftFileColumn": "Clean_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.05
                }
            ]
        }
    }


@pytest.fixture
def mock_delta_processor():
    """Mock DeltaProcessor for testing reconciliation logic"""
    with patch('app.services.reconciliation_service.DeltaProcessor') as mock_processor_class:
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Setup default mock behaviors
        mock_processor.extract_data.return_value = pd.DataFrame()
        mock_processor.apply_filters.return_value = pd.DataFrame()
        mock_processor.process_reconciliation.return_value = {
            "matched_records": [],
            "unmatched_a": [],
            "unmatched_b": [],
            "summary": {
                "matched_pairs": 0,
                "total_file_a": 0,
                "total_file_b": 0,
                "match_percentage": 0.0
            }
        }
        mock_processor.parse_excel_date.return_value = None

        yield mock_processor


@pytest.fixture
def reconciliation_expected_results():
    """Expected results for reconciliation test scenarios"""
    return {
        "perfect_match_scenario": {
            "summary": {
                "total_file_a": 20,
                "total_file_b": 22,
                "matched_pairs": 20,
                "unmatched_a_count": 0,
                "unmatched_b_count": 2,
                "match_percentage": 90.9
            },
            "matched_count": 20,
            "unmatched_refs": ["REF999", "REF888"]
        },

        "tolerance_match_scenario": {
            "tolerance_matches": [
                {
                    "ref": "REF124",
                    "amount_a": 2500.00,
                    "amount_b": 2500.01,
                    "difference": 0.01,
                    "within_tolerance": True
                },
                {
                    "ref": "REF138",
                    "amount_a": 2345.67,
                    "amount_b": 2345.68,
                    "difference": 0.01,
                    "within_tolerance": True
                }
            ]
        },

        "filtered_scenario": {
            "settled_only": {
                "file_a_count": 12,  # Records with Status = 'Settled'
                "file_b_count": 11,  # Records with Settlement_Status = 'SETTLED'
                "expected_matches": 11
            },
            "completed_only": {
                "file_a_count": 4,  # Records with Status = 'Completed'
                "file_b_count": 7,  # Records with Settlement_Status = 'COMPLETE'
                "expected_matches": 4
            }
        }
    }


@pytest.fixture
def test_file_columns():
    """Column definitions for test files"""
    return {
        "file_a_columns": [
            "Transaction_ID", "Date", "Description", "Amount_Text",
            "Status", "Account", "Reference", "Customer_ID", "Branch"
        ],
        "file_b_columns": [
            "Statement_ID", "Process_Date", "Transaction_Desc", "Net_Amount",
            "Settlement_Status", "Account_Number", "Ref_Number", "Client_Code", "Location"
        ]
    }


@pytest.fixture
def reconciliation_regex_patterns():
    """Common regex patterns used in reconciliation testing"""
    return {
        "amount_extraction": r'\$([0-9,.-]+)',
        "date_validation": r'\d{4}-\d{2}-\d{2}',
        "reference_pattern": r'REF\d{3}',
        "account_pattern": r'ACC\d{3}',
        "customer_pattern": r'CUST\d{3}'
    }


@pytest.fixture(scope="session")
def test_database_setup():
    """Setup test database for integration tests"""
    # Mock database setup - replace with actual setup if needed
    test_db_config = {
        "host": "localhost",
        "port": 5433,
        "database": "test_reconciliation",
        "user": "test_user",
        "password": "test_pass"
    }

    # Setup code would go here
    yield test_db_config

    # Cleanup code would go here


@pytest.fixture
def reconciliation_performance_data():
    """Performance benchmarks for reconciliation testing"""
    return {
        "small_dataset": {
            "rows": 100,
            "max_execution_time": 5.0,  # seconds
            "max_memory_mb": 50
        },
        "medium_dataset": {
            "rows": 1000,
            "max_execution_time": 30.0,  # seconds
            "max_memory_mb": 200
        },
        "large_dataset": {
            "rows": 10000,
            "max_execution_time": 300.0,  # seconds
            "max_memory_mb": 1000
        }
    }


# Session-scoped fixtures for expensive setup
@pytest.fixture(scope="session")
def reconciliation_test_environment():
    """Setup comprehensive test environment for reconciliation"""

    # Initialize test environment
    test_env = {
        "test_files_created": 0,
        "reconciliations_processed": 0,
        "start_time": None
    }

    import time
    test_env["start_time"] = time.time()

    yield test_env

    # Cleanup and reporting
    end_time = time.time()
    duration = end_time - test_env["start_time"]

    print(f"\n=== Reconciliation Test Session Summary ===")
    print(f"Total test files created: {test_env['test_files_created']}")
    print(f"Total reconciliations processed: {test_env['reconciliations_processed']}")
    print(f"Total session duration: {duration:.2f} seconds")


# Auto-use fixtures for common setup
@pytest.fixture(autouse=True)
def reconciliation_test_setup(reconciliation_test_environment):
    """Auto-setup for each reconciliation test"""
    # Pre-test setup
    reconciliation_test_environment["test_files_created"] += 1

    yield

    # Post-test cleanup (if needed)
    pass


# Parametrized fixtures for comprehensive testing
@pytest.fixture(params=[
    {"match_type": "equals", "tolerance": 0},
    {"match_type": "tolerance", "tolerance": 0.01},
    {"match_type": "tolerance", "tolerance": 0.05},
    {"match_type": "percentage", "tolerance": 1.0}
])
def reconciliation_rule_params(request):
    """Parametrized reconciliation rule configurations"""
    return request.param


@pytest.fixture(params=["Settled", "Pending", "Completed", "Failed"])
def status_filter_params(request):
    """Parametrized status filter values"""
    return request.param


# Helper functions available to all tests
def create_test_dataframe(data_type="file_a", num_rows=None):
    """Helper function to create test DataFrames"""
    if data_type == "file_a":
        base_data = RECONCILIATION_TEST_FILE_A
    else:
        base_data = RECONCILIATION_TEST_FILE_B

    df = pd.read_csv(io.StringIO(base_data))

    if num_rows:
        # Replicate data to reach desired number of rows
        multiplier = (num_rows // len(df)) + 1
        df = pd.concat([df] * multiplier, ignore_index=True)
        df = df.head(num_rows)

        # Make IDs unique
        if data_type == "file_a":
            df['Transaction_ID'] = df['Transaction_ID'] + '_' + df.index.astype(str)
        else:
            df['Statement_ID'] = df['Statement_ID'] + '_' + df.index.astype(str)

    return df


def assert_reconciliation_summary(result, expected_matches, expected_total_a, expected_total_b):
    """Helper function to assert reconciliation summary results"""
    assert result["summary"]["matched_pairs"] == expected_matches
    assert result["summary"]["total_file_a"] == expected_total_a
    assert result["summary"]["total_file_b"] == expected_total_b

    expected_percentage = (expected_matches / max(expected_total_a, expected_total_b)) * 100
    assert abs(result["summary"]["match_percentage"] - expected_percentage) < 0.1


def validate_reconciliation_config(config):
    """Helper function to validate reconciliation configuration"""
    required_fields = ["Files", "ReconciliationRules"]

    for field in required_fields:
        assert field in config, f"Missing required field: {field}"

    assert len(config["Files"]) >= 2, "At least 2 files required"
    assert len(config["ReconciliationRules"]) >= 1, "At least 1 reconciliation rule required"

    for rule in config["ReconciliationRules"]:
        assert "LeftFileColumn" in rule, "Missing LeftFileColumn in rule"
        assert "RightFileColumn" in rule, "Missing RightFileColumn in rule"
        assert "MatchType" in rule, "Missing MatchType in rule"


# Make helper functions available to all tests
@pytest.fixture
def reconciliation_helpers():
    """Provide helper functions to all tests"""
    return {
        "create_test_dataframe": create_test_dataframe,
        "assert_reconciliation_summary": assert_reconciliation_summary,
        "validate_reconciliation_config": validate_reconciliation_config
    }


# Pytest markers for organizing tests (combining existing and new reconciliation markers)
pytest_markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow tests that take more time",
    "file_upload: File upload related tests",
    "excel: Excel file specific tests",
    "csv: CSV file specific tests",
    "error: Error handling tests",
    "validation: Validation tests",
    "viewer: File viewer related tests",
    "save_results: Save results related tests",
    "smoke: Smoke tests for critical functionality",
    # Reconciliation-specific markers
    "reconciliation: Reconciliation system tests",
    "extraction: Data extraction tests",
    "filtering: Data filtering tests",
    "matching: Record matching tests",
    "tolerance: Tolerance-based matching tests",
    "performance: Performance and load tests"
]


def pytest_configure(config):
    """Configure pytest markers"""
    for marker in pytest_markers:
        config.addinivalue_line("markers",
                                marker)  # tests/conftest.py - Pytest fixtures and configuration for reconciliation tests


import pytest
import os
import tempfile
import pandas as pd
import io
from unittest.mock import MagicMock, patch

# Test data constants (embedded in conftest for reuse across all test files)
RECONCILIATION_TEST_FILE_A = """Transaction_ID,Date,Description,Amount_Text,Status,Account,Reference,Customer_ID,Branch
TXN001,2024-01-15,Payment to Vendor ABC,$1,234.56,Settled,ACC001,REF123,CUST001,BR01
TXN002,2024-01-16,Salary Payment,$2,500.00,Pending,ACC002,REF124,CUST002,BR02
TXN003,2024-01-17,Refund Processing,$-456.78,Completed,ACC003,REF125,CUST003,BR01
TXN004,2024-01-18,Investment Transfer,$10,000.00,Settled,ACC001,REF126,CUST001,BR03
TXN005,2024-01-19,Utility Bill Payment,$234.50,Failed,ACC004,REF127,CUST004,BR02
TXN006,2024-01-20,Customer Deposit,$3,456.78,Settled,ACC002,REF128,CUST002,BR01
TXN007,2024-01-21,Loan Disbursement,$15,000.00,Pending,ACC005,REF129,CUST005,BR03
TXN008,2024-01-22,Card Payment,$89.99,Settled,ACC003,REF130,CUST003,BR02
TXN009,2024-01-23,Wire Transfer,$5,678.90,Completed,ACC001,REF131,CUST006,BR01
TXN010,2024-01-24,ATM Withdrawal,$-200.00,Settled,ACC004,REF132,CUST004,BR02
TXN011,2024-01-25,Online Purchase,$125.75,Settled,ACC002,REF133,CUST002,BR03
TXN012,2024-01-26,Insurance Premium,$567.89,Pending,ACC006,REF134,CUST007,BR01
TXN013,2024-01-27,Dividend Payment,$1,000.00,Settled,ACC001,REF135,CUST001,BR02
TXN014,2024-01-28,Merchant Payment,$345.67,Completed,ACC003,REF136,CUST003,BR03
TXN015,2024-01-29,Subscription Fee,$29.99,Settled,ACC004,REF137,CUST004,BR01
TXN016,2024-01-30,International Transfer,$2,345.67,Pending,ACC005,REF138,CUST008,BR02
TXN017,2024-02-01,Cash Deposit,$500.00,Settled,ACC002,REF139,CUST002,BR01
TXN018,2024-02-02,Bill Payment,$78.45,Completed,ACC006,REF140,CUST007,BR03
TXN019,2024-02-03,Investment Return,$1,567.89,Settled,ACC001,REF141,CUST001,BR02
TXN020,2024-02-04,Service Charge,$-25.00,Settled,ACC003,REF142,CUST003,BR01"""

RECONCILIATION_TEST_FILE_B = """Statement_ID,Process_Date,Transaction_Desc,Net_Amount,Settlement_Status,Account_Number,Ref_Number,Client_Code,Location
STMT001,15/01/2024,Vendor ABC Payment,1234.56,SETTLED,ACC001,REF123,CUST001,BR01
STMT002,16/01/2024,Employee Salary,2500.01,PROCESSING,ACC002,REF124,CUST002,BR02
STMT003,17/01/2024,Customer Refund,-456.78,COMPLETE,ACC003,REF125,CUST003,BR01
STMT004,18/01/2024,Investment Xfer,10000.00,SETTLED,ACC001,REF126,CUST001,BR03
STMT005,19/01/2024,Utility Payment,234.50,REJECTED,ACC004,REF127,CUST004,BR02
STMT006,20/01/2024,Deposit from Customer,3456.78,SETTLED,ACC002,REF128,CUST002,BR01
STMT007,21/01/2024,Loan Payment,15000.00,PROCESSING,ACC005,REF129,CUST005,BR03
STMT008,22/01/2024,Credit Card Transaction,89.99,SETTLED,ACC003,REF130,CUST003,BR02
STMT009,23/01/2024,Wire Transfer Out,5678.90,COMPLETE,ACC001,REF131,CUST006,BR01
STMT010,24/01/2024,ATM Cash Withdrawal,-200.00,SETTLED,ACC004,REF132,CUST004,BR02
STMT011,25/01/2024,E-commerce Purchase,125.75,SETTLED,ACC002,REF133,CUST002,BR03
STMT012,26/01/2024,Insurance Payment,567.89,PROCESSING,ACC006,REF134,CUST007,BR01
STMT013,27/01/2024,Dividend Distribution,1000.00,SETTLED,ACC001,REF135,CUST001,BR02
STMT014,28/01/2024,POS Transaction,345.67,COMPLETE,ACC003,REF136,CUST003,BR03
STMT015,29/01/2024,Monthly Subscription,29.99,SETTLED,ACC004,REF137,CUST004,BR01
STMT016,30/01/2024,Foreign Exchange,2345.68,PROCESSING,ACC005,REF138,CUST008,BR02
STMT017,01/02/2024,Cash Deposit Transaction,500.00,SETTLED,ACC002,REF139,CUST002,BR01
STMT018,02/02/2024,Utility Bill Settlement,78.45,COMPLETE,ACC006,REF140,CUST007,BR03
STMT019,03/02/2024,Investment Profit,1567.89,SETTLED,ACC001,REF141,CUST001,BR02
STMT020,04/02/2024,Bank Service Fee,-25.00,SETTLED,ACC003,REF142,CUST003,BR01
STMT021,05/02/2024,Unmatched Transaction,999.99,SETTLED,ACC007,REF999,CUST999,BR04
STMT022,06/02/2024,Another Unmatched,777.77,COMPLETE,ACC008,REF888,CUST888,BR05"""


@pytest.fixture(scope="session")
def reconciliation_test_data():
    """Provide reconciliation test data for all tests"""
    return {
        "file_a_csv": RECONCILIATION_TEST_FILE_A,
        "file_b_csv": RECONCILIATION_TEST_FILE_B,
        "file_a_df": pd.read_csv(io.StringIO(RECONCILIATION_TEST_FILE_A)),
        "file_b_df": pd.read_csv(io.StringIO(RECONCILIATION_TEST_FILE_B))
    }


@pytest.fixture
def temp_reconciliation_files(reconciliation_test_data):
    """Create temporary CSV files for reconciliation testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_a_path = os.path.join(temp_dir, "financial_transactions.csv")
        file_b_path = os.path.join(temp_dir, "bank_statements.csv")

        with open(file_a_path, 'w') as f:
            f.write(reconciliation_test_data["file_a_csv"])

        with open(file_b_path, 'w') as f:
            f.write(reconciliation_test_data["file_b_csv"])

        yield {
            "file_a_path": file_a_path,
            "file_b_path": file_b_path,
            "temp_dir": temp_dir,
            "file_a_name": "financial_transactions.csv",
            "file_b_name": "bank_statements.csv"
        }


@pytest.fixture
def mock_reconciliation_api():
    """Mock reconciliation API client for testing"""
    client = MagicMock()

    # Mock file upload responses
    client.upload_file.side_effect = lambda x: {
        "file_id": f"test_file_{hash(x) % 1000:03d}",
        "filename": os.path.basename(x),
        "status": "success",
        "total_rows": 20 if "financial" in x else 22,
        "columns": ["Transaction_ID", "Date", "Amount_Text"] if "financial" in x else ["Statement_ID", "Process_Date",
                                                                                       "Net_Amount"]
    }

    # Mock column unique values
    client.get_column_unique_values.return_value = {
        "unique_values": ["Settled", "Pending", "Completed", "Failed"],
        "total_unique": 4,
        "is_date_column": False
    }

    # Mock reconciliation processing
    client.process_reconciliation.return_value = {
        "reconciliation_id": "recon_001",
        "status": "completed",
        "summary": {
            "total_file_a": 20,
            "total_file_b": 22,
            "matched_pairs": 20,
            "unmatched_a_count": 0,
            "unmatched_b_count": 2,
            "match_percentage": 90.9
        },
        "matched_records": [],
        "unmatched_records": []
    }

    return client


@pytest.fixture
def reconciliation_configs():
    """Provide various reconciliation configurations for testing"""
    return {
        "basic_reference_match": {
            "Files": [
                {"Name": "FileA", "Extract": [], "Filter": []},
                {"Name": "FileB", "Extract": [], "Filter": []}
            ],
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Reference",
                    "RightFileColumn": "Ref_Number",
                    "MatchType": "equals"
                }
            ]
        },

        "amount_extraction_with_tolerance": {
            "Files": [
                {
                    "Name": "FileA",
                    "Extract": [
                        {
                            "ResultColumnName": "Extracted_Amount",
                            "SourceColumn": "Amount_Text",
                            "MatchType": "regex",
                            "Patterns": ["\\$([0-9,.-]+)"]
                        }
                    ],
                    "Filter": []
                },
                {"Name": "FileB", "Extract": [], "Filter": []}
            ],
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Reference",
                    "RightFileColumn": "Ref_Number",
                    "MatchType": "equals"
                },
                {
                    "LeftFileColumn": "Extracted_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.01
                }
            ]
        },

        "status_filtered_reconciliation": {
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
                    "MatchType": "equals"
                }
            ]
        },

        "multi_column_complex_match": {
            "Files": [
                {
                    "Name": "FileA",
                    "Extract": [
                        {
                            "ResultColumnName": "Clean_Amount",
                            "SourceColumn": "Amount_Text",
                            "MatchType": "regex",
                            "Patterns": ["\\$([0-9,.-]+)"]
                        }
                    ],
                    "Filter": [
                        {
                            "ColumnName": "Status",
                            "MatchType": "equals",
                            "Value": "Settled"
                        },
                        {
                            "ColumnName": "Status",
                            "MatchType": "equals",
                            "Value": "Completed"
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
                        },
                        {
                            "ColumnName": "Settlement_Status",
                            "MatchType": "equals",
                            "Value": "COMPLETE"
                        }
                    ]
                }
            ],
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Reference",
                    "RightFileColumn": "Ref_Number",
                    "MatchType": "equals"
                },
                {
                    "LeftFileColumn": "Account",
                    "RightFileColumn": "Account_Number",
                    "MatchType": "equals"
                },
                {
                    "LeftFileColumn": "Customer_ID",
                    "RightFileColumn": "Client_Code",
                    "MatchType": "equals"
                },
                {
                    "LeftFileColumn": "Clean_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.05
                }
            ]
        }
    }


@pytest.fixture
def mock_delta_processor():
    """Mock DeltaProcessor for testing reconciliation logic"""
    with patch('app.services.reconciliation_service.DeltaProcessor') as mock_processor_class:
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Setup default mock behaviors
        mock_processor.extract_data.return_value = pd.DataFrame()
        mock_processor.apply_filters.return_value = pd.DataFrame()
        mock_processor.process_reconciliation.return_value = {
            "matched_records": [],
            "unmatched_a": [],
            "unmatched_b": [],
            "summary": {
                "matched_pairs": 0,
                "total_file_a": 0,
                "total_file_b": 0,
                "match_percentage": 0.0
            }
        }
        mock_processor.parse_excel_date.return_value = None

        yield mock_processor


@pytest.fixture
def reconciliation_expected_results():
    """Expected results for reconciliation test scenarios"""
    return {
        "perfect_match_scenario": {
            "summary": {
                "total_file_a": 20,
                "total_file_b": 22,
                "matched_pairs": 20,
                "unmatched_a_count": 0,
                "unmatched_b_count": 2,
                "match_percentage": 90.9
            },
            "matched_count": 20,
            "unmatched_refs": ["REF999", "REF888"]
        },

        "tolerance_match_scenario": {
            "tolerance_matches": [
                {
                    "ref": "REF124",
                    "amount_a": 2500.00,
                    "amount_b": 2500.01,
                    "difference": 0.01,
                    "within_tolerance": True
                },
                {
                    "ref": "REF138",
                    "amount_a": 2345.67,
                    "amount_b": 2345.68,
                    "difference": 0.01,
                    "within_tolerance": True
                }
            ]
        },

        "filtered_scenario": {
            "settled_only": {
                "file_a_count": 12,  # Records with Status = 'Settled'
                "file_b_count": 11,  # Records with Settlement_Status = 'SETTLED'
                "expected_matches": 11
            },
            "completed_only": {
                "file_a_count": 4,  # Records with Status = 'Completed'
                "file_b_count": 7,  # Records with Settlement_Status = 'COMPLETE'
                "expected_matches": 4
            }
        }
    }


@pytest.fixture
def test_file_columns():
    """Column definitions for test files"""
    return {
        "file_a_columns": [
            "Transaction_ID", "Date", "Description", "Amount_Text",
            "Status", "Account", "Reference", "Customer_ID", "Branch"
        ],
        "file_b_columns": [
            "Statement_ID", "Process_Date", "Transaction_Desc", "Net_Amount",
            "Settlement_Status", "Account_Number", "Ref_Number", "Client_Code", "Location"
        ]
    }


@pytest.fixture
def reconciliation_regex_patterns():
    """Common regex patterns used in reconciliation testing"""
    return {
        "amount_extraction": r'\$([0-9,.-]+)',
        "date_validation": r'\d{4}-\d{2}-\d{2}',
        "reference_pattern": r'REF\d{3}',
        "account_pattern": r'ACC\d{3}',
        "customer_pattern": r'CUST\d{3}'
    }


@pytest.fixture(scope="session")
def test_database_setup():
    """Setup test database for integration tests"""
    # Mock database setup - replace with actual setup if needed
    test_db_config = {
        "host": "localhost",
        "port": 5433,
        "database": "test_reconciliation",
        "user": "test_user",
        "password": "test_pass"
    }

    # Setup code would go here
    yield test_db_config

    # Cleanup code would go here


@pytest.fixture
def reconciliation_performance_data():
    """Performance benchmarks for reconciliation testing"""
    return {
        "small_dataset": {
            "rows": 100,
            "max_execution_time": 5.0,  # seconds
            "max_memory_mb": 50
        },
        "medium_dataset": {
            "rows": 1000,
            "max_execution_time": 30.0,  # seconds
            "max_memory_mb": 200
        },
        "large_dataset": {
            "rows": 10000,
            "max_execution_time": 300.0,  # seconds
            "max_memory_mb": 1000
        }
    }


# Session-scoped fixtures for expensive setup
@pytest.fixture(scope="session")
def reconciliation_test_environment():
    """Setup comprehensive test environment for reconciliation"""

    # Initialize test environment
    test_env = {
        "test_files_created": 0,
        "reconciliations_processed": 0,
        "start_time": None
    }

    import time
    test_env["start_time"] = time.time()

    yield test_env

    # Cleanup and reporting
    end_time = time.time()
    duration = end_time - test_env["start_time"]

    print(f"\n=== Reconciliation Test Session Summary ===")
    print(f"Total test files created: {test_env['test_files_created']}")
    print(f"Total reconciliations processed: {test_env['reconciliations_processed']}")
    print(f"Total session duration: {duration:.2f} seconds")


# Auto-use fixtures for common setup
@pytest.fixture(autouse=True)
def reconciliation_test_setup(reconciliation_test_environment):
    """Auto-setup for each reconciliation test"""
    # Pre-test setup
    reconciliation_test_environment["test_files_created"] += 1

    yield

    # Post-test cleanup (if needed)
    pass


# Parametrized fixtures for comprehensive testing
@pytest.fixture(params=[
    {"match_type": "equals", "tolerance": 0},
    {"match_type": "tolerance", "tolerance": 0.01},
    {"match_type": "tolerance", "tolerance": 0.05},
    {"match_type": "percentage", "tolerance": 1.0}
])
def reconciliation_rule_params(request):
    """Parametrized reconciliation rule configurations"""
    return request.param


@pytest.fixture(params=["Settled", "Pending", "Completed", "Failed"])
def status_filter_params(request):
    """Parametrized status filter values"""
    return request.param


# Helper functions available to all tests
def create_test_dataframe(data_type="file_a", num_rows=None):
    """Helper function to create test DataFrames"""
    if data_type == "file_a":
        base_data = RECONCILIATION_TEST_FILE_A
    else:
        base_data = RECONCILIATION_TEST_FILE_B

    df = pd.read_csv(io.StringIO(base_data))

    if num_rows:
        # Replicate data to reach desired number of rows
        multiplier = (num_rows // len(df)) + 1
        df = pd.concat([df] * multiplier, ignore_index=True)
        df = df.head(num_rows)

        # Make IDs unique
        if data_type == "file_a":
            df['Transaction_ID'] = df['Transaction_ID'] + '_' + df.index.astype(str)
        else:
            df['Statement_ID'] = df['Statement_ID'] + '_' + df.index.astype(str)

    return df


def assert_reconciliation_summary(result, expected_matches, expected_total_a, expected_total_b):
    """Helper function to assert reconciliation summary results"""
    assert result["summary"]["matched_pairs"] == expected_matches
    assert result["summary"]["total_file_a"] == expected_total_a
    assert result["summary"]["total_file_b"] == expected_total_b

    expected_percentage = (expected_matches / max(expected_total_a, expected_total_b)) * 100
    assert abs(result["summary"]["match_percentage"] - expected_percentage) < 0.1


def validate_reconciliation_config(config):
    """Helper function to validate reconciliation configuration"""
    required_fields = ["Files", "ReconciliationRules"]

    for field in required_fields:
        assert field in config, f"Missing required field: {field}"

    assert len(config["Files"]) >= 2, "At least 2 files required"
    assert len(config["ReconciliationRules"]) >= 1, "At least 1 reconciliation rule required"

    for rule in config["ReconciliationRules"]:
        assert "LeftFileColumn" in rule, "Missing LeftFileColumn in rule"
        assert "RightFileColumn" in rule, "Missing RightFileColumn in rule"
        assert "MatchType" in rule, "Missing MatchType in rule"


# Make helper functions available to all tests
@pytest.fixture
def reconciliation_helpers():
    """Provide helper functions to all tests"""
    return {
        "create_test_dataframe": create_test_dataframe,
        "assert_reconciliation_summary": assert_reconciliation_summary,
        "validate_reconciliation_config": validate_reconciliation_config
    }


# Custom pytest markers registration (already in your pytest.ini, but documented here)
"""
Custom markers used in reconciliation tests:

@pytest.mark.unit - Unit tests for individual components
@pytest.mark.integration - Integration tests for end-to-end workflows  
@pytest.mark.slow - Tests that take significant time (performance tests)
@pytest.mark.csv - CSV file specific tests
@pytest.mark.error - Error handling and edge case tests
@pytest.mark.validation - Data validation tests
@pytest.mark.smoke - Critical smoke tests
@pytest.mark.file_upload - File upload related tests
"""
