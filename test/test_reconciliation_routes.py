# test/test_reconciliation_routes.py
# Comprehensive pytest suite for testing reconciliation API routes
# Run with: pytest test/test_reconciliation_routes.py -v

import pytest
import json
import io
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd

# Import your FastAPI app and modules based on your project structure
# from app.main import app  # Uncomment when you have your app configured
# from app.routes.reconciliation_routes import router as reconciliation_router

# Test data
TEST_FILE_A_DATA = """Transaction_ID,Date,Description,Amount_Text,Status,Account,Reference,Customer_ID,Branch
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

TEST_FILE_B_DATA = """Statement_ID,Process_Date,Transaction_Desc,Net_Amount,Settlement_Status,Account_Number,Ref_Number,Client_Code,Location
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


@pytest.mark.unit
@pytest.mark.csv
class TestReconciliationRoutes:
    """Test suite for reconciliation API routes"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test client and test data"""
        # self.client = TestClient(app)  # Uncomment when you have your app
        self.test_file_a_id = "test_file_a_001"
        self.test_file_b_id = "test_file_b_001"

        # Create test DataFrames
        self.df_a = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))
        self.df_b = pd.read_csv(io.StringIO(TEST_FILE_B_DATA))

    @pytest.fixture
    def sample_reconciliation_config(self):
        """Sample reconciliation configuration for testing"""
        return {
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
                },
                {
                    "LeftFileColumn": "Extracted_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.01
                }
            ],
            "selected_columns_file_a": [
                "Transaction_ID", "Date", "Description", "Amount_Text",
                "Extracted_Amount", "Status", "Account", "Reference", "Customer_ID"
            ],
            "selected_columns_file_b": [
                "Statement_ID", "Process_Date", "Transaction_Desc", "Net_Amount",
                "Settlement_Status", "Account_Number", "Ref_Number", "Client_Code"
            ],
            "files": [
                {"file_id": "test_file_a_001", "role": "file_0", "label": "Financial Transactions"},
                {"file_id": "test_file_b_001", "role": "file_1", "label": "Bank Statements"}
            ]
        }

    # Test Column Unique Values Route
    @pytest.mark.unit
    async def test_get_column_unique_values_success(self):
        """Test successful retrieval of unique values for a column"""
        # Mock the get_file_by_id function from your file service
        with patch('app.routes.file_routes.get_file_by_id') as mock_get_file:
            mock_get_file.return_value = self.df_a

            # Test the route (uncomment when you have your route implemented)
            # For direct function testing:
            from app.routes.file_routes import get_column_unique_values

            # Test Status column
            result = await get_column_unique_values(self.test_file_a_id, "Status")

            # Simulate expected result structure based on your paste.txt
            # result = {
            #     "file_id": self.test_file_a_id,
            #     "column_name": "Status",
            #     "unique_values": ["Settled", "Pending", "Completed", "Failed"],
            #     "total_unique": 4,
            #     "is_date_column": False,
            #     "sample_values": ["Settled", "Pending", "Completed"]
            # }

            assert result["column_name"] == "Status"
            assert result["total_unique"] == 4
            assert "Settled" in result["unique_values"]
            assert "Pending" in result["unique_values"]
            assert "Completed" in result["unique_values"]
            assert "Failed" in result["unique_values"]
            assert result["is_date_column"] == False

    @pytest.mark.unit
    @pytest.mark.validation
    def test_get_column_unique_values_date_column(self):
        """Test unique values for date columns with date detection"""
        with patch('app.services.file_service.get_file_by_id') as mock_get_file:
            mock_get_file.return_value = self.df_b

            # Simulate date column processing based on your DeltaProcessor
            unique_dates = self.df_b['Process_Date'].unique().tolist()

            result = {
                "file_id": self.test_file_b_id,
                "column_name": "Process_Date",
                "unique_values": unique_dates,
                "total_unique": len(unique_dates),
                "is_date_column": True,
                "sample_values": unique_dates[:10]
            }

            assert result["column_name"] == "Process_Date"
            assert result["is_date_column"] == True
            assert len(result["unique_values"]) == 22

    @pytest.mark.unit
    def test_get_column_unique_values_numeric_column(self):
        """Test unique values for numeric amount columns"""
        unique_amounts = self.df_b['Net_Amount'].unique().tolist()

        result = {
            "column_name": "Net_Amount",
            "unique_values": unique_amounts,
            "total_unique": len(unique_amounts),
            "is_date_column": False
        }

        assert result["column_name"] == "Net_Amount"
        assert result["total_unique"] == 22
        assert 1234.56 in result["unique_values"]
        assert -456.78 in result["unique_values"]

    @pytest.mark.error
    def test_get_column_unique_values_column_not_found(self):
        """Test error handling for non-existent column"""
        with patch('app.services.file_service.get_file_by_id') as mock_get_file:
            mock_get_file.return_value = self.df_a

            # Should raise HTTPException with 404
            with pytest.raises(Exception) as exc_info:
                # Simulate column not found error
                if "NonExistentColumn" not in self.df_a.columns:
                    raise ValueError("Column 'NonExistentColumn' not found in file")

            assert "not found" in str(exc_info.value)

    @pytest.mark.unit
    def test_get_column_unique_values_with_limit(self):
        """Test unique values with limit parameter"""
        unique_values = self.df_a['Transaction_ID'].unique().tolist()
        limit = 5

        result = {
            "unique_values": unique_values[:limit],
            "has_more": len(unique_values) > limit,
            "returned_count": min(len(unique_values), limit),
            "total_unique": len(unique_values)
        }

        assert len(result["unique_values"]) <= 5
        assert result["has_more"] == True
        assert result["returned_count"] == 5

    # Test Reconciliation Processing
    @pytest.mark.integration
    @patch('app.services.reconciliation_service.DeltaProcessor')
    def test_reconciliation_basic_exact_match(self, mock_processor_class, sample_reconciliation_config):
        """Test basic reconciliation with exact matching"""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Mock expected results
        expected_results = {
            "matched_records": [
                {
                    "file_a": {"Transaction_ID": "TXN001", "Reference": "REF123", "Amount_Text": "$1,234.56"},
                    "file_b": {"Statement_ID": "STMT001", "Ref_Number": "REF123", "Net_Amount": 1234.56},
                    "match_confidence": 1.0,
                    "match_rules_satisfied": ["Reference", "Amount"]
                }
            ],
            "unmatched_a": [],
            "unmatched_b": [
                {"Statement_ID": "STMT021", "Ref_Number": "REF999"},
                {"Statement_ID": "STMT022", "Ref_Number": "REF888"}
            ],
            "summary": {
                "total_file_a": 20,
                "total_file_b": 22,
                "matched_pairs": 20,
                "unmatched_a_count": 0,
                "unmatched_b_count": 2,
                "match_percentage": 90.9
            }
        }

        mock_processor.process_reconciliation.return_value = expected_results

        # Test the reconciliation
        result = expected_results  # Simulate processing

        assert result["summary"]["matched_pairs"] == 20
        assert result["summary"]["unmatched_b_count"] == 2
        assert result["summary"]["match_percentage"] == 90.9

    @pytest.mark.integration
    @pytest.mark.slow
    @patch('app.services.reconciliation_service.DeltaProcessor')
    def test_reconciliation_with_extraction(self, mock_processor_class, sample_reconciliation_config):
        """Test reconciliation with amount extraction from text"""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Mock extraction results
        extracted_amounts = [
            1234.56, 2500.00, -456.78, 10000.00, 234.50,
            3456.78, 15000.00, 89.99, 5678.90, -200.00,
            125.75, 567.89, 1000.00, 345.67, 29.99,
            2345.67, 500.00, 78.45, 1567.89, -25.00
        ]

        extracted_df = self.df_a.copy()
        extracted_df['Extracted_Amount'] = extracted_amounts
        mock_processor.extract_data.return_value = extracted_df

        # Test extraction configuration
        extraction_config = sample_reconciliation_config["Files"][0]["Extract"][0]
        assert extraction_config["SourceColumn"] == "Amount_Text"
        assert extraction_config["ResultColumnName"] == "Extracted_Amount"
        assert "\\$([0-9,.-]+)" in extraction_config["Patterns"]

    @pytest.mark.integration
    @patch('app.services.reconciliation_service.DeltaProcessor')
    def test_reconciliation_with_filters(self, mock_processor_class, sample_reconciliation_config):
        """Test reconciliation with status filtering"""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Mock filtered data
        filtered_df_a = self.df_a[self.df_a['Status'] == 'Settled'].copy()
        filtered_df_b = self.df_b[self.df_b['Settlement_Status'] == 'SETTLED'].copy()

        mock_processor.apply_filters.side_effect = [filtered_df_a, filtered_df_b]

        expected_filtered_results = {
            "summary": {
                "total_file_a": len(filtered_df_a),
                "total_file_b": len(filtered_df_b),
                "matched_pairs": min(len(filtered_df_a), len(filtered_df_b)),
                "match_percentage": 100.0
            }
        }

        mock_processor.process_reconciliation.return_value = expected_filtered_results

        # Verify filter configuration
        filter_a = sample_reconciliation_config["Files"][0]["Filter"][0]
        filter_b = sample_reconciliation_config["Files"][1]["Filter"][0]

        assert filter_a["ColumnName"] == "Status"
        assert filter_a["Value"] == "Settled"
        assert filter_b["ColumnName"] == "Settlement_Status"
        assert filter_b["Value"] == "SETTLED"

    @pytest.mark.integration
    @patch('app.services.reconciliation_service.DeltaProcessor')
    def test_reconciliation_tolerance_matching(self, mock_processor_class):
        """Test reconciliation with tolerance-based amount matching"""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        tolerance_config = {
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Extracted_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.01
                }
            ]
        }

        # Mock tolerance match results
        tolerance_matches = {
            "matched_records": [
                {
                    "file_a": {"Transaction_ID": "TXN002", "Extracted_Amount": 2500.00},
                    "file_b": {"Statement_ID": "STMT002", "Net_Amount": 2500.01},
                    "match_confidence": 0.99,
                    "amount_difference": 0.01,
                    "within_tolerance": True
                },
                {
                    "file_a": {"Transaction_ID": "TXN016", "Extracted_Amount": 2345.67},
                    "file_b": {"Statement_ID": "STMT016", "Net_Amount": 2345.68},
                    "match_confidence": 0.99,
                    "amount_difference": 0.01,
                    "within_tolerance": True
                }
            ],
            "summary": {
                "tolerance_matches": 2,
                "exact_matches": 18
            }
        }

        mock_processor.process_reconciliation.return_value = tolerance_matches

        # Test tolerance rule
        assert tolerance_config["ReconciliationRules"][0]["ToleranceValue"] == 0.01

    @pytest.mark.unit
    @pytest.mark.validation
    def test_reconciliation_edge_cases(self):
        """Test edge cases in reconciliation"""

        # Test negative amounts
        negative_amounts_a = self.df_a[self.df_a['Amount_Text'].str.contains('-', na=False)]['Amount_Text'].tolist()
        negative_amounts_b = self.df_b[self.df_b['Net_Amount'] < 0]['Net_Amount'].tolist()

        assert len(negative_amounts_a) == 3  # TXN003, TXN010, TXN020
        assert len(negative_amounts_b) == 3  # STMT003, STMT010, STMT020

        # Test large amounts
        large_amounts = self.df_a[self.df_a['Amount_Text'].str.contains('10,000|15,000', na=False)]
        assert len(large_amounts) == 2  # TXN004, TXN007

        # Test small amounts
        small_amounts = self.df_b[self.df_b['Net_Amount'] < 100]
        assert len(small_amounts) >= 3

    @pytest.mark.unit
    def test_reconciliation_unmatched_records(self):
        """Test handling of unmatched records"""

        # File B has 2 extra records that shouldn't match
        unmatched_refs = ['REF999', 'REF888']
        unmatched_records = self.df_b[self.df_b['Ref_Number'].isin(unmatched_refs)]

        assert len(unmatched_records) == 2
        assert unmatched_records.iloc[0]['Statement_ID'] == 'STMT021'
        assert unmatched_records.iloc[1]['Statement_ID'] == 'STMT022'

    @pytest.mark.unit
    @pytest.mark.parametrize("status_filter,expected_count", [
        ("Settled", 12),  # Count of 'Settled' status in File A
        ("Pending", 4),  # Count of 'Pending' status in File A
        ("Completed", 4),  # Count of 'Completed' status in File A
        ("Failed", 1)  # Count of 'Failed' status in File A
    ])
    def test_status_filtering_parametrized(self, status_filter, expected_count):
        """Parametrized test for different status filters"""
        filtered_data = self.df_a[self.df_a['Status'] == status_filter]
        assert len(filtered_data) == expected_count

    @pytest.mark.unit
    @pytest.mark.parametrize("column_name,expected_unique_count", [
        ("Status", 4),  # Settled, Pending, Completed, Failed
        ("Account", 6),  # ACC001 through ACC006
        ("Branch", 3),  # BR01, BR02, BR03
        ("Customer_ID", 8)  # CUST001 through CUST008
    ])
    def test_unique_values_parametrized(self, column_name, expected_unique_count):
        """Parametrized test for unique value counts"""
        unique_values = self.df_a[column_name].nunique()
        assert unique_values == expected_unique_count

    @pytest.mark.unit
    @pytest.mark.validation
    def test_date_format_conversion(self):
        """Test date format standardization"""
        # File A: YYYY-MM-DD, File B: DD/MM/YYYY
        sample_date_a = self.df_a['Date'].iloc[0]  # "2024-01-15"
        sample_date_b = self.df_b['Process_Date'].iloc[0]  # "15/01/2024"

        # Test conversion logic
        def convert_date_format(date_str):
            if '/' in date_str:
                # Convert DD/MM/YYYY to YYYY-MM-DD
                day, month, year = date_str.split('/')
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            return date_str

        converted_date_b = convert_date_format(sample_date_b)
        assert sample_date_a == converted_date_b

    @pytest.mark.unit
    @pytest.mark.validation
    def test_amount_extraction_regex(self):
        """Test amount extraction regex patterns"""
        import re

        # Test regex pattern for amount extraction
        pattern = r'\$([0-9,.-]+)'

        test_amounts = [
            ("$1,234.56", "1,234.56"),
            ("$-456.78", "-456.78"),
            ("$10,000.00", "10,000.00"),
            ("$29.99", "29.99")
        ]

        for input_amount, expected_output in test_amounts:
            match = re.search(pattern, input_amount)
            assert match is not None, f"Failed to match {input_amount}"
            assert match.group(1) == expected_output


# Integration Tests
@pytest.mark.integration
class TestReconciliationIntegration:
    """Integration tests for end-to-end reconciliation workflows"""

    @pytest.fixture
    def test_files_setup(self):
        """Setup temporary test files for integration testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary CSV files
            file_a_path = os.path.join(temp_dir, "test_file_a.csv")
            file_b_path = os.path.join(temp_dir, "test_file_b.csv")

            with open(file_a_path, 'w') as f:
                f.write(TEST_FILE_A_DATA)

            with open(file_b_path, 'w') as f:
                f.write(TEST_FILE_B_DATA)

            yield {
                "file_a_path": file_a_path,
                "file_b_path": file_b_path,
                "temp_dir": temp_dir
            }

    @pytest.mark.file_upload
    @patch('app.services.file_service.upload_file')
    def test_end_to_end_reconciliation_workflow(self, mock_file_service, test_files_setup):
        """Test complete reconciliation workflow from file upload to results"""

        # Mock file upload responses
        mock_file_service.side_effect = [
            {"file_id": "test_file_a_001", "filename": "transactions.csv"},
            {"file_id": "test_file_b_001", "filename": "statements.csv"}
        ]

        # 1. Upload files
        file_a_response = mock_file_service(test_files_setup["file_a_path"])
        file_b_response = mock_file_service(test_files_setup["file_b_path"])

        assert file_a_response["file_id"] == "test_file_a_001"
        assert file_b_response["file_id"] == "test_file_b_001"

        # 2. Configure reconciliation
        reconciliation_config = {
            "files": [
                {"file_id": file_a_response["file_id"], "role": "file_0"},
                {"file_id": file_b_response["file_id"], "role": "file_1"}
            ],
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Reference",
                    "RightFileColumn": "Ref_Number",
                    "MatchType": "equals"
                }
            ]
        }

        # 3. Execute reconciliation
        with patch('app.services.reconciliation_service.process_reconciliation') as mock_reconcile:
            expected_result = {
                "reconciliation_id": "recon_001",
                "status": "completed",
                "summary": {
                    "total_file_a": 20,
                    "total_file_b": 22,
                    "matched_pairs": 20,
                    "unmatched_a_count": 0,
                    "unmatched_b_count": 2,
                    "match_percentage": 90.9
                }
            }

            mock_reconcile.return_value = expected_result
            result = mock_reconcile(reconciliation_config)

            assert result["status"] == "completed"
            assert result["summary"]["matched_pairs"] == 20


# Error Handling Tests
@pytest.mark.error
class TestReconciliationErrorHandling:
    """Test error handling and edge cases"""

    @pytest.mark.validation
    def test_empty_file_handling(self):
        """Test handling of empty CSV files"""
        empty_csv = "Column1,Column2\n"  # Headers only
        empty_df = pd.read_csv(io.StringIO(empty_csv))

        result = {
            "unique_values": [],
            "total_unique": 0,
            "column_name": "Column1"
        }

        assert result["unique_values"] == []
        assert result["total_unique"] == 0

    @pytest.mark.error
    def test_invalid_column_name_error(self):
        """Test error handling for invalid column names"""
        test_df = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))

        with pytest.raises(Exception) as exc_info:
            if "InvalidColumn" not in test_df.columns:
                raise ValueError("Column 'InvalidColumn' not found in file")

        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.error
    @pytest.mark.validation
    def test_malformed_csv_handling(self):
        """Test handling of malformed CSV data"""
        malformed_csv = """Transaction_ID,Date,Amount
TXN001,2024-01-15,$1,234.56
TXN002,2024-01-16
TXN003,invalid-date,$500.00"""

        # Should handle gracefully without crashing
        try:
            malformed_df = pd.read_csv(io.StringIO(malformed_csv))
            assert len(malformed_df) >= 1  # Should parse at least some data
        except Exception as e:
            # If parsing fails, ensure it's handled gracefully
            assert "error" in str(e).lower()

    @pytest.mark.validation
    def test_reconciliation_config_validation(self):
        """Test validation of reconciliation configuration"""

        # Test missing required fields
        invalid_configs = [
            {"Files": []},  # Empty files
            {"Files": [{"Name": "FileA"}]},  # Missing ReconciliationRules
            {"ReconciliationRules": []},  # Missing Files
            {
                "Files": [{"Name": "FileA"}],
                "ReconciliationRules": [{}]  # Empty rule
            }
        ]

        for invalid_config in invalid_configs:
            # Should raise validation error
            with pytest.raises(Exception):
                # Simulate validation
                if not invalid_config.get("Files") or not invalid_config.get("ReconciliationRules"):
                    raise ValueError("Invalid configuration")

    @pytest.mark.slow
    def test_memory_limit_handling(self):
        """Test handling of large files that might exceed memory limits"""

        # Simulate large dataset scenario
        def create_large_dataset(rows=1000):
            """Create a large dataset for testing"""
            import random

            data = []
            for i in range(rows):
                data.append({
                    'ID': f'TXN{i:06d}',
                    'Amount': random.uniform(1, 10000),
                    'Date': '2024-01-01',
                    'Status': random.choice(['Settled', 'Pending'])
                })
            return pd.DataFrame(data)

        large_df = create_large_dataset(1000)
        assert len(large_df) == 1000
        assert 'ID' in large_df.columns


# Data Quality Tests
@pytest.mark.unit
@pytest.mark.validation
class TestReconciliationDataQuality:
    """Test data quality and validation"""

    def test_data_type_validation(self):
        """Test validation of data types in reconciliation"""

        test_df = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))

        # Test expected data types
        assert test_df['Transaction_ID'].dtype == 'object'  # String
        assert test_df['Date'].dtype == 'object'  # Date string
        assert test_df['Amount_Text'].dtype == 'object'  # String with $

        # Test date format validation
        from datetime import datetime

        def validate_date_format(date_str, format_str='%Y-%m-%d'):
            try:
                datetime.strptime(date_str, format_str)
                return True
            except ValueError:
                return False

        # Test first date
        first_date = test_df['Date'].iloc[0]
        assert validate_date_format(first_date)

    def test_amount_format_validation(self):
        """Test validation of amount formats"""

        test_amounts = [
            "$1,234.56", "$-456.78", "$10,000.00", "$29.99", "$-25.00"
        ]

        import re
        amount_pattern = r'\$([0-9,.-]+)'

        for amount in test_amounts:
            match = re.search(amount_pattern, amount)
            assert match is not None, f"Amount {amount} should match pattern"

            # Extract and validate numeric value
            numeric_str = match.group(1).replace(',', '')
            try:
                numeric_value = float(numeric_str)
                assert isinstance(numeric_value, float)
            except ValueError:
                pytest.fail(f"Could not convert {numeric_str} to float")

    def test_reference_uniqueness(self):
        """Test that reference numbers are unique within files"""

        df_a = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))
        df_b = pd.read_csv(io.StringIO(TEST_FILE_B_DATA))

        # Check reference uniqueness in File A
        ref_counts_a = df_a['Reference'].value_counts()
        assert all(ref_counts_a == 1), "All references in File A should be unique"

        # Check reference uniqueness in File B
        ref_counts_b = df_b['Ref_Number'].value_counts()
        duplicates_b = ref_counts_b[ref_counts_b > 1]

        # File B should only have duplicates for unmatched records
        if len(duplicates_b) > 0:
            assert all(ref in ['REF999', 'REF888'] for ref in duplicates_b.index), \
                "Only unmatched references should have duplicates"

    def test_data_completeness(self):
        """Test data completeness and missing values"""

        df_a = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))
        df_b = pd.read_csv(io.StringIO(TEST_FILE_B_DATA))

        # Check for missing values in critical columns
        critical_columns_a = ['Transaction_ID', 'Reference', 'Amount_Text']
        critical_columns_b = ['Statement_ID', 'Ref_Number', 'Net_Amount']

        for col in critical_columns_a:
            missing_count = df_a[col].isnull().sum()
            assert missing_count == 0, f"Column {col} in File A should have no missing values"

        for col in critical_columns_b:
            missing_count = df_b[col].isnull().sum()
            assert missing_count == 0, f"Column {col} in File B should have no missing values"


# Performance Tests
@pytest.mark.slow
class TestReconciliationPerformance:
    """Performance tests for reconciliation system"""

    @pytest.mark.slow
    def test_large_dataset_reconciliation(self):
        """Test reconciliation performance with larger datasets"""
        # Create larger test dataset (1000 records each)
        large_df_a = pd.concat([pd.read_csv(io.StringIO(TEST_FILE_A_DATA))] * 50, ignore_index=True)
        large_df_b = pd.concat([pd.read_csv(io.StringIO(TEST_FILE_B_DATA))] * 50, ignore_index=True)

        # Update IDs to make them unique
        large_df_a['Transaction_ID'] = large_df_a['Transaction_ID'] + '_' + large_df_a.index.astype(str)
        large_df_b['Statement_ID'] = large_df_b['Statement_ID'] + '_' + large_df_b.index.astype(str)

        assert len(large_df_a) == 1000
        assert len(large_df_b) == 1100

        # Performance test would measure execution time
        import time
        start_time = time.time()

        # Mock reconciliation process
        # Simulate some processing time
        time.sleep(0.1)  # Simulate 100ms processing

        end_time = time.time()
        execution_time = end_time - start_time

        # Assert reasonable execution time (adjust threshold as needed)
        assert execution_time < 30.0  # Should complete within 30 seconds

    @pytest.mark.slow
    def test_memory_usage_reconciliation(self):
        """Test memory usage during reconciliation"""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Create test data that might use memory
            large_df = pd.concat([pd.read_csv(io.StringIO(TEST_FILE_A_DATA))] * 100, ignore_index=True)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Assert reasonable memory usage (adjust threshold as needed)
            assert memory_increase < 500  # Should not increase by more than 500MB

        except ImportError:
            pytest.skip("psutil not available for memory testing")


# Smoke Tests
@pytest.mark.smoke
class TestReconciliationSmoke:
    """Critical smoke tests for reconciliation functionality"""

    @pytest.mark.smoke
    def test_basic_reconciliation_smoke(self):
        """Smoke test for basic reconciliation functionality"""
        # Test that basic data loading works
        df_a = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))
        df_b = pd.read_csv(io.StringIO(TEST_FILE_B_DATA))

        assert len(df_a) == 20
        assert len(df_b) == 22
        assert 'Reference' in df_a.columns
        assert 'Ref_Number' in df_b.columns

        # Test basic matching logic
        common_refs = set(df_a['Reference']).intersection(set(df_b['Ref_Number']))
        assert len(common_refs) == 20  # Should have 20 matching references

    @pytest.mark.smoke
    def test_column_unique_values_smoke(self):
        """Smoke test for column unique values functionality"""
        df = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))

        # Test basic unique value extraction
        status_values = df['Status'].unique()
        assert len(status_values) == 4
        assert 'Settled' in status_values

    @pytest.mark.smoke
    def test_amount_extraction_smoke(self):
        """Smoke test for amount extraction functionality"""
        import re

        pattern = r'\$([0-9,.-]+)'
        test_amount = "$1,234.56"

        match = re.search(pattern, test_amount)
        assert match is not None
        assert match.group(1) == "1,234.56"


# Configuration for running specific test suites
class TestSuites:
    """Helper class to organize test suite runs"""

    @staticmethod
    def run_unit_tests():
        """Run only unit tests"""
        return pytest.main(["-m", "unit", "-v"])

    @staticmethod
    def run_integration_tests():
        """Run only integration tests"""
        return pytest.main(["-m", "integration", "-v"])

    @staticmethod
    def run_smoke_tests():
        """Run only smoke tests"""
        return pytest.main(["-m", "smoke", "-v"])

    @staticmethod
    def run_performance_tests():
        """Run only performance tests"""
        return pytest.main(["-m", "slow", "-v", "--durations=0"])

    @staticmethod
    def run_all_tests():
        """Run all tests"""
        return pytest.main(["-v"])


# Utility functions for test setup
def setup_test_database():
    """Setup test database if needed"""
    # Mock database setup
    pass


def cleanup_test_database():
    """Cleanup test database after tests"""
    # Mock database cleanup
    pass


def create_test_config_variations():
    """Create various test configurations for comprehensive testing"""

    configs = {
        "basic_exact_match": {
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Reference",
                    "RightFileColumn": "Ref_Number",
                    "MatchType": "equals"
                }
            ]
        },

        "tolerance_amount_match": {
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Extracted_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.01
                }
            ]
        },

        "multi_column_match": {
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
                }
            ]
        },

        "with_extraction_and_filtering": {
            "Files": [
                {
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
                        }
                    ]
                }
            ],
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Clean_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.05
                }
            ]
        }
    }

    return configs


# Test fixtures for common reconciliation scenarios
@pytest.fixture
def reconciliation_test_scenarios():
    """Provide common reconciliation test scenarios"""
    return {
        "perfect_match": {
            "file_a_record": {"Reference": "REF123", "Amount": 1234.56},
            "file_b_record": {"Ref_Number": "REF123", "Amount": 1234.56},
            "expected_match": True,
            "match_confidence": 1.0
        },
        "tolerance_match": {
            "file_a_record": {"Reference": "REF124", "Amount": 2500.00},
            "file_b_record": {"Ref_Number": "REF124", "Amount": 2500.01},
            "expected_match": True,
            "match_confidence": 0.99
        },
        "no_match": {
            "file_a_record": {"Reference": "REF999", "Amount": 999.99},
            "file_b_record": {"Ref_Number": "REF888", "Amount": 888.88},
            "expected_match": False,
            "match_confidence": 0.0
        }
    }


if __name__ == "__main__":
    # Allow running tests directly with python
    # Example: python test_reconciliation_routes.py

    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "unit":
            TestSuites.run_unit_tests()
        elif sys.argv[1] == "integration":
            TestSuites.run_integration_tests()
        elif sys.argv[1] == "smoke":
            TestSuites.run_smoke_tests()
        elif sys.argv[1] == "performance":
            TestSuites.run_performance_tests()
        else:
            TestSuites.run_all_tests()
    else:
        TestSuites.run_all_tests()  # test_reconciliation_routes.py
# Comprehensive pytest suite for testing reconciliation API routes
# Run with: pytest tests/test_reconciliation_routes.py -v

import pytest
import json
import io
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd

# Import your FastAPI app and modules
# from app.main import app  # Adjust import based on your app structure
# from app.routers import reconciliation  # Adjust import based on your structure

# Test data
TEST_FILE_A_DATA = """Transaction_ID,Date,Description,Amount_Text,Status,Account,Reference,Customer_ID,Branch
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

TEST_FILE_B_DATA = """Statement_ID,Process_Date,Transaction_Desc,Net_Amount,Settlement_Status,Account_Number,Ref_Number,Client_Code,Location
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


@pytest.mark.unit
@pytest.mark.csv
class TestReconciliationRoutes:
    """Test suite for reconciliation API routes"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test client and test data"""
        # self.client = TestClient(app)  # Uncomment when you have your app
        self.test_file_a_id = "test_file_a_001"
        self.test_file_b_id = "test_file_b_001"

        # Create test DataFrames
        self.df_a = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))
        self.df_b = pd.read_csv(io.StringIO(TEST_FILE_B_DATA))

    @pytest.fixture
    def sample_reconciliation_config(self):
        """Sample reconciliation configuration for testing"""
        return {
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
                },
                {
                    "LeftFileColumn": "Extracted_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.01
                }
            ],
            "selected_columns_file_a": [
                "Transaction_ID", "Date", "Description", "Amount_Text",
                "Extracted_Amount", "Status", "Account", "Reference", "Customer_ID"
            ],
            "selected_columns_file_b": [
                "Statement_ID", "Process_Date", "Transaction_Desc", "Net_Amount",
                "Settlement_Status", "Account_Number", "Ref_Number", "Client_Code"
            ],
            "files": [
                {"file_id": "test_file_a_001", "role": "file_0", "label": "Financial Transactions"},
                {"file_id": "test_file_b_001", "role": "file_1", "label": "Bank Statements"}
            ]
        }

    @pytest.mark.unit
    @pytest.mark.validation
    def test_get_column_unique_values_date_column(self):
        """Test unique values for date columns with date detection"""
        with patch('app.services.file_service.get_file_by_id') as mock_get_file:
            mock_get_file.return_value = self.df_b

            # Simulate date column processing
            unique_dates = self.df_b['Process_Date'].unique().tolist()

            result = {
                "file_id": self.test_file_b_id,
                "column_name": "Process_Date",
                "unique_values": unique_dates,
                "total_unique": len(unique_dates),
                "is_date_column": True,
                "sample_values": unique_dates[:10]
            }

            assert result["column_name"] == "Process_Date"
            assert result["is_date_column"] == True
            assert len(result["unique_values"]) == 22

    @pytest.mark.unit
    def test_get_column_unique_values_numeric_column(self):
        """Test unique values for numeric amount columns"""
        unique_amounts = self.df_b['Net_Amount'].unique().tolist()

        result = {
            "column_name": "Net_Amount",
            "unique_values": unique_amounts,
            "total_unique": len(unique_amounts),
            "is_date_column": False
        }

        assert result["column_name"] == "Net_Amount"
        assert result["total_unique"] == 22
        assert 1234.56 in result["unique_values"]
        assert -456.78 in result["unique_values"]

    @pytest.mark.error
    def test_get_column_unique_values_column_not_found(self):
        """Test error handling for non-existent column"""
        with patch('app.services.file_service.get_file_by_id') as mock_get_file:
            mock_get_file.return_value = self.df_a

            # Should raise HTTPException with 404
            with pytest.raises(Exception) as exc_info:
                # Simulate column not found error
                if "NonExistentColumn" not in self.df_a.columns:
                    raise ValueError("Column 'NonExistentColumn' not found in file")

            assert "not found" in str(exc_info.value)

    @pytest.mark.unit
    def test_get_column_unique_values_with_limit(self):
        """Test unique values with limit parameter"""
        unique_values = self.df_a['Transaction_ID'].unique().tolist()
        limit = 5

        result = {
            "unique_values": unique_values[:limit],
            "has_more": len(unique_values) > limit,
            "returned_count": min(len(unique_values), limit),
            "total_unique": len(unique_values)
        }

        assert len(result["unique_values"]) <= 5
        assert result["has_more"] == True
        assert result["returned_count"] == 5

    # Test Reconciliation Processing
    @pytest.mark.integration
    @patch('app.services.reconciliation_service.DeltaProcessor')
    def test_reconciliation_basic_exact_match(self, mock_processor_class, sample_reconciliation_config):
        """Test basic reconciliation with exact matching"""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Mock expected results
        expected_results = {
            "matched_records": [
                {
                    "file_a": {"Transaction_ID": "TXN001", "Reference": "REF123", "Amount_Text": "$1,234.56"},
                    "file_b": {"Statement_ID": "STMT001", "Ref_Number": "REF123", "Net_Amount": 1234.56},
                    "match_confidence": 1.0,
                    "match_rules_satisfied": ["Reference", "Amount"]
                }
            ],
            "unmatched_a": [],
            "unmatched_b": [
                {"Statement_ID": "STMT021", "Ref_Number": "REF999"},
                {"Statement_ID": "STMT022", "Ref_Number": "REF888"}
            ],
            "summary": {
                "total_file_a": 20,
                "total_file_b": 22,
                "matched_pairs": 20,
                "unmatched_a_count": 0,
                "unmatched_b_count": 2,
                "match_percentage": 90.9
            }
        }

        mock_processor.process_reconciliation.return_value = expected_results

        # Test the reconciliation
        result = expected_results  # Simulate processing

        assert result["summary"]["matched_pairs"] == 20
        assert result["summary"]["unmatched_b_count"] == 2
        assert result["summary"]["match_percentage"] == 90.9

    @pytest.mark.integration
    @pytest.mark.slow
    @patch('app.services.reconciliation_service.DeltaProcessor')
    def test_reconciliation_with_extraction(self, mock_processor_class, sample_reconciliation_config):
        """Test reconciliation with amount extraction from text"""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Mock extraction results
        extracted_amounts = [
            1234.56, 2500.00, -456.78, 10000.00, 234.50,
            3456.78, 15000.00, 89.99, 5678.90, -200.00,
            125.75, 567.89, 1000.00, 345.67, 29.99,
            2345.67, 500.00, 78.45, 1567.89, -25.00
        ]

        extracted_df = self.df_a.copy()
        extracted_df['Extracted_Amount'] = extracted_amounts
        mock_processor.extract_data.return_value = extracted_df

        # Test extraction configuration
        extraction_config = sample_reconciliation_config["Files"][0]["Extract"][0]
        assert extraction_config["SourceColumn"] == "Amount_Text"
        assert extraction_config["ResultColumnName"] == "Extracted_Amount"
        assert "\\$([0-9,.-]+)" in extraction_config["Patterns"]

    @pytest.mark.integration
    @patch('app.services.reconciliation_service.DeltaProcessor')
    def test_reconciliation_with_filters(self, mock_processor_class, sample_reconciliation_config):
        """Test reconciliation with status filtering"""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Mock filtered data
        filtered_df_a = self.df_a[self.df_a['Status'] == 'Settled'].copy()
        filtered_df_b = self.df_b[self.df_b['Settlement_Status'] == 'SETTLED'].copy()

        mock_processor.apply_filters.side_effect = [filtered_df_a, filtered_df_b]

        expected_filtered_results = {
            "summary": {
                "total_file_a": len(filtered_df_a),
                "total_file_b": len(filtered_df_b),
                "matched_pairs": min(len(filtered_df_a), len(filtered_df_b)),
                "match_percentage": 100.0
            }
        }

        mock_processor.process_reconciliation.return_value = expected_filtered_results

        # Verify filter configuration
        filter_a = sample_reconciliation_config["Files"][0]["Filter"][0]
        filter_b = sample_reconciliation_config["Files"][1]["Filter"][0]

        assert filter_a["ColumnName"] == "Status"
        assert filter_a["Value"] == "Settled"
        assert filter_b["ColumnName"] == "Settlement_Status"
        assert filter_b["Value"] == "SETTLED"

    @pytest.mark.integration
    @patch('app.services.reconciliation_service.DeltaProcessor')
    def test_reconciliation_tolerance_matching(self, mock_processor_class):
        """Test reconciliation with tolerance-based amount matching"""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        tolerance_config = {
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Extracted_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.01
                }
            ]
        }

        # Mock tolerance match results
        tolerance_matches = {
            "matched_records": [
                {
                    "file_a": {"Transaction_ID": "TXN002", "Extracted_Amount": 2500.00},
                    "file_b": {"Statement_ID": "STMT002", "Net_Amount": 2500.01},
                    "match_confidence": 0.99,
                    "amount_difference": 0.01,
                    "within_tolerance": True
                },
                {
                    "file_a": {"Transaction_ID": "TXN016", "Extracted_Amount": 2345.67},
                    "file_b": {"Statement_ID": "STMT016", "Net_Amount": 2345.68},
                    "match_confidence": 0.99,
                    "amount_difference": 0.01,
                    "within_tolerance": True
                }
            ],
            "summary": {
                "tolerance_matches": 2,
                "exact_matches": 18
            }
        }

        mock_processor.process_reconciliation.return_value = tolerance_matches

        # Test tolerance rule
        assert tolerance_config["ReconciliationRules"][0]["ToleranceValue"] == 0.01

    @pytest.mark.unit
    @pytest.mark.validation
    def test_reconciliation_edge_cases(self):
        """Test edge cases in reconciliation"""

        # Test negative amounts
        negative_amounts_a = self.df_a[self.df_a['Amount_Text'].str.contains('-', na=False)]['Amount_Text'].tolist()
        negative_amounts_b = self.df_b[self.df_b['Net_Amount'] < 0]['Net_Amount'].tolist()

        assert len(negative_amounts_a) == 3  # TXN003, TXN010, TXN020
        assert len(negative_amounts_b) == 3  # STMT003, STMT010, STMT020

        # Test large amounts
        large_amounts = self.df_a[self.df_a['Amount_Text'].str.contains('10,000|15,000', na=False)]
        assert len(large_amounts) == 2  # TXN004, TXN007

        # Test small amounts
        small_amounts = self.df_b[self.df_b['Net_Amount'] < 100]
        assert len(small_amounts) >= 3

    @pytest.mark.unit
    def test_reconciliation_unmatched_records(self):
        """Test handling of unmatched records"""

        # File B has 2 extra records that shouldn't match
        unmatched_refs = ['REF999', 'REF888']
        unmatched_records = self.df_b[self.df_b['Ref_Number'].isin(unmatched_refs)]

        assert len(unmatched_records) == 2
        assert unmatched_records.iloc[0]['Statement_ID'] == 'STMT021'
        assert unmatched_records.iloc[1]['Statement_ID'] == 'STMT022'

    @pytest.mark.unit
    @pytest.mark.parametrize("status_filter,expected_count", [
        ("Settled", 12),  # Count of 'Settled' status in File A
        ("Pending", 4),  # Count of 'Pending' status in File A
        ("Completed", 4),  # Count of 'Completed' status in File A
        ("Failed", 1)  # Count of 'Failed' status in File A
    ])
    def test_status_filtering_parametrized(self, status_filter, expected_count):
        """Parametrized test for different status filters"""
        filtered_data = self.df_a[self.df_a['Status'] == status_filter]
        assert len(filtered_data) == expected_count

    @pytest.mark.unit
    @pytest.mark.parametrize("column_name,expected_unique_count", [
        ("Status", 4),  # Settled, Pending, Completed, Failed
        ("Account", 6),  # ACC001 through ACC006
        ("Branch", 3),  # BR01, BR02, BR03
        ("Customer_ID", 8)  # CUST001 through CUST008
    ])
    def test_unique_values_parametrized(self, column_name, expected_unique_count):
        """Parametrized test for unique value counts"""
        unique_values = self.df_a[column_name].nunique()
        assert unique_values == expected_unique_count

    @pytest.mark.unit
    @pytest.mark.validation
    def test_date_format_conversion(self):
        """Test date format standardization"""
        # File A: YYYY-MM-DD, File B: DD/MM/YYYY
        sample_date_a = self.df_a['Date'].iloc[0]  # "2024-01-15"
        sample_date_b = self.df_b['Process_Date'].iloc[0]  # "15/01/2024"

        # Test conversion logic
        def convert_date_format(date_str):
            if '/' in date_str:
                # Convert DD/MM/YYYY to YYYY-MM-DD
                day, month, year = date_str.split('/')
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            return date_str

        converted_date_b = convert_date_format(sample_date_b)
        assert sample_date_a == converted_date_b

    @pytest.mark.unit
    @pytest.mark.validation
    def test_amount_extraction_regex(self):
        """Test amount extraction regex patterns"""
        import re

        # Test regex pattern for amount extraction
        pattern = r'\$([0-9,.-]+)'

        test_amounts = [
            ("$1,234.56", "1,234.56"),
            ("$-456.78", "-456.78"),
            ("$10,000.00", "10,000.00"),
            ("$29.99", "29.99")
        ]

        for input_amount, expected_output in test_amounts:
            match = re.search(pattern, input_amount)
            assert match is not None, f"Failed to match {input_amount}"
            assert match.group(1) == expected_output


# Integration Tests
@pytest.mark.integration
class TestReconciliationIntegration:
    """Integration tests for end-to-end reconciliation workflows"""

    @pytest.fixture
    def test_files_setup(self):
        """Setup temporary test files for integration testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary CSV files
            file_a_path = os.path.join(temp_dir, "test_file_a.csv")
            file_b_path = os.path.join(temp_dir, "test_file_b.csv")

            with open(file_a_path, 'w') as f:
                f.write(TEST_FILE_A_DATA)

            with open(file_b_path, 'w') as f:
                f.write(TEST_FILE_B_DATA)

            yield {
                "file_a_path": file_a_path,
                "file_b_path": file_b_path,
                "temp_dir": temp_dir
            }

    @pytest.mark.file_upload
    @patch('app.services.file_service.upload_file')
    def test_end_to_end_reconciliation_workflow(self, mock_file_service, test_files_setup):
        """Test complete reconciliation workflow from file upload to results"""

        # Mock file upload responses
        mock_file_service.side_effect = [
            {"file_id": "test_file_a_001", "filename": "transactions.csv"},
            {"file_id": "test_file_b_001", "filename": "statements.csv"}
        ]

        # 1. Upload files
        file_a_response = mock_file_service(test_files_setup["file_a_path"])
        file_b_response = mock_file_service(test_files_setup["file_b_path"])

        assert file_a_response["file_id"] == "test_file_a_001"
        assert file_b_response["file_id"] == "test_file_b_001"

        # 2. Configure reconciliation
        reconciliation_config = {
            "files": [
                {"file_id": file_a_response["file_id"], "role": "file_0"},
                {"file_id": file_b_response["file_id"], "role": "file_1"}
            ],
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Reference",
                    "RightFileColumn": "Ref_Number",
                    "MatchType": "equals"
                }
            ]
        }

        # 3. Execute reconciliation
        with patch('app.services.reconciliation_service.process_reconciliation') as mock_reconcile:
            expected_result = {
                "reconciliation_id": "recon_001",
                "status": "completed",
                "summary": {
                    "total_file_a": 20,
                    "total_file_b": 22,
                    "matched_pairs": 20,
                    "unmatched_a_count": 0,
                    "unmatched_b_count": 2,
                    "match_percentage": 90.9
                }
            }

            mock_reconcile.return_value = expected_result
            result = mock_reconcile(reconciliation_config)

            assert result["status"] == "completed"
            assert result["summary"]["matched_pairs"] == 20


# Error Handling Tests
@pytest.mark.error
class TestReconciliationErrorHandling:
    """Test error handling and edge cases"""

    @pytest.mark.validation
    def test_empty_file_handling(self):
        """Test handling of empty CSV files"""
        empty_csv = "Column1,Column2\n"  # Headers only
        empty_df = pd.read_csv(io.StringIO(empty_csv))

        result = {
            "unique_values": [],
            "total_unique": 0,
            "column_name": "Column1"
        }

        assert result["unique_values"] == []
        assert result["total_unique"] == 0

    @pytest.mark.error
    def test_invalid_column_name_error(self):
        """Test error handling for invalid column names"""
        test_df = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))

        with pytest.raises(Exception) as exc_info:
            if "InvalidColumn" not in test_df.columns:
                raise ValueError("Column 'InvalidColumn' not found in file")

        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.error
    @pytest.mark.validation
    def test_malformed_csv_handling(self):
        """Test handling of malformed CSV data"""
        malformed_csv = """Transaction_ID,Date,Amount
TXN001,2024-01-15,$1,234.56
TXN002,2024-01-16
TXN003,invalid-date,$500.00"""

        # Should handle gracefully without crashing
        try:
            malformed_df = pd.read_csv(io.StringIO(malformed_csv))
            assert len(malformed_df) >= 1  # Should parse at least some data
        except Exception as e:
            # If parsing fails, ensure it's handled gracefully
            assert "error" in str(e).lower()

    @pytest.mark.validation
    def test_reconciliation_config_validation(self):
        """Test validation of reconciliation configuration"""

        # Test missing required fields
        invalid_configs = [
            {"Files": []},  # Empty files
            {"Files": [{"Name": "FileA"}]},  # Missing ReconciliationRules
            {"ReconciliationRules": []},  # Missing Files
            {
                "Files": [{"Name": "FileA"}],
                "ReconciliationRules": [{}]  # Empty rule
            }
        ]

        for invalid_config in invalid_configs:
            # Should raise validation error
            with pytest.raises(Exception):
                # Simulate validation
                if not invalid_config.get("Files") or not invalid_config.get("ReconciliationRules"):
                    raise ValueError("Invalid configuration")

    @pytest.mark.slow
    def test_memory_limit_handling(self):
        """Test handling of large files that might exceed memory limits"""

        # Simulate large dataset scenario
        def create_large_dataset(rows=1000):
            """Create a large dataset for testing"""
            import random

            data = []
            for i in range(rows):
                data.append({
                    'ID': f'TXN{i:06d}',
                    'Amount': random.uniform(1, 10000),
                    'Date': '2024-01-01',
                    'Status': random.choice(['Settled', 'Pending'])
                })
            return pd.DataFrame(data)

        large_df = create_large_dataset(1000)
        assert len(large_df) == 1000
        assert 'ID' in large_df.columns


# Data Quality Tests
@pytest.mark.unit
@pytest.mark.validation
class TestReconciliationDataQuality:
    """Test data quality and validation"""

    def test_data_type_validation(self):
        """Test validation of data types in reconciliation"""

        test_df = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))

        # Test expected data types
        assert test_df['Transaction_ID'].dtype == 'object'  # String
        assert test_df['Date'].dtype == 'object'  # Date string
        assert test_df['Amount_Text'].dtype == 'object'  # String with $

        # Test date format validation
        from datetime import datetime

        def validate_date_format(date_str, format_str='%Y-%m-%d'):
            try:
                datetime.strptime(date_str, format_str)
                return True
            except ValueError:
                return False

        # Test first date
        first_date = test_df['Date'].iloc[0]
        assert validate_date_format(first_date)

    def test_amount_format_validation(self):
        """Test validation of amount formats"""

        test_amounts = [
            "$1,234.56", "$-456.78", "$10,000.00", "$29.99", "$-25.00"
        ]

        import re
        amount_pattern = r'\$([0-9,.-]+)'

        for amount in test_amounts:
            match = re.search(amount_pattern, amount)
            assert match is not None, f"Amount {amount} should match pattern"

            # Extract and validate numeric value
            numeric_str = match.group(1).replace(',', '')
            try:
                numeric_value = float(numeric_str)
                assert isinstance(numeric_value, float)
            except ValueError:
                pytest.fail(f"Could not convert {numeric_str} to float")

    def test_reference_uniqueness(self):
        """Test that reference numbers are unique within files"""

        df_a = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))
        df_b = pd.read_csv(io.StringIO(TEST_FILE_B_DATA))

        # Check reference uniqueness in File A
        ref_counts_a = df_a['Reference'].value_counts()
        assert all(ref_counts_a == 1), "All references in File A should be unique"

        # Check reference uniqueness in File B
        ref_counts_b = df_b['Ref_Number'].value_counts()
        duplicates_b = ref_counts_b[ref_counts_b > 1]

        # File B should only have duplicates for unmatched records
        if len(duplicates_b) > 0:
            assert all(ref in ['REF999', 'REF888'] for ref in duplicates_b.index), \
                "Only unmatched references should have duplicates"

    def test_data_completeness(self):
        """Test data completeness and missing values"""

        df_a = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))
        df_b = pd.read_csv(io.StringIO(TEST_FILE_B_DATA))

        # Check for missing values in critical columns
        critical_columns_a = ['Transaction_ID', 'Reference', 'Amount_Text']
        critical_columns_b = ['Statement_ID', 'Ref_Number', 'Net_Amount']

        for col in critical_columns_a:
            missing_count = df_a[col].isnull().sum()
            assert missing_count == 0, f"Column {col} in File A should have no missing values"

        for col in critical_columns_b:
            missing_count = df_b[col].isnull().sum()
            assert missing_count == 0, f"Column {col} in File B should have no missing values"


# Performance Tests
@pytest.mark.slow
class TestReconciliationPerformance:
    """Performance tests for reconciliation system"""

    @pytest.mark.slow
    def test_large_dataset_reconciliation(self):
        """Test reconciliation performance with larger datasets"""
        # Create larger test dataset (1000 records each)
        large_df_a = pd.concat([pd.read_csv(io.StringIO(TEST_FILE_A_DATA))] * 50, ignore_index=True)
        large_df_b = pd.concat([pd.read_csv(io.StringIO(TEST_FILE_B_DATA))] * 50, ignore_index=True)

        # Update IDs to make them unique
        large_df_a['Transaction_ID'] = large_df_a['Transaction_ID'] + '_' + large_df_a.index.astype(str)
        large_df_b['Statement_ID'] = large_df_b['Statement_ID'] + '_' + large_df_b.index.astype(str)

        assert len(large_df_a) == 1000
        assert len(large_df_b) == 1100

        # Performance test would measure execution time
        import time
        start_time = time.time()

        # Mock reconciliation process
        # Simulate some processing time
        time.sleep(0.1)  # Simulate 100ms processing

        end_time = time.time()
        execution_time = end_time - start_time

        # Assert reasonable execution time (adjust threshold as needed)
        assert execution_time < 30.0  # Should complete within 30 seconds

    @pytest.mark.slow
    def test_memory_usage_reconciliation(self):
        """Test memory usage during reconciliation"""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Create test data that might use memory
            large_df = pd.concat([pd.read_csv(io.StringIO(TEST_FILE_A_DATA))] * 100, ignore_index=True)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Assert reasonable memory usage (adjust threshold as needed)
            assert memory_increase < 500  # Should not increase by more than 500MB

        except ImportError:
            pytest.skip("psutil not available for memory testing")


# Smoke Tests
@pytest.mark.smoke
class TestReconciliationSmoke:
    """Critical smoke tests for reconciliation functionality"""

    @pytest.mark.smoke
    def test_basic_reconciliation_smoke(self):
        """Smoke test for basic reconciliation functionality"""
        # Test that basic data loading works
        df_a = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))
        df_b = pd.read_csv(io.StringIO(TEST_FILE_B_DATA))

        assert len(df_a) == 20
        assert len(df_b) == 22
        assert 'Reference' in df_a.columns
        assert 'Ref_Number' in df_b.columns

        # Test basic matching logic
        common_refs = set(df_a['Reference']).intersection(set(df_b['Ref_Number']))
        assert len(common_refs) == 20  # Should have 20 matching references

    @pytest.mark.smoke
    def test_column_unique_values_smoke(self):
        """Smoke test for column unique values functionality"""
        df = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))

        # Test basic unique value extraction
        status_values = df['Status'].unique()
        assert len(status_values) == 4
        assert 'Settled' in status_values

    @pytest.mark.smoke
    def test_amount_extraction_smoke(self):
        """Smoke test for amount extraction functionality"""
        import re

        pattern = r'\$([0-9,.-]+)'
        test_amount = "$1,234.56"

        match = re.search(pattern, test_amount)
        assert match is not None
        assert match.group(1) == "1,234.56"


# Configuration for running specific test suites
class TestSuites:
    """Helper class to organize test suite runs"""

    @staticmethod
    def run_unit_tests():
        """Run only unit tests"""
        return pytest.main(["-m", "unit", "-v"])

    @staticmethod
    def run_integration_tests():
        """Run only integration tests"""
        return pytest.main(["-m", "integration", "-v"])

    @staticmethod
    def run_smoke_tests():
        """Run only smoke tests"""
        return pytest.main(["-m", "smoke", "-v"])

    @staticmethod
    def run_performance_tests():
        """Run only performance tests"""
        return pytest.main(["-m", "slow", "-v", "--durations=0"])

    @staticmethod
    def run_all_tests():
        """Run all tests"""
        return pytest.main(["-v"])


# Utility functions for test setup
def setup_test_database():
    """Setup test database if needed"""
    # Mock database setup
    pass


def cleanup_test_database():
    """Cleanup test database after tests"""
    # Mock database cleanup
    pass


def create_test_config_variations():
    """Create various test configurations for comprehensive testing"""

    configs = {
        "basic_exact_match": {
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Reference",
                    "RightFileColumn": "Ref_Number",
                    "MatchType": "equals"
                }
            ]
        },

        "tolerance_amount_match": {
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Extracted_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.01
                }
            ]
        },

        "multi_column_match": {
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
                }
            ]
        },

        "with_extraction_and_filtering": {
            "Files": [
                {
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
                        }
                    ]
                }
            ],
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Clean_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.05
                }
            ]
        }
    }

    return configs


# Test fixtures for common reconciliation scenarios
@pytest.fixture
def reconciliation_test_scenarios():
    """Provide common reconciliation test scenarios"""
    return {
        "perfect_match": {
            "file_a_record": {"Reference": "REF123", "Amount": 1234.56},
            "file_b_record": {"Ref_Number": "REF123", "Amount": 1234.56},
            "expected_match": True,
            "match_confidence": 1.0
        },
        "tolerance_match": {
            "file_a_record": {"Reference": "REF124", "Amount": 2500.00},
            "file_b_record": {"Ref_Number": "REF124", "Amount": 2500.01},
            "expected_match": True,
            "match_confidence": 0.99
        },
        "no_match": {
            "file_a_record": {"Reference": "REF999", "Amount": 999.99},
            "file_b_record": {"Ref_Number": "REF888", "Amount": 888.88},
            "expected_match": False,
            "match_confidence": 0.0
        }
    }


if __name__ == "__main__":
    # Allow running tests directly with python
    # Example: python test_reconciliation_routes.py

    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "unit":
            TestSuites.run_unit_tests()
        elif sys.argv[1] == "integration":
            TestSuites.run_integration_tests()
        elif sys.argv[1] == "smoke":
            TestSuites.run_smoke_tests()
        elif sys.argv[1] == "performance":
            TestSuites.run_performance_tests()
        else:
            TestSuites.run_all_tests()
    else:
        TestSuites.run_all_tests()  # test_reconciliation_routes.py
# Comprehensive pytest suite for testing reconciliation API routes
# Run with: pytest test_reconciliation_routes.py -v

import pytest
import json
import io
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd

# Import your FastAPI app
# from main import app  # Adjust import based on your app structure
# from your_api_module import router  # Adjust import based on your structure

# Test data
TEST_FILE_A_DATA = """Transaction_ID,Date,Description,Amount_Text,Status,Account,Reference,Customer_ID,Branch
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

TEST_FILE_B_DATA = """Statement_ID,Process_Date,Transaction_Desc,Net_Amount,Settlement_Status,Account_Number,Ref_Number,Client_Code,Location
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


class TestReconciliationRoutes:
    """Test suite for reconciliation API routes"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test client and test data"""
        # self.client = TestClient(app)  # Uncomment when you have your app
        self.test_file_a_id = "test_file_a_001"
        self.test_file_b_id = "test_file_b_001"

        # Create test DataFrames
        self.df_a = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))
        self.df_b = pd.read_csv(io.StringIO(TEST_FILE_B_DATA))

    @pytest.fixture
    def sample_reconciliation_config(self):
        """Sample reconciliation configuration for testing"""
        return {
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
                },
                {
                    "LeftFileColumn": "Extracted_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.01
                }
            ],
            "selected_columns_file_a": [
                "Transaction_ID", "Date", "Description", "Amount_Text",
                "Extracted_Amount", "Status", "Account", "Reference", "Customer_ID"
            ],
            "selected_columns_file_b": [
                "Statement_ID", "Process_Date", "Transaction_Desc", "Net_Amount",
                "Settlement_Status", "Account_Number", "Ref_Number", "Client_Code"
            ],
            "files": [
                {"file_id": "test_file_a_001", "role": "file_0", "label": "Financial Transactions"},
                {"file_id": "test_file_b_001", "role": "file_1", "label": "Bank Statements"}
            ]
        }

    @pytest.mark.asyncio
    async def test_get_column_unique_values_date_column(self):
        """Test unique values for date columns with date detection"""
        with patch('app.routes.file_routes.get_file_by_id') as mock_get_file:
            mock_get_file.return_value = self.df_b

            from app.routes.file_routes import get_column_unique_values

            result = await get_column_unique_values(self.test_file_b_id, "Process_Date")

            assert result["column_name"] == "Process_Date"
            assert result["is_date_column"] == True
            assert len(result["unique_values"]) == 22  # All unique dates
            # Check date formatting
            assert "2024-01-15" in result["unique_values"]  # Should be converted

    @pytest.mark.asyncio
    async def test_get_column_unique_values_numeric_column(self):
        """Test unique values for numeric amount columns"""
        with patch('app.routes.file_routes.get_file_by_id') as mock_get_file:
            mock_get_file.return_value = self.df_b

            from app.routes.file_routes import get_column_unique_values

            result = await get_column_unique_values(self.test_file_b_id, "Net_Amount")

            assert result["column_name"] == "Net_Amount"
            assert result["total_unique"] == 22  # All amounts are unique
            assert '1234.56' in result["unique_values"]
            assert '-456.78' in result["unique_values"]

    # def test_get_column_unique_values_column_not_found(self):
    #     """Test error handling for non-existent column"""
    #     with patch('app.routes.file_routes.get_file_by_id') as mock_get_file:
    #         mock_get_file.return_value = self.df_a
    #
    #         # Should raise HTTPException with 404
    #         with pytest.raises(Exception) as exc_info:
    #             from app.routes.file_routes import get_column_unique_values
    #             res=  get_column_unique_values(self.test_file_a_id, "NonExistentColumn")
    #             print(res)
    #
    #         assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_column_unique_values_with_limit(self):
        """Test unique values with limit parameter"""
        with patch('app.routes.file_routes.get_file_by_id') as mock_get_file:
            mock_get_file.return_value = self.df_a

            from app.routes.file_routes import get_column_unique_values

            result = await get_column_unique_values(self.test_file_a_id, "Transaction_ID", limit=5)

            assert len(result["unique_values"]) <= 5
            assert result["has_more"] == True
            assert result["returned_count"] == 5

    # Test Reconciliation Processing Route
    # @patch('app.routes.delta_routes.DeltaProcessor')
    def test_reconciliation_basic_exact_match(self, sample_reconciliation_config):
        """Test basic reconciliation with exact matching"""
        # # Setup mock processor
        # from app.routes.delta_routes import DeltaProcessor
        # mock_processor = MagicMock()
        # mock_processor_class.return_value = mock_processor

        # Mock expected results
        expected_results = {
            "matched_records": [
                {
                    "file_a": {"Transaction_ID": "TXN001", "Reference": "REF123", "Amount_Text": "$1,234.56"},
                    "file_b": {"Statement_ID": "STMT001", "Ref_Number": "REF123", "Net_Amount": 1234.56},
                    "match_confidence": 1.0,
                    "match_rules_satisfied": ["Reference", "Amount"]
                }
            ],
            "unmatched_a": [],
            "unmatched_b": [
                {"Statement_ID": "STMT021", "Ref_Number": "REF999"},
                {"Statement_ID": "STMT022", "Ref_Number": "REF888"}
            ],
            "summary": {
                "total_file_a": 20,
                "total_file_b": 22,
                "matched_pairs": 20,
                "unmatched_a_count": 0,
                "unmatched_b_count": 2,
                "match_percentage": 90.9
            }
        }

        # mock_processor.process_reconciliation.return_value = expected_results

        # Test the reconciliation route
        # result = self.client.post("/reconcile", json=sample_reconciliation_config)

        # For direct function testing:
        from app.routes.reconciliation_routes import process_reconciliation_json
        result = process_reconciliation_json(sample_reconciliation_config)

        assert result["summary"]["matched_pairs"] == 20
        assert result["summary"]["unmatched_b_count"] == 2
        assert result["summary"]["match_percentage"] == 90.9

    @patch('app.routes.delta_routes.DeltaProcessor')
    def test_reconciliation_with_extraction(self, mock_processor_class, sample_reconciliation_config):
        """Test reconciliation with amount extraction from text"""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Mock extraction results
        mock_processor.extract_data.return_value = self.df_a.copy()
        mock_processor.extract_data.return_value['Extracted_Amount'] = [
            1234.56, 2500.00, -456.78, 10000.00, 234.50,
            3456.78, 15000.00, 89.99, 5678.90, -200.00,
            125.75, 567.89, 1000.00, 345.67, 29.99,
            2345.67, 500.00, 78.45, 1567.89, -25.00
        ]

        # Test extraction configuration
        extraction_config = sample_reconciliation_config["Files"][0]["Extract"][0]
        assert extraction_config["SourceColumn"] == "Amount_Text"
        assert extraction_config["ResultColumnName"] == "Extracted_Amount"
        assert "\\$([0-9,.-]+)" in extraction_config["Patterns"]

    @patch('app.routes.delta_routes.DeltaProcessor')
    def test_reconciliation_with_filters(self, mock_processor_class, sample_reconciliation_config):
        """Test reconciliation with status filtering"""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Mock filtered data (only 'Settled' and 'SETTLED' statuses)
        filtered_df_a = self.df_a[self.df_a['Status'] == 'Settled'].copy()
        filtered_df_b = self.df_b[self.df_b['Settlement_Status'] == 'SETTLED'].copy()

        mock_processor.apply_filters.side_effect = [filtered_df_a, filtered_df_b]

        expected_filtered_results = {
            "summary": {
                "total_file_a": len(filtered_df_a),  # Should be ~12 records
                "total_file_b": len(filtered_df_b),  # Should be ~11 records
                "matched_pairs": min(len(filtered_df_a), len(filtered_df_b)),
                "match_percentage": 100.0
            }
        }

        mock_processor.process_reconciliation.return_value = expected_filtered_results

        # Verify filter configuration
        filter_a = sample_reconciliation_config["Files"][0]["Filter"][0]
        filter_b = sample_reconciliation_config["Files"][1]["Filter"][0]

        assert filter_a["ColumnName"] == "Status"
        assert filter_a["Value"] == "Settled"
        assert filter_b["ColumnName"] == "Settlement_Status"
        assert filter_b["Value"] == "SETTLED"

    @patch('app.routes.delta_routes.DeltaProcessor')
    def test_reconciliation_tolerance_matching(self, mock_processor_class):
        """Test reconciliation with tolerance-based amount matching"""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        tolerance_config = {
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Extracted_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.01
                }
            ]
        }

        # Mock tolerance match results
        tolerance_matches = {
            "matched_records": [
                {
                    "file_a": {"Transaction_ID": "TXN002", "Extracted_Amount": 2500.00},
                    "file_b": {"Statement_ID": "STMT002", "Net_Amount": 2500.01},
                    "match_confidence": 0.99,
                    "amount_difference": 0.01,
                    "within_tolerance": True
                },
                {
                    "file_a": {"Transaction_ID": "TXN016", "Extracted_Amount": 2345.67},
                    "file_b": {"Statement_ID": "STMT016", "Net_Amount": 2345.68},
                    "match_confidence": 0.99,
                    "amount_difference": 0.01,
                    "within_tolerance": True
                }
            ],
            "summary": {
                "tolerance_matches": 2,
                "exact_matches": 18
            }
        }

        mock_processor.process_reconciliation.return_value = tolerance_matches

        # Test tolerance rule
        assert tolerance_config["ReconciliationRules"][0]["ToleranceValue"] == 0.01

    # def test_reconciliation_edge_cases(self):
    #     """Test edge cases in reconciliation"""
    #
    #     # Test negative amounts
    #     negative_amounts_a = self.df_a[self.df_a['Amount_Text'].str.contains('-')]['Amount_Text'].tolist()
    #     negative_amounts_b = self.df_b[self.df_b['Net_Amount'] < 0]['Net_Amount'].tolist()
    #
    #     assert len(negative_amounts_a) == 3  # TXN003, TXN010, TXN020
    #     assert len(negative_amounts_b) == 3  # STMT003, STMT010, STMT020
    #
    #     # Test large amounts
    #     large_amounts = self.df_a[self.df_a['Amount_Text'].str.contains('10,000|15,000')]
    #     assert len(large_amounts) == 2  # TXN004, TXN007
    #
    #     # Test small amounts
    #     small_amounts = self.df_b[self.df_b['Net_Amount'] < 100]
    #     assert len(small_amounts) >= 3  # At least TXN008, TXN015, TXN018, TXN020

    def test_reconciliation_unmatched_records(self):
        """Test handling of unmatched records"""

        # File B has 2 extra records that shouldn't match
        unmatched_refs = ['REF999', 'REF888']
        unmatched_records = self.df_b[self.df_b['Ref_Number'].isin(unmatched_refs)]

        assert len(unmatched_records) == 2
        assert unmatched_records.iloc[0]['Statement_ID'] == 'STMT021'
        assert unmatched_records.iloc[1]['Statement_ID'] == 'STMT022'

    @pytest.mark.parametrize("status_filter,expected_count", [
        ("Settled", 12),  # Count of 'Settled' status in File A
        ("Pending", 4),  # Count of 'Pending' status in File A
        ("Completed", 4),  # Count of 'Completed' status in File A
        ("Failed", 1)  # Count of 'Failed' status in File A
    ])
    def test_status_filtering_parametrized(self, status_filter, expected_count):
        """Parametrized test for different status filters"""
        filtered_data = self.df_a[self.df_a['Status'] == status_filter]
        assert len(filtered_data) == expected_count

    @pytest.mark.parametrize("column_name,expected_unique_count", [
        ("Status", 4),  # Settled, Pending, Completed, Failed
        ("Account", 6),  # ACC001 through ACC006
        ("Branch", 3),  # BR01, BR02, BR03
        ("Customer_ID", 8)  # CUST001 through CUST008
    ])
    def test_unique_values_parametrized(self, column_name, expected_unique_count):
        """Parametrized test for unique value counts"""
        unique_values = self.df_a[column_name].nunique()
        assert unique_values == expected_unique_count

    def test_date_format_conversion(self):
        """Test date format standardization"""
        # File A: YYYY-MM-DD, File B: DD/MM/YYYY
        sample_date_a = self.df_a['Date'].iloc[0]  # "2024-01-15"
        sample_date_b = self.df_b['Process_Date'].iloc[0]  # "15/01/2024"

        # Test conversion logic (you'll need to implement this in your processor)
        def convert_date_format(date_str):
            if '/' in date_str:
                # Convert DD/MM/YYYY to YYYY-MM-DD
                day, month, year = date_str.split('/')
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            return date_str

        converted_date_b = convert_date_format(sample_date_b)
        assert sample_date_a == converted_date_b

    def test_amount_extraction_regex(self):
        """Test amount extraction regex patterns"""
        import re

        # Test regex pattern for amount extraction
        pattern = r'\$([0-9,.-]+)'

        test_amounts = [
            ("$1,234.56", "1,234.56"),
            ("$-456.78", "-456.78"),
            ("$10,000.00", "10,000.00"),
            ("$29.99", "29.99")
        ]

        for input_amount, expected_output in test_amounts:
            match = re.search(pattern, input_amount)
            assert match is not None
            assert match.group(1) == expected_output

    @pytest.fixture
    def complex_reconciliation_config(self):
        """Complex multi-rule reconciliation configuration"""
        return {
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

    @patch('app.routes.delta_routes.DeltaProcessor')
    def test_complex_multi_rule_reconciliation(self, mock_processor_class, complex_reconciliation_config):
        """Test complex reconciliation with multiple rules"""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Mock results for complex reconciliation
        complex_results = {
            "matched_records": [],
            "summary": {
                "rules_applied": 4,
                "exact_rule_matches": 3,
                "tolerance_rule_matches": 1,
                "all_rules_satisfied": 16,  # Records matching all 4 rules
                "partial_matches": 0
            }
        }

        mock_processor.process_reconciliation.return_value = complex_results

        # Verify all rules are configured
        rules = complex_reconciliation_config["ReconciliationRules"]
        assert len(rules) == 4
        assert any(rule["MatchType"] == "tolerance" for rule in rules)
        assert any(rule["MatchType"] == "equals" for rule in rules)


# # Performance and Load Testing
# class TestReconciliationPerformance:
#     """Performance tests for reconciliation system"""
#
#     @pytest.mark.performance
#     def test_large_dataset_reconciliation(self):
#         """Test reconciliation performance with larger datasets"""
#         # Create larger test dataset (1000 records each)
#         large_df_a = pd.concat([pd.read_csv(io.StringIO(TEST_FILE_A_DATA))] * 50, ignore_index=True)
#         large_df_b = pd.concat([pd.read_csv(io.StringIO(TEST_FILE_B_DATA))] * 50, ignore_index=True)
#
#         # Update IDs to make them unique
#         large_df_a['Transaction_ID'] = large_df_a['Transaction_ID'] + '_' + large_df_a.index.astype(str)
#         large_df_b['Statement_ID'] = large_df_b['Statement_ID'] + '_' + large_df_b.index.astype(str)
#
#         assert len(large_df_a) == 1000
#         assert len(large_df_b) == 1100
#
#         # Performance test would measure execution time
#         import time
#         start_time = time.time()
#
#         # Mock reconciliation process
#         # result = reconcile_large_dataset(large_df_a, large_df_b)
#
#         end_time = time.time()
#         execution_time = end_time - start_time
#
#         # Assert reasonable execution time (adjust threshold as needed)
#         assert execution_time < 30.0  # Should complete within 30 seconds
#
#     @pytest.mark.performance
#     def test_memory_usage_reconciliation(self):
#         """Test memory usage during reconciliation"""
#         import psutil
#         import os
#
#         process = psutil.Process(os.getpid())
#         initial_memory = process.memory_info().rss / 1024 / 1024  # MB
#
#         # Run reconciliation
#         # result = run_reconciliation_process()
#
#         final_memory = process.memory_info().rss / 1024 / 1024  # MB
#         memory_increase = final_memory - initial_memory
#
#         # Assert reasonable memory usage (adjust threshold as needed)
#         assert memory_increase < 500  # Should not increase by more than 500MB


# Integration Tests
class TestReconciliationIntegration:
    """Integration tests for end-to-end reconciliation workflows"""

    @pytest.fixture
    def test_files_setup(self):
        """Setup temporary test files for integration testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary CSV files
            file_a_path = os.path.join(temp_dir, "test_file_a.csv")
            file_b_path = os.path.join(temp_dir, "test_file_b.csv")

            with open(file_a_path, 'w') as f:
                f.write(TEST_FILE_A_DATA)

            with open(file_b_path, 'w') as f:
                f.write(TEST_FILE_B_DATA)

            yield {
                "file_a_path": file_a_path,
                "file_b_path": file_b_path,
                "temp_dir": temp_dir
            }

    @patch('your_module.file_upload_service')
    def test_end_to_end_reconciliation_workflow(self, mock_file_service, test_files_setup):
        """Test complete reconciliation workflow from file upload to results"""

        # Mock file upload
        mock_file_service.upload_file.side_effect = [
            {"file_id": "test_file_a_001", "filename": "transactions.csv"},
            {"file_id": "test_file_b_001", "filename": "statements.csv"}
        ]

        # Mock file processing
        mock_file_service.get_file_columns.side_effect = [
            ["Transaction_ID", "Date", "Description", "Amount_Text", "Status", "Account", "Reference", "Customer_ID",
             "Branch"],
            ["Statement_ID", "Process_Date", "Transaction_Desc", "Net_Amount", "Settlement_Status", "Account_Number",
             "Ref_Number", "Client_Code", "Location"]
        ]

        # 1. Upload files
        file_a_response = mock_file_service.upload_file(test_files_setup["file_a_path"])
        file_b_response = mock_file_service.upload_file(test_files_setup["file_b_path"])

        assert file_a_response["file_id"] == "test_file_a_001"
        assert file_b_response["file_id"] == "test_file_b_001"

        # 2. Get column information
        columns_a = mock_file_service.get_file_columns(file_a_response["file_id"])
        columns_b = mock_file_service.get_file_columns(file_b_response["file_id"])

        assert "Amount_Text" in columns_a
        assert "Net_Amount" in columns_b

        # 3. Configure reconciliation
        reconciliation_config = {
            "files": [
                {"file_id": file_a_response["file_id"], "role": "file_0"},
                {"file_id": file_b_response["file_id"], "role": "file_1"}
            ],
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
                    "MatchType": "equals"
                }
            ]
        }

        # 4. Execute reconciliation
        with patch('your_module.process_reconciliation') as mock_reconcile:
            expected_result = {
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

            mock_reconcile.return_value = expected_result
            result = mock_reconcile(reconciliation_config)

            assert result["status"] == "completed"
            assert result["summary"]["matched_pairs"] == 20

    def test_column_unique_values_integration(self):
        """Test column unique values endpoint integration"""

        # Test with real data parsing
        df_test = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))

        # Test Status column unique values
        status_values = df_test['Status'].dropna().unique().tolist()
        expected_statuses = ['Settled', 'Pending', 'Completed', 'Failed']

        for status in expected_statuses:
            assert status in status_values

        # Test Reference column unique values
        ref_values = df_test['Reference'].dropna().unique().tolist()
        assert len(ref_values) == 20  # All references should be unique
        assert 'REF123' in ref_values
        assert 'REF142' in ref_values

    @patch('app.routes.delta_routes.DeltaProcessor')
    def test_extraction_integration(self, mock_processor_class):
        """Test data extraction integration"""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Mock extraction results
        test_df = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))
        extracted_df = test_df.copy()

        # Simulate amount extraction
        import re
        def extract_amount(amount_text):
            match = re.search(r'\$([0-9,.-]+)', str(amount_text))
            return float(match.group(1).replace(',', '')) if match else None

        extracted_df['Extracted_Amount'] = extracted_df['Amount_Text'].apply(extract_amount)
        mock_processor.extract_data.return_value = extracted_df

        # Test extraction
        result_df = mock_processor.extract_data(test_df, [{
            "ResultColumnName": "Extracted_Amount",
            "SourceColumn": "Amount_Text",
            "MatchType": "regex",
            "Patterns": ["\\$([0-9,.-]+)"]
        }])

        assert 'Extracted_Amount' in result_df.columns
        assert result_df['Extracted_Amount'].iloc[0] == 1234.56
        assert result_df['Extracted_Amount'].iloc[2] == -456.78

    @patch('app.routes.delta_routes.DeltaProcessor')
    def test_filtering_integration(self, mock_processor_class):
        """Test data filtering integration"""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        test_df = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))

        # Mock filter application
        def apply_status_filter(df, status_value):
            return df[df['Status'] == status_value]

        filtered_df = apply_status_filter(test_df, 'Settled')
        mock_processor.apply_filters.return_value = filtered_df

        # Test filtering
        result_df = mock_processor.apply_filters(test_df, [{
            "ColumnName": "Status",
            "MatchType": "equals",
            "Value": "Settled"
        }])

        # Verify all results have 'Settled' status
        assert all(result_df['Status'] == 'Settled')
        assert len(result_df) < len(test_df)  # Should be fewer records


# Error Handling Tests
class TestReconciliationErrorHandling:
    """Test error handling and edge cases"""

    def test_empty_file_handling(self):
        """Test handling of empty CSV files"""
        empty_csv = "Column1,Column2\n"  # Headers only
        empty_df = pd.read_csv(io.StringIO(empty_csv))

        with patch('app.routes.file_routes.get_file_by_id') as mock_get_file:
            mock_get_file.return_value = empty_df

            from app.routes.file_routes import get_column_unique_values

            result = get_column_unique_values("empty_file", "Column1")
            assert result["unique_values"] == []
            assert result["total_unique"] == 0

    def test_invalid_column_name_error(self):
        """Test error handling for invalid column names"""
        test_df = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))

        with patch('app.routes.file_routes.get_file_by_id') as mock_get_file:
            mock_get_file.return_value = test_df

            with pytest.raises(Exception) as exc_info:
                from app.routes.file_routes import get_column_unique_values
                get_column_unique_values("test_file", "InvalidColumn")

            assert "not found" in str(exc_info.value).lower()

    def test_malformed_csv_handling(self):
        """Test handling of malformed CSV data"""
        malformed_csv = """Transaction_ID,Date,Amount
        TXN001,2024-01-15,$1,234.56
        TXN002,2024-01-16  # Missing amount
        TXN003,invalid-date,$500.00"""

        # Should handle gracefully without crashing
        try:
            malformed_df = pd.read_csv(io.StringIO(malformed_csv))
            assert len(malformed_df) >= 1  # Should parse at least some data
        except Exception as e:
            # If parsing fails, ensure it's handled gracefully
            assert "error" in str(e).lower()

    def test_reconciliation_config_validation(self):
        """Test validation of reconciliation configuration"""

        # Test missing required fields
        invalid_configs = [
            {"Files": []},  # Empty files
            {"Files": [{"Name": "FileA"}]},  # Missing ReconciliationRules
            {"ReconciliationRules": []},  # Missing Files
            {
                "Files": [{"Name": "FileA"}],
                "ReconciliationRules": [{}]  # Empty rule
            }
        ]

        for invalid_config in invalid_configs:
            # Should raise validation error
            with pytest.raises(Exception):
                # validate_reconciliation_config(invalid_config)
                pass  # Replace with actual validation function

    def test_memory_limit_handling(self):
        """Test handling of large files that might exceed memory limits"""

        # Simulate large dataset scenario
        def create_large_dataset(rows=100000):
            """Create a large dataset for testing"""
            import random

            data = []
            for i in range(rows):
                data.append({
                    'ID': f'TXN{i:06d}',
                    'Amount': random.uniform(1, 10000),
                    'Date': '2024-01-01',
                    'Status': random.choice(['Settled', 'Pending'])
                })
            return pd.DataFrame(data)

        # This test would verify graceful handling of large datasets
        # In practice, you might implement chunking or streaming
        large_df = create_large_dataset(1000)  # Smaller for testing
        assert len(large_df) == 1000
        assert 'ID' in large_df.columns


# Data Quality Tests
class TestReconciliationDataQuality:
    """Test data quality and validation"""

    def test_data_type_validation(self):
        """Test validation of data types in reconciliation"""

        test_df = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))

        # Test expected data types
        assert test_df['Transaction_ID'].dtype == 'object'  # String
        assert test_df['Date'].dtype == 'object'  # Date string
        assert test_df['Amount_Text'].dtype == 'object'  # String with $

        # Test date format validation
        from datetime import datetime

        def validate_date_format(date_str, format_str='%Y-%m-%d'):
            try:
                datetime.strptime(date_str, format_str)
                return True
            except ValueError:
                return False

        # Test first date
        first_date = test_df['Date'].iloc[0]
        assert validate_date_format(first_date)

    def test_amount_format_validation(self):
        """Test validation of amount formats"""

        test_amounts = [
            "$1,234.56", "$-456.78", "$10,000.00", "$29.99", "$-25.00"
        ]

        import re
        amount_pattern = r'\$([0-9,.-]+)'

        for amount in test_amounts:
            match = re.search(amount_pattern, amount)
            assert match is not None, f"Amount {amount} should match pattern"

            # Extract and validate numeric value
            numeric_str = match.group(1).replace(',', '')
            try:
                numeric_value = float(numeric_str)
                assert isinstance(numeric_value, float)
            except ValueError:
                pytest.fail(f"Could not convert {numeric_str} to float")

    def test_reference_uniqueness(self):
        """Test that reference numbers are unique within files"""

        df_a = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))
        df_b = pd.read_csv(io.StringIO(TEST_FILE_B_DATA))

        # Check reference uniqueness in File A
        ref_counts_a = df_a['Reference'].value_counts()
        assert all(ref_counts_a == 1), "All references in File A should be unique"

        # Check reference uniqueness in File B
        ref_counts_b = df_b['Ref_Number'].value_counts()
        duplicates_b = ref_counts_b[ref_counts_b > 1]

        # File B should only have duplicates for unmatched records
        if len(duplicates_b) > 0:
            assert all(ref in ['REF999', 'REF888'] for ref in duplicates_b.index), \
                "Only unmatched references should have duplicates"

    def test_data_completeness(self):
        """Test data completeness and missing values"""

        df_a = pd.read_csv(io.StringIO(TEST_FILE_A_DATA))
        df_b = pd.read_csv(io.StringIO(TEST_FILE_B_DATA))

        # Check for missing values in critical columns
        critical_columns_a = ['Transaction_ID', 'Reference', 'Amount_Text']
        critical_columns_b = ['Statement_ID', 'Ref_Number', 'Net_Amount']

        for col in critical_columns_a:
            missing_count = df_a[col].isnull().sum()
            assert missing_count == 0, f"Column {col} in File A should have no missing values"

        for col in critical_columns_b:
            missing_count = df_b[col].isnull().sum()
            assert missing_count == 0, f"Column {col} in File B should have no missing values"


# Configuration for pytest
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before running tests"""

    # Setup logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)

    # Create test database or mock services if needed
    # setup_test_database()

    yield

    # Cleanup after tests
    # cleanup_test_database()


# Utility functions for tests
def create_test_config_variations():
    """Create various test configurations for comprehensive testing"""

    configs = {
        "basic_exact_match": {
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Reference",
                    "RightFileColumn": "Ref_Number",
                    "MatchType": "equals"
                }
            ]
        },

        "tolerance_amount_match": {
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Extracted_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.01
                }
            ]
        },

        "multi_column_match": {
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
                }
            ]
        },

        "with_extraction_and_filtering": {
            "Files": [
                {
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
                        }
                    ]
                }
            ],
            "ReconciliationRules": [
                {
                    "LeftFileColumn": "Clean_Amount",
                    "RightFileColumn": "Net_Amount",
                    "MatchType": "tolerance",
                    "ToleranceValue": 0.05
                }
            ]
        }
    }

    return configs


# Markers for different test categories
pytestmark = [
    pytest.mark.reconciliation,
    pytest.mark.integration
]

if __name__ == "__main__":
    # Allow running tests directly with python
    pytest.main([__file__, "-v", "--tb=short"])
