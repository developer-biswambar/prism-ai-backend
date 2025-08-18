# tests/test_file_routes.py
import io
from unittest.mock import patch, MagicMock

import pytest


class TestFileUpload:
    """Test file upload functionality"""

    @pytest.mark.file_upload
    def test_upload_csv_file_success(self, client, sample_csv_file, mock_storage):
        """Test successful CSV file upload"""
        response = client.post(
            "/files/upload",
            files={"file": ("test.csv", sample_csv_file, "text/csv")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "File uploaded successfully" in data["message"]
        assert data["data"]["filename"] == "test.csv"
        assert data["data"]["file_type"] == "csv"
        assert data["data"]["total_rows"] == 5
        assert data["data"]["total_columns"] == 4
        assert len(mock_storage) == 1

    @pytest.mark.file_upload
    def test_upload_excel_file_success(self, client, sample_excel_file, mock_storage):
        """Test successful Excel file upload"""
        response = client.post(
            "/files/upload",
            files={"file": ("test.xlsx", sample_excel_file,
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["filename"] == "test.xlsx"
        assert data["data"]["file_type"] == "excel"
        assert data["data"]["total_rows"] == 4  # First sheet (Sales)
        assert len(mock_storage) == 1

    @pytest.mark.file_upload
    def test_upload_with_custom_name(self, client, sample_csv_file, mock_storage):
        """Test file upload with custom name"""
        custom_name = "My Custom File"
        response = client.post(
            "/files/upload",
            files={"file": ("test.csv", sample_csv_file, "text/csv")},
            data={"custom_name": custom_name}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["custom_name"] == custom_name
        assert custom_name in data["message"]

    @pytest.mark.file_upload
    def test_upload_excel_with_sheet_selection(self, client, sample_excel_file, mock_storage):
        """Test Excel upload with specific sheet selection"""
        response = client.post(
            "/files/upload",
            files={"file": ("test.xlsx", sample_excel_file,
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            data={"sheet_name": "Employees"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["sheet_name"] == "Employees"
        assert data["data"]["total_rows"] == 5  # Employee sheet has 5 rows
        assert "Employees" in data["message"]

    @pytest.mark.error
    @pytest.mark.file_upload
    def test_upload_invalid_file_type(self, client, invalid_file):
        """Test upload with invalid file type"""
        response = client.post(
            "/files/upload",
            files={"file": ("test.txt", invalid_file, "text/plain")}
        )

        assert response.status_code == 400
        assert "Only CSV and Excel files are supported" in response.json()["detail"]

    @pytest.mark.error
    @pytest.mark.file_upload
    def test_upload_file_too_large(self, client, mock_env_vars):
        """Test upload with file exceeding size limit"""
        # Create a file larger than the 10MB limit set in mock_env_vars
        large_content = "x" * (11 * 1024 * 1024)  # 11MB
        large_file = io.BytesIO(large_content.encode('utf-8'))

        response = client.post(
            "/files/upload",
            files={"file": ("large.csv", large_file, "text/csv")}
        )

        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]

    @pytest.mark.error
    @pytest.mark.file_upload
    def test_upload_corrupted_excel(self, client, corrupted_excel_file):
        """Test upload with corrupted Excel file"""
        response = client.post(
            "/files/upload",
            files={"file": ("corrupted.xlsx", corrupted_excel_file,
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        )

        assert response.status_code == 400
        assert "Error reading file" in response.json()["detail"]

    @pytest.mark.error
    @pytest.mark.file_upload
    def test_upload_duplicate_custom_name(self, client, sample_csv_file, uploaded_file_with_data):
        """Test upload with duplicate custom name"""
        existing_name = "Existing File"

        # Update existing file to have custom name
        file_id = list(uploaded_file_with_data.keys())[0] if isinstance(uploaded_file_with_data,
                                                                        dict) else "test-file-id-123"
        uploaded_file_with_data["info"]["custom_name"] = existing_name

        response = client.post(
            "/files/upload",
            files={"file": ("new.csv", sample_csv_file, "text/csv")},
            data={"custom_name": existing_name}
        )

        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    @pytest.mark.file_upload
    def test_upload_empty_csv(self, client, empty_csv_file):
        """Test upload with empty CSV file"""
        response = client.post(
            "/files/upload",
            files={"file": ("empty.csv", empty_csv_file, "text/csv")}
        )

        assert response.status_code == 400
        assert "Error reading file" in response.json()["detail"]

    @pytest.mark.file_upload
    def test_upload_csv_with_special_characters(self, client, csv_with_special_chars):
        """Test upload with CSV containing special characters"""
        response = client.post(
            "/files/upload",
            files={"file": ("special.csv", csv_with_special_chars, "text/csv")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["total_rows"] == 3


class TestFileValidation:
    """Test file validation functionality"""

    @pytest.mark.validation
    def test_validate_available_filename(self, client, mock_storage):
        """Test validation of available filename"""
        response = client.post(
            "/files/validate-name",
            json={"filename": "new_file.csv"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["isValid"] is True
        assert "available" in data["message"]

    @pytest.mark.validation
    def test_validate_existing_filename(self, client, uploaded_file_with_data):
        """Test validation of existing filename"""
        response = client.post(
            "/files/validate-name",
            json={"filename": "test.csv"}  # This filename exists in the fixture
        )

        assert response.status_code == 200
        data = response.json()
        assert data["isValid"] is False
        assert "already exists" in data["error"]

    @pytest.mark.validation
    def test_validate_empty_filename(self, client):
        """Test validation of empty filename"""
        response = client.post(
            "/files/validate-name",
            json={"filename": ""}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["isValid"] is False
        assert "required" in data["error"]

    @pytest.mark.validation
    def test_validate_whitespace_filename(self, client):
        """Test validation of whitespace-only filename"""
        response = client.post(
            "/files/validate-name",
            json={"filename": "   "}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["isValid"] is False
        assert "required" in data["error"]


class TestExcelSheetAnalysis:
    """Test Excel sheet analysis functionality"""

    @pytest.mark.excel
    def test_analyze_excel_sheets_success(self, client, sample_excel_file):
        """Test successful Excel sheet analysis"""
        response = client.post(
            "/files/analyze-sheets",
            files={"file": ("test.xlsx", sample_excel_file,
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["sheets"]) == 3  # Sales, Employees, Empty

        # Check Sales sheet
        sales_sheet = next(sheet for sheet in data["sheets"] if sheet["sheet_name"] == "Sales")
        assert sales_sheet["row_count"] == 4
        assert sales_sheet["column_count"] == 3
        assert "product" in sales_sheet["columns"]

    @pytest.mark.excel
    @pytest.mark.error
    def test_analyze_non_excel_file(self, client, sample_csv_file):
        """Test Excel analysis on non-Excel file"""
        response = client.post(
            "/files/analyze-sheets",
            files={"file": ("test.csv", sample_csv_file, "text/csv")}
        )

        assert response.status_code == 400
        assert "Only Excel files are supported" in response.json()["detail"]

    @pytest.mark.excel
    @pytest.mark.error
    def test_analyze_corrupted_excel(self, client, corrupted_excel_file):
        """Test Excel analysis on corrupted file"""
        response = client.post(
            "/files/analyze-sheets",
            files={"file": ("corrupted.xlsx", corrupted_excel_file,
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        )

        assert response.status_code == 200
        data = response.json()
        # The endpoint returns success=True but with empty sheets when extraction fails
        assert data["success"] is True
        assert len(data["sheets"]) == 0
        assert "Found 0 sheets" in data["message"]


class TestFileRetrieval:
    """Test file retrieval functionality"""

    def test_list_files_empty(self, client, mock_storage):
        """Test listing files when storage is empty"""
        response = client.get("/files/")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["files"]) == 0
        assert data["data"]["summary"]["total_files"] == 0

    def test_list_files_with_data(self, client, uploaded_file_with_data):
        """Test listing files with uploaded data"""
        response = client.get("/files/")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["files"]) == 1
        assert data["data"]["summary"]["total_files"] == 1
        assert data["data"]["summary"]["total_rows"] == 5

    def test_get_file_info_basic(self, client, uploaded_file_with_data):
        """Test getting basic file info"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["info"]["file_id"] == file_id
        assert "sample_data" not in data["data"]

    def test_get_file_info_with_sample(self, client, uploaded_file_with_data):
        """Test getting file info with sample data"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}?include_sample=true&sample_rows=3")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "sample_data" in data["data"]
        assert len(data["data"]["sample_data"]) == 3
        assert "column_statistics" in data["data"]

    @pytest.mark.error
    def test_get_nonexistent_file(self, client, mock_storage):
        """Test getting info for nonexistent file"""
        response = client.get("/files/nonexistent-id")

        assert response.status_code == 404
        assert "File not found" in response.json()["detail"]


class TestFilePreview:
    """Test file preview functionality"""

    def test_preview_file_default(self, client, uploaded_file_with_data):
        """Test file preview with default parameters"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}/preview")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["rows"]) == 5  # All rows in sample data
        assert data["data"]["pagination"]["total_rows"] == 5

    def test_preview_file_with_pagination(self, client, uploaded_file_with_data):
        """Test file preview with pagination"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}/preview?start_row=1&num_rows=2")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["rows"]) == 2
        assert data["data"]["pagination"]["start_row"] == 1
        assert data["data"]["pagination"]["returned_rows"] == 2

    def test_preview_file_with_columns(self, client, uploaded_file_with_data):
        """Test file preview with specific columns"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}/preview?columns=name,age")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["rows"][0].keys()) == 2
        assert "name" in data["data"]["rows"][0]
        assert "age" in data["data"]["rows"][0]

    @pytest.mark.error
    def test_preview_nonexistent_file(self, client, mock_storage):
        """Test preview for nonexistent file"""
        response = client.get("/files/nonexistent-id/preview")

        assert response.status_code == 404
        assert "File not found" in response.json()["detail"]


class TestFileDeletion:
    """Test file deletion functionality"""

    def test_delete_file_success(self, client, uploaded_file_with_data):
        """Test successful file deletion"""
        file_id = "test-file-id-123"
        response = client.delete(f"/files/{file_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deleted successfully" in data["message"]

    def test_bulk_delete_success(self, client, mock_storage):
        """Test successful bulk deletion"""
        # Add multiple files to storage
        for i in range(3):
            file_id = f"test-file-{i}"
            mock_storage[file_id] = {
                "info": {"filename": f"test{i}.csv", "total_rows": 10},
                "data": None
            }

        file_ids = ["test-file-0", "test-file-1", "test-file-2"]
        response = client.post("/files/bulk-delete", json={"file_ids": file_ids})

        assert response.status_code == 200
        assert len(mock_storage) == 0

    @pytest.mark.error
    def test_delete_nonexistent_file(self, client, mock_storage):
        """Test deletion of nonexistent file"""
        response = client.delete("/files/nonexistent-id")

        assert response.status_code == 404
        assert "File not found" in response.json()["detail"]


class TestExcelSheetOperations:
    """Test Excel sheet-specific operations"""

    def test_get_file_sheets_success(self, client, mock_storage):
        """Test getting sheets for Excel file"""
        # Mock Excel file data
        file_id = "excel-file-id"
        mock_storage[file_id] = {
            "info": {
                "filename": "test.xlsx",
                "is_excel": True,
                "selected_sheet": "Sheet1",
                "available_sheets": [
                    {"sheet_name": "Sheet1", "row_count": 10},
                    {"sheet_name": "Sheet2", "row_count": 5}
                ]
            },
            "data": None
        }

        response = client.get(f"/files/{file_id}/sheets")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["current_sheet"] == "Sheet1"
        assert len(data["data"]["available_sheets"]) == 2

    @pytest.mark.error
    def test_get_sheets_for_csv_file(self, client, uploaded_file_with_data):
        """Test getting sheets for CSV file (should fail)"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}/sheets")

        assert response.status_code == 400
        assert "not an Excel file" in response.json()["detail"]

    def test_select_sheet_success(self, client, mock_storage):
        """Test selecting different sheet in Excel file"""
        # Mock Excel file with raw content
        file_id = "excel-file-id"
        mock_storage[file_id] = {
            "info": {
                "filename": "test.xlsx",
                "is_excel": True,
                "available_sheets": [
                    {"sheet_name": "Sheet1"},
                    {"sheet_name": "Sheet2"}
                ]
            },
            "data": None,
            "raw_content": b"mock excel content"
        }

        with patch('pandas.read_excel') as mock_read_excel:
            mock_df = MagicMock()
            mock_df.__len__.return_value = 15
            mock_df.columns = ["col1", "col2"]
            mock_df.dtypes = {"col1": "object", "col2": "int64"}
            mock_read_excel.return_value = mock_df

            response = client.post(
                f"/files/{file_id}/select-sheet",
                json={"sheet_name": "Sheet2"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["selected_sheet"] == "Sheet2"

    @pytest.mark.error
    def test_select_nonexistent_sheet(self, client, mock_storage):
        """Test selecting non-existent sheet"""
        file_id = "excel-file-id"
        mock_storage[file_id] = {
            "info": {
                "filename": "test.xlsx",
                "is_excel": True,
                "available_sheets": [{"sheet_name": "Sheet1"}]
            },
            "data": None
        }

        response = client.post(
            f"/files/{file_id}/select-sheet",
            json={"sheet_name": "NonExistentSheet"}
        )

        assert response.status_code == 400
        assert "not found" in response.json()["detail"]


class TestErrorHandling:
    """Test error handling across all endpoints"""

    @pytest.mark.error
    def test_malformed_json_request(self, client):
        """Test handling of malformed JSON requests"""
        response = client.post(
            "/files/validate-name",
            data="invalid json content",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    @pytest.mark.error
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields"""
        response = client.post("/files/bulk-delete", json={})

        assert response.status_code == 422

    @pytest.mark.error
    @patch('app.routes.file_routes.logger')
    def test_pandas_read_error_handling(self, mock_logger, client, sample_csv_file):
        """Test that pandas read errors are properly handled and logged"""
        with patch('pandas.read_csv', side_effect=Exception("Pandas read error")):
            response = client.post(
                "/files/upload",
                files={"file": ("test.csv", sample_csv_file, "text/csv")}
            )

            # Your endpoint converts pandas errors to 400 Bad Request
            assert response.status_code == 400
            assert "Error reading file" in response.json()["detail"]
            mock_logger.error.assert_called()

    @pytest.mark.error
    @patch('app.routes.file_routes.logger')
    def test_error_logging(self, mock_logger, client, sample_csv_file):
        """Test that errors are properly logged"""
        with patch('pandas.read_csv', side_effect=Exception("Test error")):
            response = client.post(
                "/files/upload",
                files={"file": ("test.csv", sample_csv_file, "text/csv")}
            )

            # Just verify that error was logged, regardless of status code
            mock_logger.error.assert_called()
            # Verify the error message contains expected content
            call_args = mock_logger.error.call_args[0][0]
            assert "Error reading file" in call_args


# Parametrized tests for multiple file types
class TestParametrizedFileOperations:
    """Test operations across different file types"""

    @pytest.mark.parametrize("filename,content_type,expected_type", [
        ("test.csv", "text/csv", "csv"),
        ("test.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "excel"),
        ("test.xls", "application/vnd.ms-excel", "excel")
    ])
    def test_upload_different_file_types(self, client, sample_csv_file, filename, content_type, expected_type):
        """Test uploading different file types"""
        # Use sample_csv_file for all types for simplicity in this test
        # In real tests, you'd have specific file fixtures for each type
        response = client.post(
            "/files/upload",
            files={"file": (filename, sample_csv_file, content_type)}
        )

        if expected_type == "csv":
            assert response.status_code == 200
            assert response.json()["data"]["file_type"] == expected_type
        else:
            # Excel files with CSV content will fail
            assert response.status_code == 400

    @pytest.mark.parametrize("rows,expected_status", [
        (10, 200),
        (50, 200),
        (100, 200),
        (1000, 200)
    ])
    def test_preview_different_row_counts(self, client, uploaded_file_with_data, rows, expected_status):
        """Test preview with different row counts"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}/preview?num_rows={rows}")

        assert response.status_code == expected_status
        if expected_status == 200:
            data = response.json()
            returned_rows = min(rows, 5)  # Sample data has 5 rows
            assert len(data["data"]["rows"]) == returned_rows
