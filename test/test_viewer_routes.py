# tests/test_viewer_routes.py
from unittest.mock import patch

import pandas as pd
import pytest


class TestGetFileData:
    """Test file data retrieval functionality"""

    @pytest.mark.viewer
    def test_get_file_data_default_pagination(self, client, uploaded_file_with_data):
        """Test getting file data with default pagination"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}/data")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Retrieved" in data["message"]

        file_data = data["data"]
        assert file_data["filename"] == "test.csv"
        assert len(file_data["columns"]) == 4
        assert len(file_data["rows"]) == 5  # All rows from sample data
        assert file_data["total_rows"] == 5
        assert file_data["current_page"] == 1
        assert file_data["page_size"] == 1000
        assert file_data["total_pages"] == 1

    @pytest.mark.viewer
    def test_get_file_data_custom_pagination(self, client, uploaded_file_with_data):
        """Test getting file data with custom pagination"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}/data?page=1&page_size=2")

        assert response.status_code == 200
        data = response.json()
        file_data = data["data"]

        assert len(file_data["rows"]) == 2
        assert file_data["current_page"] == 1
        assert file_data["page_size"] == 2
        assert file_data["total_pages"] == 3  # 5 rows / 2 per page = 3 pages

    @pytest.mark.viewer
    def test_get_file_data_second_page(self, client, uploaded_file_with_data):
        """Test getting second page of file data"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}/data?page=2&page_size=2")

        assert response.status_code == 200
        data = response.json()
        file_data = data["data"]

        assert len(file_data["rows"]) == 2
        assert file_data["current_page"] == 2
        assert file_data["total_pages"] == 3

    @pytest.mark.viewer
    def test_get_file_data_last_page_partial(self, client, uploaded_file_with_data):
        """Test getting last page with partial data"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}/data?page=3&page_size=2")

        assert response.status_code == 200
        data = response.json()
        file_data = data["data"]

        assert len(file_data["rows"]) == 1  # Last page has only 1 row
        assert file_data["current_page"] == 3

    @pytest.mark.viewer
    def test_get_file_data_with_nan_values(self, client, mock_storage):
        """Test getting file data that contains NaN values"""
        # Create DataFrame with NaN values
        df_with_nan = pd.DataFrame({
            'name': ['John', None, 'Jane'],
            'age': [25, 30, None],
            'score': [85.5, None, 92.0]
        })

        file_id = "test-nan-file"
        mock_storage[file_id] = {
            "info": {"filename": "test_nan.csv", "total_rows": 3},
            "data": df_with_nan
        }

        response = client.get(f"/files/{file_id}/data")

        assert response.status_code == 200
        data = response.json()
        rows = data["data"]["rows"]

        # Check that NaN values are converted to empty strings
        assert rows[1]["name"] == ""  # None/NaN converted to empty string
        assert rows[2]["age"] == ""  # None/NaN converted to empty string

    @pytest.mark.viewer
    @pytest.mark.error
    def test_get_file_data_nonexistent_file(self, client, mock_storage):
        """Test getting data for nonexistent file"""
        response = client.get("/files/nonexistent-id/data")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @pytest.mark.viewer
    @pytest.mark.error
    def test_get_file_data_invalid_page(self, client, uploaded_file_with_data):
        """Test getting file data with invalid page parameters"""
        file_id = "test-file-id-123"

        # Test page number less than 1
        response = client.get(f"/files/{file_id}/data?page=0")
        assert response.status_code == 422

        # Test page size exceeding limit
        response = client.get(f"/files/{file_id}/data?page_size=6000")
        assert response.status_code == 422


class TestGetFileInfo:
    """Test file info retrieval functionality"""

    @pytest.mark.viewer
    def test_get_file_info_success(self, client, uploaded_file_with_data):
        """Test getting file info successfully"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}/info")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        file_info = data["data"]
        assert file_info["file_id"] == file_id
        assert file_info["filename"] == "test.csv"
        assert file_info["total_rows"] == 5
        assert file_info["total_columns"] == 4
        assert "upload_time" in file_info

    @pytest.mark.viewer
    def test_get_file_info_with_missing_filename(self, client, mock_storage):
        """Test getting file info when filename is missing"""
        file_id = "test-no-filename"
        df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_storage[file_id] = {
            "info": {},  # Missing filename
            "data": df
        }

        response = client.get(f"/files/{file_id}/info")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["filename"] == "Unknown File"

    @pytest.mark.viewer
    @pytest.mark.error
    def test_get_file_info_nonexistent_file(self, client, mock_storage):
        """Test getting info for nonexistent file"""
        response = client.get("/files/nonexistent-id/info")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestUpdateFileData:
    """Test file data update functionality"""

    @pytest.mark.viewer
    def test_update_file_data_success(self, client, uploaded_file_with_data):
        """Test successful file data update"""
        file_id = "test-file-id-123"

        # New data to update
        updated_data = {
            "columns": ["name", "age", "city"],
            "rows": [
                {"name": "Alice", "age": 30, "city": "Boston"},
                {"name": "Bob", "age": 25, "city": "Seattle"}
            ]
        }

        response = client.put(
            f"/files/{file_id}/data",
            json={"data": updated_data}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "updated successfully" in data["message"]

        result_data = data["data"]
        assert result_data["total_rows"] == 2
        assert result_data["columns"] == 3
        assert "last_modified" in result_data

    @pytest.mark.viewer
    def test_update_file_data_changes_storage(self, client, mock_storage):
        """Test that file data update actually changes stored data"""
        file_id = "test-update"
        original_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        mock_storage[file_id] = {
            "info": {"filename": "test.csv"},
            "data": original_df
        }

        # Update with new data
        updated_data = {
            "columns": ["new_col1", "new_col2"],
            "rows": [
                {"new_col1": 10, "new_col2": "x"},
                {"new_col1": 20, "new_col2": "y"}
            ]
        }

        response = client.put(
            f"/files/{file_id}/data",
            json={"data": updated_data}
        )

        assert response.status_code == 200

        # Verify storage was actually updated
        stored_df = mock_storage[file_id]["data"]
        assert list(stored_df.columns) == ["new_col1", "new_col2"]
        assert len(stored_df) == 2
        assert stored_df.iloc[0]["new_col1"] == 10

    @pytest.mark.viewer
    @pytest.mark.error
    def test_update_file_data_missing_rows(self, client, uploaded_file_with_data):
        """Test update with missing rows field"""
        file_id = "test-file-id-123"

        response = client.put(
            f"/files/{file_id}/data",
            json={"data": {"columns": ["col1"]}}  # Missing rows
        )

        assert response.status_code == 400
        assert "Missing 'rows' or 'columns'" in response.json()["detail"]

    @pytest.mark.viewer
    @pytest.mark.error
    def test_update_file_data_missing_columns(self, client, uploaded_file_with_data):
        """Test update with missing columns field"""
        file_id = "test-file-id-123"

        response = client.put(
            f"/files/{file_id}/data",
            json={"data": {"rows": [{"col1": 1}]}}  # Missing columns
        )

        assert response.status_code == 400
        assert "Missing 'rows' or 'columns'" in response.json()["detail"]

    @pytest.mark.viewer
    @pytest.mark.error
    def test_update_nonexistent_file(self, client, mock_storage):
        """Test updating nonexistent file"""
        response = client.put(
            "/files/nonexistent-id/data",
            json={"data": {"columns": ["col1"], "rows": [{"col1": 1}]}}
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestDownloadFile:
    """Test file download functionality"""

    @pytest.mark.viewer
    def test_download_csv_format(self, client, uploaded_file_with_data):
        """Test downloading file in CSV format"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}/download?format=csv")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "attachment" in response.headers["content-disposition"]
        assert "test_modified.csv" in response.headers["content-disposition"]

        # Check CSV content
        csv_content = response.content.decode('utf-8')
        assert "name,age,city,salary" in csv_content  # Header row
        assert "John Doe" in csv_content  # Data row

    @pytest.mark.viewer
    def test_download_excel_format(self, client, uploaded_file_with_data):
        """Test downloading file in Excel format"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}/download?format=xlsx")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        assert "attachment" in response.headers["content-disposition"]
        assert "test_modified.xlsx" in response.headers["content-disposition"]

        # Verify it's actually Excel content (starts with Excel signature)
        assert response.content.startswith(b'PK')  # Excel files are ZIP format

    @pytest.mark.viewer
    def test_download_default_format(self, client, uploaded_file_with_data):
        """Test downloading with default format (CSV)"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}/download")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"

    @pytest.mark.viewer
    def test_download_with_custom_filename(self, client, mock_storage):
        """Test download uses original filename for naming"""
        file_id = "test-custom-name"
        df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        mock_storage[file_id] = {
            "info": {"filename": "my_special_data.csv"},
            "data": df
        }

        response = client.get(f"/files/{file_id}/download?format=csv")

        assert response.status_code == 200
        assert "my_special_data_modified.csv" in response.headers["content-disposition"]

    @pytest.mark.viewer
    @pytest.mark.error
    def test_download_invalid_format(self, client, uploaded_file_with_data):
        """Test download with invalid format"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}/download?format=pdf")

        assert response.status_code == 422  # Regex validation should catch this

    @pytest.mark.viewer
    @pytest.mark.error
    def test_download_nonexistent_file(self, client, mock_storage):
        """Test downloading nonexistent file"""
        response = client.get("/files/nonexistent-id/download")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestGetFileStats:
    """Test file statistics functionality"""

    @pytest.mark.viewer
    def test_get_file_stats_basic(self, client, uploaded_file_with_data):
        """Test getting basic file statistics"""
        file_id = "test-file-id-123"

        # Patch the problematic memory_usage calculation to return a simple int
        with patch('pandas.DataFrame.memory_usage') as mock_memory:
            mock_memory.return_value.sum.return_value = 1000  # Simple int

            response = client.get(f"/files/{file_id}/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        stats = data["data"]
        assert stats["filename"] == "test.csv"
        assert stats["total_rows"] == 5
        assert stats["total_columns"] == 4
        assert "memory_usage" in stats
        assert "column_types" in stats
        assert "null_counts" in stats

    @pytest.mark.viewer
    def test_get_file_stats_column_classification(self, client, mock_storage):
        """Test that file stats correctly classifies column types"""
        # Create DataFrame with different column types (avoiding datetime for now)
        df = pd.DataFrame({
            'numeric_int': [1, 2, 3, 4, 5],
            'numeric_float': [1.1, 2.2, 3.3, 4.4, 5.5],
            'text_col': ['a', 'b', 'c', 'd', 'e']
        })

        file_id = "test-stats"
        mock_storage[file_id] = {
            "info": {"filename": "stats_test.csv"},
            "data": df
        }

        response = client.get(f"/files/{file_id}/stats")

        assert response.status_code == 200
        stats = response.json()["data"]

        assert "numeric_int" in stats["numeric_columns"]
        assert "numeric_float" in stats["numeric_columns"]
        assert "text_col" in stats["text_columns"]
        # Note: datetime_columns will be empty for this test
        assert isinstance(stats["datetime_columns"], list)

    @pytest.mark.viewer
    def test_get_file_stats_numeric_statistics(self, client, mock_storage):
        """Test numeric statistics calculation"""
        df = pd.DataFrame({
            'numbers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'text': ['a'] * 10
        })

        file_id = "test-numeric-stats"
        mock_storage[file_id] = {
            "info": {"filename": "numeric_test.csv"},
            "data": df
        }

        # Mock the memory_usage to avoid serialization issues
        with patch('pandas.DataFrame.memory_usage') as mock_memory:
            mock_memory.return_value.sum.return_value = 2000

            response = client.get(f"/files/{file_id}/stats")

        assert response.status_code == 200
        stats = response.json()["data"]

        numeric_stats = stats["numeric_statistics"]["numbers"]
        assert numeric_stats["count"] == 10
        assert numeric_stats["mean"] == 5.5
        assert numeric_stats["min"] == 1.0
        assert numeric_stats["max"] == 10.0
        assert numeric_stats["median"] == 5.5

    @pytest.mark.viewer
    def test_get_file_stats_with_nulls(self, client, mock_storage):
        """Test statistics with null values"""
        df = pd.DataFrame({
            'col_with_nulls': [1, 2, None, 4, None],
            'col_without_nulls': [1, 2, 3, 4, 5]
        })

        file_id = "test-nulls"
        mock_storage[file_id] = {
            "info": {"filename": "nulls_test.csv"},
            "data": df
        }

        response = client.get(f"/files/{file_id}/stats")

        assert response.status_code == 200
        stats = response.json()["data"]

        assert stats["null_counts"]["col_with_nulls"] == 2
        assert stats["null_counts"]["col_without_nulls"] == 0

    @pytest.mark.viewer
    def test_get_file_stats_memory_usage_serialization(self, client, mock_storage):
        """Test that memory usage can be serialized properly"""
        df = pd.DataFrame({
            'simple_col': [1, 2, 3, 4, 5]
        })

        file_id = "test-memory"
        mock_storage[file_id] = {
            "info": {"filename": "memory_test.csv"},
            "data": df
        }

        response = client.get(f"/files/{file_id}/stats")

        assert response.status_code == 200
        stats = response.json()["data"]

        # Memory usage should be a number (int or float)
        assert isinstance(stats["memory_usage"], (int, float))
        assert stats["memory_usage"] > 0

    @pytest.mark.viewer
    @pytest.mark.error
    def test_get_file_stats_nonexistent_file(self, client, mock_storage):
        """Test getting stats for nonexistent file"""
        response = client.get("/files/nonexistent-id/stats")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestValidateFileData:
    """Test file data validation functionality"""

    @pytest.mark.viewer
    def test_validate_clean_data(self, client, uploaded_file_with_data):
        """Test validation of clean data"""
        response = client.post(f"/files/test-file-id-123/validate")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        validation = data["data"]
        assert validation["is_valid"] is True
        assert validation["filename"] == "test.csv"
        assert len(validation["issues"]) == 0

    @pytest.mark.viewer
    def test_validate_data_with_duplicates(self, client, mock_storage):
        """Test validation detects duplicate rows"""
        df = pd.DataFrame({
            'col1': [1, 2, 1, 3],  # Duplicate row: 1
            'col2': ['a', 'b', 'a', 'c']  # Duplicate row: 'a'
        })

        file_id = "test-duplicates"
        mock_storage[file_id] = {
            "info": {"filename": "duplicates.csv"},
            "data": df
        }

        response = client.post(f"/files/{file_id}/validate")

        assert response.status_code == 200
        validation = response.json()["data"]

        assert validation["summary"]["duplicate_rows"] == 1
        assert any("duplicate" in warning.lower() for warning in validation["warnings"])

    @pytest.mark.viewer
    def test_validate_data_with_empty_rows(self, client, mock_storage):
        """Test validation detects empty rows"""
        df = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': ['a', None, 'c']
        })

        file_id = "test-empty-rows"
        mock_storage[file_id] = {
            "info": {"filename": "empty_rows.csv"},
            "data": df
        }

        response = client.post(f"/files/{file_id}/validate")

        assert response.status_code == 200
        validation = response.json()["data"]

        assert validation["summary"]["empty_rows"] == 1
        assert any("empty" in warning.lower() for warning in validation["warnings"])

    @pytest.mark.viewer
    def test_validate_data_with_all_null_columns(self, client, mock_storage):
        """Test validation detects columns with all null values"""
        df = pd.DataFrame({
            'good_col': [1, 2, 3],
            'all_null_col': [None, None, None],
            'another_good_col': ['a', 'b', 'c']
        })

        file_id = "test-null-columns"
        mock_storage[file_id] = {
            "info": {"filename": "null_columns.csv"},
            "data": df
        }

        response = client.post(f"/files/{file_id}/validate")

        assert response.status_code == 200
        validation = response.json()["data"]

        assert "all_null_col" in validation["summary"]["columns_with_all_nulls"]
        assert any("all null values" in warning.lower() for warning in validation["warnings"])

    @pytest.mark.viewer
    def test_validate_mixed_numeric_text_column(self, client, mock_storage):
        """Test validation detects mixed numeric/text columns"""
        df = pd.DataFrame({
            'mixed_col': ['1', '2', 'text', '4', '5', '6', '7', '8', '9', '10']  # 80% numeric, 20% text
        })

        file_id = "test-mixed-column"
        mock_storage[file_id] = {
            "info": {"filename": "mixed_column.csv"},
            "data": df
        }

        response = client.post(f"/files/{file_id}/validate")

        assert response.status_code == 200
        validation = response.json()["data"]

        # Should detect that mixed_col is mostly numeric but has some text
        mixed_warning = any(
            "mostly numeric but contains some text" in warning.lower()
            for warning in validation["warnings"]
        )
        assert mixed_warning

    @pytest.mark.viewer
    @pytest.mark.error
    def test_validate_nonexistent_file(self, client, mock_storage):
        """Test validation of nonexistent file"""
        response = client.post("/files/nonexistent-id/validate")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestViewerRoutesIntegration:
    """Integration tests for viewer routes"""

    @pytest.mark.viewer
    @pytest.mark.integration
    def test_full_workflow_view_update_download(self, client, uploaded_file_with_data):
        """Test complete workflow: view data -> update -> download"""
        file_id = "test-file-id-123"

        # 1. Get original data
        response = client.get(f"/files/{file_id}/data")
        assert response.status_code == 200
        original_data = response.json()["data"]

        # 2. Update the data
        updated_data = {
            "columns": ["name", "age"],
            "rows": [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30}
            ]
        }

        response = client.put(
            f"/files/{file_id}/data",
            json={"data": updated_data}
        )
        assert response.status_code == 200

        # 3. Verify data was updated
        response = client.get(f"/files/{file_id}/data")
        assert response.status_code == 200
        new_data = response.json()["data"]
        assert len(new_data["columns"]) == 2
        assert len(new_data["rows"]) == 2
        assert new_data["rows"][0]["name"] == "Alice"

        # 4. Download updated file
        response = client.get(f"/files/{file_id}/download?format=csv")
        assert response.status_code == 200
        csv_content = response.content.decode('utf-8')
        assert "Alice" in csv_content
        assert "Bob" in csv_content

    @pytest.mark.viewer
    @pytest.mark.integration
    def test_pagination_consistency(self, client, mock_storage):
        """Test that pagination is consistent across requests"""
        # Create dataset with 95 rows (not evenly divisible by 10)
        df = pd.DataFrame({
            'id': range(1, 96),  # 95 rows
            'value': [f"value_{i}" for i in range(1, 96)]
        })

        file_id = "test-large-file"
        mock_storage[file_id] = {
            "info": {"filename": "large_file.csv"},
            "data": df
        }

        page_size = 10
        total_rows_collected = 0

        # Test first few pages explicitly
        for page in range(1, 10):  # Pages 1-9
            response = client.get(f"/files/{file_id}/data?page={page}&page_size={page_size}")
            assert response.status_code == 200

            data = response.json()["data"]
            rows_in_page = len(data["rows"])

            if page <= 9:  # First 9 pages should have 10 rows each
                assert rows_in_page == 10

            total_rows_collected += rows_in_page

        # Test the last page (page 10) - should have 5 rows (95 - 90 = 5)
        response = client.get(f"/files/{file_id}/data?page=10&page_size={page_size}")
        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["rows"]) == 5  # Last 5 rows
        total_rows_collected += len(data["rows"])

        assert total_rows_collected == 95


class TestViewerRoutesErrorHandling:
    """Test error handling in viewer routes"""

    @pytest.mark.viewer
    @pytest.mark.error
    @patch('app.routes.viewer_routes.logger')
    def test_get_data_internal_error(self, mock_logger, client, mock_storage):
        """Test handling of internal errors in get_file_data"""
        file_id = "test-error"
        mock_storage[file_id] = {
            "info": {"filename": "error_test.csv"},
            "data": "not_a_dataframe"  # This will cause an error
        }

        response = client.get(f"/files/{file_id}/data")

        assert response.status_code == 500
        assert "Failed to retrieve file data" in response.json()["detail"]
        mock_logger.error.assert_called()

    @pytest.mark.viewer
    @pytest.mark.error
    @patch('app.routes.viewer_routes.logger')
    def test_update_data_internal_error(self, mock_logger, client, mock_storage):
        """Test handling of internal errors in update_file_data"""
        file_id = "test-update-error"
        mock_storage[file_id] = {
            "info": {"filename": "error_test.csv"},
            "data": pd.DataFrame({'col1': [1, 2, 3]})
        }

        # Invalid data that will cause pandas error
        invalid_data = {
            "columns": ["col1"],
            "rows": [{"col1": "invalid_structure"}] * 10000  # Too much data might cause memory error
        }

        with patch('pandas.DataFrame', side_effect=Exception("Memory error")):
            response = client.put(
                f"/files/{file_id}/data",
                json={"data": invalid_data}
            )

            assert response.status_code == 500
            assert "Failed to update file data" in response.json()["detail"]
            mock_logger.error.assert_called()


# Parametrized tests for different file formats and edge cases
class TestViewerRoutesParametrized:
    """Parametrized tests for viewer routes"""

    @pytest.mark.parametrize("page,page_size,expected_rows", [
        (1, 2, 2),
        (2, 2, 2),
        (3, 2, 1),  # Last page with partial data
        (1, 10, 5),  # Page size larger than total rows
    ])
    @pytest.mark.viewer
    def test_pagination_scenarios(self, client, uploaded_file_with_data, page, page_size, expected_rows):
        """Test various pagination scenarios"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}/data?page={page}&page_size={page_size}")

        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["rows"]) == expected_rows

    @pytest.mark.parametrize("format,content_type", [
        ("csv", "text/csv"),
        ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    ])
    @pytest.mark.viewer
    def test_download_formats(self, client, uploaded_file_with_data, format, content_type):
        """Test downloading in different formats"""
        file_id = "test-file-id-123"
        response = client.get(f"/files/{file_id}/download?format={format}")

        assert response.status_code == 200
        assert content_type in response.headers["content-type"]
