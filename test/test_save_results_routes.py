# tests/test_save_results_routes.py
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import save results routes
try:
    from app.routes.save_results_routes import router as save_results_router

    HAS_SAVE_RESULTS = True
except ImportError:
    HAS_SAVE_RESULTS = False

# Mock storage
test_save_results_storage = {}


@pytest.fixture(scope="session")
def save_results_app():
    """Create FastAPI app with save results routes only"""
    if not HAS_SAVE_RESULTS:
        pytest.skip("save_results_routes not available")

    test_app = FastAPI(title="Save Results Test App")
    test_app.include_router(save_results_router)
    return test_app


@pytest.fixture
def client(save_results_app):
    """Create test client for save results routes"""
    return TestClient(save_results_app)


@pytest.fixture(autouse=True)
def mock_storage():
    """Mock storage for save results tests"""
    test_save_results_storage.clear()

    with patch('app.services.storage_service.uploaded_files', test_save_results_storage):
        yield test_save_results_storage


@pytest.fixture
def mock_storage_info():
    """Mock storage info function"""
    return {
        "backend_class": "InMemoryStorage",
        "storage_type": "memory"
    }


# Skip all save results tests if the module doesn't exist
pytestmark = pytest.mark.skipif(not HAS_SAVE_RESULTS, reason="save_results_routes not available")


class TestSaveResults:
    """Test save results functionality"""

    @pytest.fixture
    def mock_reconciliation_storage(self):
        """Mock reconciliation storage with sample data"""
        mock_storage = MagicMock()
        sample_results = {
            'matched': [
                {'id': 1, 'name': 'John', 'amount_a': 100, 'amount_b': 100},
                {'id': 2, 'name': 'Jane', 'amount_a': 200, 'amount_b': 200}
            ],
            'unmatched_file_a': [
                {'id': 3, 'name': 'Bob', 'amount_a': 300}
            ],
            'unmatched_file_b': [
                {'id': 4, 'name': 'Alice', 'amount_b': 400}
            ]
        }
        mock_storage.get_results.return_value = sample_results
        return mock_storage

    @pytest.fixture
    def mock_delta_storage(self):
        """Mock delta storage with sample data"""
        delta_data = {
            'test-delta-id': {
                'unchanged': pd.DataFrame([
                    {'id': 1, 'name': 'John', 'status': 'unchanged'}
                ]),
                'amended': pd.DataFrame([
                    {'id': 2, 'name': 'Jane', 'old_value': 100, 'new_value': 150}
                ]),
                'deleted': pd.DataFrame([
                    {'id': 3, 'name': 'Bob', 'status': 'deleted'}
                ]),
                'newly_added': pd.DataFrame([
                    {'id': 4, 'name': 'Alice', 'status': 'new'}
                ])
            }
        }
        return delta_data

    @pytest.mark.save_results
    def test_save_endpoint_exists(self, client):
        """Test that save endpoint exists and validates input"""
        # Send empty request - should get validation error (422), not 404
        response = client.post("/save-results/save", json={})
        assert response.status_code == 422  # Validation error, not 404

    @pytest.mark.save_results
    def test_health_endpoint(self, client, mock_storage, mock_storage_info):
        """Test health check endpoint"""
        with patch('app.services.storage_service.get_storage_info', return_value=mock_storage_info):
            response = client.get("/save-results/health")
            assert response.status_code == 200
            data = response.json()
            assert data["service"] == "save_results"

    @pytest.mark.save_results
    def test_save_reconciliation_all_results(self, client, mock_storage, mock_reconciliation_storage):
        """Test saving all reconciliation results"""
        with patch('app.services.reconciliation_service.optimized_reconciliation_storage', mock_reconciliation_storage):
            request_data = {
                "result_id": "test-reconciliation-id",
                "result_type": "all",
                "file_format": "excel",
                "process_type": "reconciliation"
            }

            response = client.post("/save-results/save", json=request_data)

            assert response.status_code == 200
            data = response.json()
            saved_info = data["saved_file_info"]

            # All results should include matched + unmatched_a + unmatched_b = 4 rows
            assert saved_info["total_rows"] == 4
            assert saved_info["file_format"] == "excel"
            assert "Result_Type" in saved_info["columns"]

    @pytest.mark.save_results
    def test_save_delta_amended_results(self, client, mock_storage, mock_delta_storage):
        """Test saving delta amended results"""
        with patch('app.routes.delta_routes.delta_storage', mock_delta_storage):
            request_data = {
                "result_id": "test-delta-id",
                "result_type": "amended",
                "file_format": "csv",
                "process_type": "delta"
            }

            response = client.post("/save-results/save", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

            saved_info = data["saved_file_info"]
            assert saved_info["original_result_id"] == "test-delta-id"
            assert saved_info["process_type"] == "delta"
            assert saved_info["result_type"] == "amended"
            assert saved_info["total_rows"] == 1

    @pytest.mark.save_results
    def test_save_delta_all_results(self, client, mock_storage, mock_delta_storage):
        """Test saving all delta results"""
        with patch('app.routes.delta_routes.delta_storage', mock_delta_storage):
            request_data = {
                "result_id": "test-delta-id",
                "result_type": "all",
                "file_format": "csv",
                "process_type": "delta"
            }

            response = client.post("/save-results/save", json=request_data)

            assert response.status_code == 200
            data = response.json()
            saved_info = data["saved_file_info"]

            # All delta results = unchanged + amended + deleted + newly_added = 4 rows
            assert saved_info["total_rows"] == 4
            assert "Delta_Category" in saved_info["columns"]

    @pytest.mark.save_results
    def test_save_with_custom_filename(self, client, mock_storage, mock_reconciliation_storage):
        """Test saving with custom filename"""
        with patch('app.services.reconciliation_service.optimized_reconciliation_storage', mock_reconciliation_storage):
            request_data = {
                "result_id": "test-reconciliation-id",
                "result_type": "matched",
                "file_format": "csv",
                "process_type": "reconciliation",
                "custom_filename": "my_custom_results"
            }

            response = client.post("/save-results/save", json=request_data)

            assert response.status_code == 200
            data = response.json()
            saved_info = data["saved_file_info"]
            assert "my_custom_results.csv" in saved_info["filename"]

    @pytest.mark.save_results
    @pytest.mark.error
    def test_save_invalid_process_type(self, client, mock_storage):
        """Test saving with invalid process type"""
        request_data = {
            "result_id": "test-id",
            "result_type": "matched",
            "file_format": "csv",
            "process_type": "invalid_process"
        }

        response = client.post("/save-results/save", json=request_data)

        assert response.status_code == 400
        assert "process_type must be" in response.json()["detail"]

    @pytest.mark.save_results
    @pytest.mark.error
    def test_save_nonexistent_reconciliation_id(self, client, mock_storage):
        """Test saving with nonexistent reconciliation ID"""
        with patch('app.services.reconciliation_service.optimized_reconciliation_storage') as mock_storage_service:
            mock_storage_service.get_results.return_value = None

            request_data = {
                "result_id": "nonexistent-id",
                "result_type": "matched",
                "file_format": "csv",
                "process_type": "reconciliation"
            }

            response = client.post("/save-results/save", json=request_data)

            assert response.status_code == 404
            assert "not found" in response.json()["detail"]

    @pytest.mark.save_results
    @pytest.mark.error
    def test_save_nonexistent_delta_id(self, client, mock_storage):
        """Test saving with nonexistent delta ID"""
        with patch('app.routes.delta_routes.delta_storage', {}):
            request_data = {
                "result_id": "nonexistent-delta-id",
                "result_type": "amended",
                "file_format": "csv",
                "process_type": "delta"
            }

            response = client.post("/save-results/save", json=request_data)

            assert response.status_code == 404
            assert "Delta ID not found" in response.json()["detail"]

    @pytest.mark.save_results
    @pytest.mark.error
    def test_save_invalid_result_type_reconciliation(self, client, mock_storage, mock_reconciliation_storage):
        """Test saving with invalid result type for reconciliation"""
        with patch('app.services.reconciliation_service.optimized_reconciliation_storage', mock_reconciliation_storage):
            request_data = {
                "result_id": "test-reconciliation-id",
                "result_type": "invalid_type",
                "file_format": "csv",
                "process_type": "reconciliation"
            }

            response = client.post("/save-results/save", json=request_data)

            assert response.status_code == 400
            assert "Invalid result_type for reconciliation" in response.json()["detail"]

    @pytest.mark.save_results
    @pytest.mark.error
    def test_save_invalid_result_type_delta(self, client, mock_storage, mock_delta_storage):
        """Test saving with invalid result type for delta"""
        with patch('app.routes.delta_routes.delta_storage', mock_delta_storage):
            request_data = {
                "result_id": "test-delta-id",
                "result_type": "invalid_type",
                "file_format": "csv",
                "process_type": "delta"
            }

            response = client.post("/save-results/save", json=request_data)

            assert response.status_code == 400
            assert "Invalid result_type for delta" in response.json()["detail"]

    @pytest.mark.save_results
    @pytest.mark.error
    def test_save_empty_data(self, client, mock_storage):
        """Test saving when result data is empty"""
        mock_storage_service = MagicMock()
        mock_storage_service.get_results.return_value = {
            'matched': [],
            'unmatched_file_a': [],
            'unmatched_file_b': []
        }

        with patch('app.services.reconciliation_service.optimized_reconciliation_storage', mock_storage_service):
            request_data = {
                "result_id": "test-reconciliation-id",
                "result_type": "matched",
                "file_format": "csv",
                "process_type": "reconciliation"
            }

            response = client.post("/save-results/save", json=request_data)

            assert response.status_code == 404
            assert "No data found" in response.json()["detail"]


class TestListSavedResults:
    """Test listing saved results functionality"""

    @pytest.mark.save_results
    def test_list_saved_results_empty(self, client, mock_storage):
        """Test listing when no saved results exist"""
        response = client.get("/save-results/list")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_count"] == 0
        assert len(data["saved_files"]) == 0

    @pytest.mark.save_results
    def test_list_saved_results_with_data(self, client, mock_storage):
        """Test listing when saved results exist"""
        # Add some saved results to storage
        saved_file_1 = {
            "info": {
                "filename": "saved_reconciliation_matched_20240715.csv",
                "is_saved_result": True,
                "file_type": "csv",
                "total_rows": 100,
                "total_columns": 5,
                "file_size_mb": 0.5,
                "description": "Test saved results",
                "created_at": "2024-07-15T10:00:00"
            },
            "data": pd.DataFrame({'col1': [1, 2, 3]})
        }

        saved_file_2 = {
            "info": {
                "filename": "saved_delta_amended_20240715.csv",
                "is_saved_result": True,
                "file_type": "csv",
                "total_rows": 50,
                "total_columns": 3,
                "file_size_mb": 0.2,
                "created_at": "2024-07-15T11:00:00"
            },
            "data": pd.DataFrame({'col1': [4, 5, 6]})
        }

        # Add regular uploaded file (should not appear in saved results)
        regular_file = {
            "info": {
                "filename": "regular_upload.csv",
                "is_saved_result": False,
                "file_type": "csv"
            },
            "data": pd.DataFrame({'col1': [7, 8, 9]})
        }

        mock_storage["saved-1"] = saved_file_1
        mock_storage["saved-2"] = saved_file_2
        mock_storage["regular-1"] = regular_file

        response = client.get("/save-results/list")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_count"] == 2  # Only saved results

        saved_files = data["saved_files"]
        assert len(saved_files) == 2

        # Check that results are sorted by creation date (newest first)
        assert saved_files[0]["created_at"] == "2024-07-15T11:00:00"
        assert saved_files[1]["created_at"] == "2024-07-15T10:00:00"

        # Verify saved file structure
        assert "saved_file_id" in saved_files[0]
        assert saved_files[0]["filename"] == "saved_delta_amended_20240715.csv"
        assert saved_files[0]["total_rows"] == 50


class TestSavedFileInfo:
    """Test getting saved file info functionality"""

    @pytest.mark.save_results
    def test_get_saved_file_info_success(self, client, mock_storage):
        """Test getting saved file info successfully"""
        saved_file_id = "test-saved-file"
        mock_storage[saved_file_id] = {
            "info": {
                "filename": "test_saved_results.csv",
                "is_saved_result": True,
                "file_type": "csv",
                "total_rows": 100,
                "description": "Test description"
            },
            "data": pd.DataFrame()
        }

        response = client.get(f"/save-results/info/{saved_file_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["file_info"]["filename"] == "test_saved_results.csv"
        assert data["file_info"]["is_saved_result"] is True

    @pytest.mark.save_results
    @pytest.mark.error
    def test_get_saved_file_info_not_found(self, client, mock_storage):
        """Test getting info for nonexistent saved file"""
        response = client.get("/save-results/info/nonexistent-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @pytest.mark.save_results
    @pytest.mark.error
    def test_get_saved_file_info_not_saved_result(self, client, mock_storage):
        """Test getting info for file that's not a saved result"""
        file_id = "regular-file"
        mock_storage[file_id] = {
            "info": {
                "filename": "regular_file.csv",
                "is_saved_result": False,  # Not a saved result
                "file_type": "csv"
            },
            "data": pd.DataFrame()
        }

        response = client.get(f"/save-results/info/{file_id}")

        assert response.status_code == 404
        assert "not a saved result" in response.json()["detail"]


class TestDeleteSavedFile:
    """Test deleting saved files functionality"""

    @pytest.mark.save_results
    def test_delete_saved_file_success(self, client, mock_storage):
        """Test successful deletion of saved file"""
        saved_file_id = "test-saved-file"
        mock_storage[saved_file_id] = {
            "info": {
                "filename": "test_saved_results.csv",
                "is_saved_result": True,
                "file_type": "csv"
            },
            "data": pd.DataFrame()
        }

        response = client.delete(f"/save-results/delete/{saved_file_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deleted successfully" in data["message"]

        # Verify file was actually removed from storage
        assert saved_file_id not in mock_storage

    @pytest.mark.save_results
    @pytest.mark.error
    def test_delete_saved_file_not_found(self, client, mock_storage):
        """Test deleting nonexistent saved file"""
        response = client.delete("/save-results/delete/nonexistent-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @pytest.mark.save_results
    @pytest.mark.error
    def test_delete_file_not_saved_result(self, client, mock_storage):
        """Test deleting file that's not a saved result"""
        file_id = "regular-file"
        mock_storage[file_id] = {
            "info": {
                "filename": "regular_file.csv",
                "is_saved_result": False,
                "file_type": "csv"
            },
            "data": pd.DataFrame()
        }

        response = client.delete(f"/save-results/delete/{file_id}")

        assert response.status_code == 404
        assert "not a saved result" in response.json()["detail"]


class TestDownloadSavedFile:
    """Test downloading saved files functionality"""

    @pytest.mark.save_results
    def test_download_saved_file_csv(self, client, mock_storage):
        """Test downloading saved file as CSV"""
        saved_file_id = "test-saved-file"
        test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['John', 'Jane', 'Bob'],
            'amount': [100, 200, 300]
        })

        mock_storage[saved_file_id] = {
            "info": {
                "filename": "test_saved_results.csv",
                "is_saved_result": True,
                "file_type": "csv"
            },
            "data": test_df
        }

        response = client.get(f"/save-results/download/{saved_file_id}?format=csv")

        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "attachment" in response.headers["content-disposition"]
        assert "test_saved_results.csv" in response.headers["content-disposition"]

        # Check CSV content
        content = response.content.decode('utf-8')
        assert "id,name,amount" in content
        assert "John" in content

    @pytest.mark.save_results
    def test_download_saved_file_excel(self, client, mock_storage):
        """Test downloading saved file as Excel"""
        saved_file_id = "test-saved-file"
        test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['John', 'Jane', 'Bob']
        })

        mock_storage[saved_file_id] = {
            "info": {
                "filename": "test_saved_results.csv",
                "is_saved_result": True,
                "file_type": "csv"
            },
            "data": test_df
        }

        response = client.get(f"/save-results/download/{saved_file_id}?format=excel")

        assert response.status_code == 200
        assert "spreadsheet" in response.headers["content-type"]
        assert "test_saved_results.xlsx" in response.headers["content-disposition"]

    @pytest.mark.save_results
    @pytest.mark.error
    def test_download_saved_file_not_found(self, client, mock_storage):
        """Test downloading nonexistent saved file"""
        response = client.get("/save-results/download/nonexistent-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @pytest.mark.save_results
    @pytest.mark.error
    def test_download_file_not_saved_result(self, client, mock_storage):
        """Test downloading file that's not a saved result"""
        file_id = "regular-file"
        mock_storage[file_id] = {
            "info": {
                "filename": "regular_file.csv",
                "is_saved_result": False,
                "file_type": "csv"
            },
            "data": pd.DataFrame()
        }

        response = client.get(f"/save-results/download/{file_id}")

        assert response.status_code == 404
        assert "not a saved result" in response.json()["detail"]


class TestHealthCheck:
    """Test health check functionality"""

    @pytest.mark.save_results
    def test_health_check_success(self, client, mock_storage):
        """Test successful health check"""
        # Add some saved results to storage
        mock_storage["saved-1"] = {
            "info": {"is_saved_result": True, "filename": "test1.csv"},
            "data": pd.DataFrame()
        }
        mock_storage["saved-2"] = {
            "info": {"is_saved_result": True, "filename": "test2.csv"},
            "data": pd.DataFrame()
        }
        mock_storage["regular-1"] = {
            "info": {"is_saved_result": False, "filename": "regular.csv"},
            "data": pd.DataFrame()
        }

        with patch('app.services.storage_service.get_storage_info') as mock_get_info:
            mock_get_info.return_value = {
                "backend_class": "InMemoryStorage",
                "storage_type": "memory"
            }

            response = client.get("/save-results/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "save_results"
        assert data["saved_results_count"] == 2
        assert data["total_files_in_storage"] == 3
        assert "features" in data
        assert len(data["features"]) > 0

    @pytest.mark.save_results
    @pytest.mark.error
    def test_health_check_error(self, client, mock_storage):
        """Test health check when there's an error"""
        with patch('app.services.storage_service.get_storage_info', side_effect=Exception("Storage error")):
            response = client.get("/save-results/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert data["service"] == "save_results"
        assert "error" in data


class TestResultsSaverUnit:
    """Unit tests for ResultsSaver class methods"""

    @pytest.fixture
    def results_saver(self):
        """Create ResultsSaver instance for testing"""
        from app.routes.save_results_routes import ResultsSaver
        return ResultsSaver()

    @pytest.mark.save_results
    @pytest.mark.unit
    def test_generate_filename_default(self, results_saver):
        """Test default filename generation"""
        filename = results_saver.generate_filename(
            "test-result-id-12345",
            "reconciliation",
            "matched",
            "csv"
        )

        assert "saved_reconciliation_matched" in filename
        assert "test-res" in filename  # First 8 chars of result_id
        assert filename.endswith(".csv")
        assert len(filename.split("_")) >= 5  # Has timestamp components

    @pytest.mark.save_results
    @pytest.mark.unit
    def test_generate_filename_custom(self, results_saver):
        """Test custom filename generation"""
        filename = results_saver.generate_filename(
            "test-result-id",
            "delta",
            "amended",
            "excel",
            "My Custom Results File"
        )

        assert filename == "My Custom Results File.excel"

    @pytest.mark.save_results
    @pytest.mark.unit
    def test_generate_filename_custom_with_extension(self, results_saver):
        """Test custom filename that already has extension"""
        filename = results_saver.generate_filename(
            "test-result-id",
            "delta",
            "amended",
            "csv",
            "My Results.csv"
        )

        assert filename == "My Results.csv"

    @pytest.mark.save_results
    @pytest.mark.unit
    def test_generate_filename_sanitization(self, results_saver):
        """Test filename sanitization for special characters"""
        filename = results_saver.generate_filename(
            "test-result-id",
            "reconciliation",
            "matched",
            "csv",
            "File/With\\Special:Characters*?.csv"
        )

        # Special characters should be removed, only alphanumeric, spaces, hyphens, underscores allowed
        assert "/" not in filename
        assert "\\" not in filename
        assert ":" not in filename
        assert "*" not in filename
        assert "?" not in filename
        assert filename.endswith(".csv")

    @pytest.mark.save_results
    @pytest.mark.unit
    def test_save_dataframe_to_storage(self, results_saver, mock_storage):
        """Test saving DataFrame to storage"""
        test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [100, 200, 300]
        })

        saved_file_id = "test-saved-id"
        filename = "test_file.csv"

        saved_info = results_saver.save_dataframe_to_storage(
            test_df, saved_file_id, filename, "csv", "Test description"
        )

        # Check returned info
        assert saved_info.saved_file_id == saved_file_id
        assert saved_info.filename == filename
        assert saved_info.total_rows == 3
        assert saved_info.total_columns == 3
        assert saved_info.file_format == "csv"
        assert saved_info.description == "Test description"

        # Check that data was stored
        assert saved_file_id in mock_storage
        stored_data = mock_storage[saved_file_id]
        assert stored_data["info"]["is_saved_result"] is True
        assert len(stored_data["data"]) == 3


class TestSaveResultsParametrized:
    """Parametrized tests for save results functionality"""

    @pytest.mark.parametrize("process_type,result_type", [
        ("reconciliation", "matched"),
        ("reconciliation", "unmatched_a"),
        ("reconciliation", "unmatched_b"),
        ("reconciliation", "all"),
        ("delta", "unchanged"),
        ("delta", "amended"),
        ("delta", "deleted"),
        ("delta", "newly_added"),
        ("delta", "all")
    ])
    @pytest.mark.save_results
    def test_save_different_result_types(self, client, mock_storage, process_type, result_type):
        """Test saving different types of results"""
        # Mock the appropriate storage based on process type
        if process_type == "reconciliation":
            mock_reconciliation_storage = MagicMock()
            mock_reconciliation_storage.get_results.return_value = {
                'matched': [{'id': 1, 'name': 'test'}],
                'unmatched_file_a': [{'id': 2, 'name': 'test'}],
                'unmatched_file_b': [{'id': 3, 'name': 'test'}]
            }
            patch_target = 'app.services.reconciliation_service.optimized_reconciliation_storage'
            mock_storage_obj = mock_reconciliation_storage
        else:  # delta
            mock_delta_storage = {
                'test-delta-id': {
                    'unchanged': pd.DataFrame([{'id': 1, 'name': 'test'}]),
                    'amended': pd.DataFrame([{'id': 2, 'name': 'test'}]),
                    'deleted': pd.DataFrame([{'id': 3, 'name': 'test'}]),
                    'newly_added': pd.DataFrame([{'id': 4, 'name': 'test'}])
                }
            }
            patch_target = 'app.routes.delta_routes.delta_storage'
            mock_storage_obj = mock_delta_storage

        with patch(patch_target, mock_storage_obj):
            request_data = {
                "result_id": "test-delta-id" if process_type == "delta" else "test-reconciliation-id",
                "result_type": result_type,
                "file_format": "csv",
                "process_type": process_type
            }

            response = client.post("/save-results/save", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            saved_info = data["saved_file_info"]
            assert saved_info["process_type"] == process_type
            assert saved_info["result_type"] == result_type

    @pytest.mark.parametrize("file_format", ["csv", "excel"])
    @pytest.mark.save_results
    def test_save_different_file_formats(self, client, mock_storage, file_format):
        """Test saving in different file formats"""
        mock_reconciliation_storage = MagicMock()
        mock_reconciliation_storage.get_results.return_value = {
            'matched': [{'id': 1, 'name': 'test', 'amount': 100}]
        }

        with patch('app.services.reconciliation_service.optimized_reconciliation_storage', mock_reconciliation_storage):
            request_data = {
                "result_id": "test-reconciliation-id",
                "result_type": "matched",
                "file_format": file_format,
                "process_type": "reconciliation"
            }

            response = client.post("/save-results/save", json=request_data)

            assert response.status_code == 200
            data = response.json()
            saved_info = data["saved_file_info"]
            assert saved_info["file_format"] == file_format


class TestSaveResultsIntegration:
    """Integration tests for save results functionality"""

    @pytest.mark.save_results
    @pytest.mark.integration
    def test_save_and_list_workflow(self, client, mock_storage):
        """Test complete workflow: save results -> list -> get info -> delete"""
        # Setup mock reconciliation storage
        mock_reconciliation_storage = MagicMock()
        mock_reconciliation_storage.get_results.return_value = {
            'matched': [
                {'id': 1, 'name': 'John', 'amount': 100},
                {'id': 2, 'name': 'Jane', 'amount': 200}
            ]
        }

        with patch('app.services.reconciliation_service.optimized_reconciliation_storage', mock_reconciliation_storage):
            # 1. Save results
            save_request = {
                "result_id": "test-reconciliation-id",
                "result_type": "matched",
                "file_format": "csv",
                "process_type": "reconciliation",
                "description": "Integration test results"
            }

            save_response = client.post("/save-results/save", json=save_request)
            assert save_response.status_code == 200

            saved_file_id = save_response.json()["saved_file_info"]["saved_file_id"]

            # 2. List saved results
            list_response = client.get("/save-results/list")
            assert list_response.status_code == 200
            list_data = list_response.json()
            assert list_data["total_count"] == 1
            assert list_data["saved_files"][0]["saved_file_id"] == saved_file_id

            # 3. Get file info
            info_response = client.get(f"/save-results/info/{saved_file_id}")
            assert info_response.status_code == 200
            info_data = info_response.json()
            assert info_data["file_info"]["is_saved_result"] is True
            assert info_data["file_info"]["description"] == "Integration test results"

            # 4. Download the file
            download_response = client.get(f"/save-results/download/{saved_file_id}?format=csv")
            assert download_response.status_code == 200
            assert "John" in download_response.content.decode('utf-8')

            # 5. Delete the file
            delete_response = client.delete(f"/save-results/delete/{saved_file_id}")
            assert delete_response.status_code == 200

            # 6. Verify it's deleted
            list_response_after = client.get("/save-results/list")
            assert list_response_after.json()["total_count"] == 0

    @pytest.mark.save_results
    @pytest.mark.integration
    def test_save_multiple_results_and_ordering(self, client, mock_storage):
        """Test saving multiple results and verify proper ordering"""
        mock_reconciliation_storage = MagicMock()
        mock_reconciliation_storage.get_results.return_value = {
            'matched': [{'id': 1, 'name': 'test'}]
        }

        with patch('app.services.reconciliation_service.optimized_reconciliation_storage', mock_reconciliation_storage):
            saved_file_ids = []

            # Save 3 different results with slight delays to ensure different timestamps
            for i in range(3):
                request_data = {
                    "result_id": f"test-reconciliation-id-{i}",
                    "result_type": "matched",
                    "file_format": "csv",
                    "process_type": "reconciliation",
                    "custom_filename": f"test_results_{i}"
                }

                response = client.post("/save-results/save", json=request_data)
                assert response.status_code == 200
                saved_file_ids.append(response.json()["saved_file_info"]["saved_file_id"])

            # List results and verify ordering (newest first)
            list_response = client.get("/save-results/list")
            assert list_response.status_code == 200

            list_data = list_response.json()
            assert list_data["total_count"] == 3

            # Files should be ordered by creation time (newest first)
            saved_files = list_data["saved_files"]
            for i in range(len(saved_files) - 1):
                current_time = saved_files[i]["created_at"]
                next_time = saved_files[i + 1]["created_at"]
                assert current_time >= next_time


class TestSaveResultsErrorHandling:
    """Test error handling in save results routes"""

    @pytest.mark.save_results
    @pytest.mark.error
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields"""
        # Missing process_type
        incomplete_request = {
            "result_id": "test-id",
            "result_type": "matched",
            "file_format": "csv"
            # Missing process_type
        }

        response = client.post("/save-results/save", json=incomplete_request)
        assert response.status_code == 422

    @pytest.mark.save_results
    @pytest.mark.error
    def test_invalid_json_request(self, client):
        """Test handling of invalid JSON"""
        response = client.post(
            "/save-results/save",
            data="invalid json content",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    @pytest.mark.save_results
    @pytest.mark.error
    @patch('app.routes.save_results_routes.logger')
    def test_dataframe_creation_error(self, mock_logger, client, mock_storage):
        """Test handling when DataFrame creation fails"""
        # Mock reconciliation storage that returns invalid data
        mock_reconciliation_storage = MagicMock()
        mock_reconciliation_storage.get_results.return_value = {
            'matched': "invalid_data_not_list"  # This will cause DataFrame creation to fail
        }

        with patch('app.services.reconciliation_service.optimized_reconciliation_storage', mock_reconciliation_storage):
            request_data = {
                "result_id": "test-reconciliation-id",
                "result_type": "matched",
                "file_format": "csv",
                "process_type": "reconciliation"
            }

            response = client.post("/save-results/save", json=request_data)

            assert response.status_code == 500
            mock_logger.error.assert_called()


class TestSaveResultsEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.mark.save_results
    def test_save_very_large_dataset(self, client, mock_storage):
        """Test saving a large dataset"""
        # Create large dataset
        large_data = [{'id': i, 'name': f'user_{i}', 'value': i * 100} for i in range(10000)]

        mock_reconciliation_storage = MagicMock()
        mock_reconciliation_storage.get_results.return_value = {
            'matched': large_data
        }

        with patch('app.services.reconciliation_service.optimized_reconciliation_storage', mock_reconciliation_storage):
            request_data = {
                "result_id": "large-dataset-id",
                "result_type": "matched",
                "file_format": "csv",
                "process_type": "reconciliation"
            }

            response = client.post("/save-results/save", json=request_data)

            assert response.status_code == 200
            data = response.json()
            saved_info = data["saved_file_info"]
            assert saved_info["total_rows"] == 10000
            assert saved_info["file_size_mb"] > 0

    @pytest.mark.save_results
    def test_save_single_row_dataset(self, client, mock_storage):
        """Test saving a dataset with only one row"""
        mock_reconciliation_storage = MagicMock()
        mock_reconciliation_storage.get_results.return_value = {
            'matched': [{'id': 1, 'name': 'single_user', 'value': 100}]
        }

        with patch('app.services.reconciliation_service.optimized_reconciliation_storage', mock_reconciliation_storage):
            request_data = {
                "result_id": "single-row-id",
                "result_type": "matched",
                "file_format": "csv",
                "process_type": "reconciliation"
            }

            response = client.post("/save-results/save", json=request_data)

            assert response.status_code == 200
            data = response.json()
            saved_info = data["saved_file_info"]
            assert saved_info["total_rows"] == 1

    @pytest.mark.save_results
    def test_save_with_special_characters_in_data(self, client, mock_storage):
        """Test saving data with special characters"""
        special_data = [
            {'id': 1, 'name': 'John "The Great" O\'Connor', 'description': 'Line 1\nLine 2'},
            {'id': 2, 'name': 'Café & Restaurant', 'description': 'Unicode: résumé naïve'},
            {'id': 3, 'name': 'Data,with,commas', 'description': 'Quotes "inside" text'}
        ]

        mock_reconciliation_storage = MagicMock()
        mock_reconciliation_storage.get_results.return_value = {
            'matched': special_data
        }

        with patch('app.services.reconciliation_service.optimized_reconciliation_storage', mock_reconciliation_storage):
            request_data = {
                "result_id": "special-chars-id",
                "result_type": "matched",
                "file_format": "csv",
                "process_type": "reconciliation"
            }

            response = client.post("/save-results/save", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    @pytest.mark.save_results
    def test_save_with_mixed_data_types(self, client, mock_storage):
        """Test saving data with mixed data types"""
        mixed_data = [
            {'id': 1, 'name': 'John', 'amount': 100.50, 'active': True, 'date': '2024-01-01'},
            {'id': 2, 'name': 'Jane', 'amount': 200, 'active': False, 'date': '2024-01-02'},
            {'id': 3, 'name': 'Bob', 'amount': None, 'active': None, 'date': None}
        ]

        mock_reconciliation_storage = MagicMock()
        mock_reconciliation_storage.get_results.return_value = {
            'matched': mixed_data
        }

        with patch('app.services.reconciliation_service.optimized_reconciliation_storage', mock_reconciliation_storage):
            request_data = {
                "result_id": "mixed-types-id",
                "result_type": "matched",
                "file_format": "csv",
                "process_type": "reconciliation"
            }

            response = client.post("/save-results/save", json=request_data)

            assert response.status_code == 200
            data = response.json()
            saved_info = data["saved_file_info"]
            assert saved_info["total_rows"] == 3
            assert "data_types" in saved_info

    @pytest.mark.save_results
    def test_save_with_very_long_filename(self, client, mock_storage):
        """Test saving with very long custom filename"""
        mock_reconciliation_storage = MagicMock()
        mock_reconciliation_storage.get_results.return_value = {
            'matched': [{'id': 1, 'name': 'test'}]
        }

        very_long_filename = "a" * 200 + "_results"  # Very long filename

        with patch('app.services.reconciliation_service.optimized_reconciliation_storage', mock_reconciliation_storage):
            request_data = {
                "result_id": "long-filename-id",
                "result_type": "matched",
                "file_format": "csv",
                "process_type": "reconciliation",
                "custom_filename": very_long_filename
            }

            response = client.post("/save-results/save", json=request_data)

            # Should still succeed, filename gets sanitized
            assert response.status_code == 200
            data = response.json()
            saved_info = data["saved_file_info"]
            assert saved_info["filename"].endswith(".csv")

    @pytest.mark.save_results
    def test_concurrent_saves_different_ids(self, client, mock_storage):
        """Test saving multiple results concurrently (simulated)"""
        mock_reconciliation_storage = MagicMock()
        mock_reconciliation_storage.get_results.return_value = {
            'matched': [{'id': 1, 'name': 'test'}]
        }

        with patch('app.services.reconciliation_service.optimized_reconciliation_storage', mock_reconciliation_storage):
            # Simulate concurrent saves by making multiple rapid requests
            responses = []
            for i in range(5):
                request_data = {
                    "result_id": f"concurrent-test-{i}",
                    "result_type": "matched",
                    "file_format": "csv",
                    "process_type": "reconciliation"
                }

                response = client.post("/save-results/save", json=request_data)
                responses.append(response)

            # All should succeed
            for response in responses:
                assert response.status_code == 200

            # All should have unique file IDs
            file_ids = [r.json()["saved_file_info"]["saved_file_id"] for r in responses]
            assert len(set(file_ids)) == 5  # All unique


# Performance and stress tests
class TestSaveResultsPerformance:
    """Performance and stress tests"""

    @pytest.mark.save_results
    @pytest.mark.slow
    def test_save_performance_large_dataset(self, client, mock_storage):
        """Test performance with large dataset"""
        import time

        # Create dataset with 50k rows
        large_data = [{'id': i, 'name': f'user_{i}', 'value': i} for i in range(50000)]

        mock_reconciliation_storage = MagicMock()
        mock_reconciliation_storage.get_results.return_value = {
            'matched': large_data
        }

        with patch('app.services.reconciliation_service.optimized_reconciliation_storage', mock_reconciliation_storage):
            request_data = {
                "result_id": "performance-test",
                "result_type": "matched",
                "file_format": "csv",
                "process_type": "reconciliation"
            }

            start_time = time.time()
            response = client.post("/save-results/save", json=request_data)
            end_time = time.time()

            assert response.status_code == 200

            # Should complete within reasonable time (adjust threshold as needed)
            processing_time = end_time - start_time
            assert processing_time < 30  # Should complete within 30 seconds

            data = response.json()
            saved_info = data["saved_file_info"]
            assert saved_info["total_rows"] == 50000
