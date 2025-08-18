# tests/conftest_save_results.py - Configuration specifically for save results tests
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import the base test storage from main conftest
from .conftest import test_uploaded_files

# Import save results routes
try:
    from app.routes.save_results_routes import router as save_results_router

    HAS_SAVE_RESULTS = True
except ImportError as e:
    print(f"Warning: Could not import save_results_routes: {e}")
    save_results_router = None
    HAS_SAVE_RESULTS = False


@pytest.fixture(scope="session")
def save_results_app():
    """Create FastAPI app with save results routes only"""
    if not HAS_SAVE_RESULTS:
        pytest.skip("save_results_routes not available")

    test_app = FastAPI(title="Save Results Test App")
    test_app.include_router(save_results_router)
    return test_app


@pytest.fixture
def save_results_client(save_results_app):
    """Create test client for save results routes"""
    return TestClient(save_results_app)


@pytest.fixture(autouse=True, scope="function")
def mock_save_results_storage():
    """Mock storage specifically for save results tests"""
    test_uploaded_files.clear()

    # Patch all the storage references that save_results might use
    with patch('app.services.storage_service.uploaded_files', test_uploaded_files), \
            patch('app.routes.save_results_routes.uploaded_files', test_uploaded_files):
        yield test_uploaded_files


@pytest.fixture
def mock_reconciliation_storage():
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
def mock_delta_storage():
    """Mock delta storage with sample data"""
    return {
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


@pytest.fixture
def mock_storage_info():
    """Mock storage info function"""
    return {
        "backend_class": "InMemoryStorage",
        "storage_type": "memory"
    }


@pytest.fixture
def sample_save_request():
    """Sample save request data"""
    return {
        "result_id": "test-reconciliation-id",
        "result_type": "matched",
        "file_format": "csv",
        "process_type": "reconciliation",
        "description": "Test saved results"
    }


# Skip all save results tests if the module doesn't exist
def pytest_runtest_setup(item):
    """Skip save results tests if module not available"""
    if "test_save_results" in item.nodeid and not HAS_SAVE_RESULTS:
        pytest.skip("save_results_routes module not available")
