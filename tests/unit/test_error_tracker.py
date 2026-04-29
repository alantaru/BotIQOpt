import os
import json
import pytest
from utils.ErrorTracker import ErrorTracker

@pytest.fixture
def error_tracker():
    return ErrorTracker(max_errors=5)

def test_add_error(error_tracker):
    error_tracker.add_error("API", "Connection failed", "traceback_string")
    assert len(error_tracker.errors) == 1
    assert error_tracker.error_counts["API"] == 1
    assert error_tracker.errors[0]["type"] == "API"

def test_add_critical_error(error_tracker):
    error_tracker.add_error("System", "Fatal error", critical=True)
    assert len(error_tracker.errors) == 1
    assert len(error_tracker.critical_errors) == 1
    assert error_tracker.critical_errors[0]["type"] == "System"

def test_max_errors(error_tracker):
    for i in range(10):
        error_tracker.add_error("Type", f"Message {i}")
    assert len(error_tracker.errors) == 5
    assert error_tracker.errors[-1]["message"] == "Message 9"

def test_get_errors_filtering(error_tracker):
    error_tracker.add_error("API", "Msg 1")
    error_tracker.add_error("Model", "Msg 2")
    error_tracker.add_error("API", "Msg 3", critical=True)
    
    assert len(error_tracker.get_errors(error_type="API")) == 2
    assert len(error_tracker.get_errors(critical_only=True)) == 1
    assert error_tracker.get_errors(error_type="API", limit=1)[0]["message"] == "Msg 3"

def test_clear_errors(error_tracker):
    error_tracker.add_error("API", "Msg 1")
    error_tracker.add_error("Model", "Msg 2")
    
    error_tracker.clear_errors(error_type="API")
    assert len(error_tracker.errors) == 1
    assert error_tracker.errors[0]["type"] == "Model"
    assert "API" not in error_tracker.error_counts
    
    error_tracker.clear_errors()
    assert len(error_tracker.errors) == 0

def test_persistence(error_tracker, tmp_path):
    test_file = tmp_path / "errors.json"
    error_tracker.add_error("Test", "Save me")
    error_tracker.save_to_file(str(test_file))
    
    new_tracker = ErrorTracker()
    assert new_tracker.load_from_file(str(test_file))
    assert len(new_tracker.errors) == 1
    assert new_tracker.errors[0]["message"] == "Save me"

def test_persistence_edge_cases(error_tracker, tmp_path):
    # Test get_error_counts (Line 95)
    error_tracker.add_error("TypeA", "Msg")
    counts = error_tracker.get_error_counts()
    assert counts["TypeA"] == 1
    
    # Test load_from_file case: file not found (Lines 147-148)
    assert error_tracker.load_from_file("non_existent_file.json") is False
    
    # Test load_from_file case: invalid JSON (Lines 159-161)
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, "w") as f:
        f.write("invalid json {")
    assert error_tracker.load_from_file(str(invalid_file)) is False
    
    # Test save_to_file case: exception (Lines 132-134)
    # Using a directory name as a filename triggers an error on open() for writing
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    assert error_tracker.save_to_file(str(test_dir)) is False

