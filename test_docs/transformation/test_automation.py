#!/usr/bin/env python3
"""
Comprehensive Transformation Testing Automation Script

This script automates the testing of all transformation scenarios,
including the critical account_summary field generation issue.
"""

import requests
import json
import csv
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class TestStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"

@dataclass
class TestResult:
    test_id: str
    name: str
    status: TestStatus
    message: str
    details: Dict[str, Any]
    execution_time: float

class TransformationTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.uploaded_files = {}
        self.test_results = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def check_server_health(self) -> bool:
        """Check if the server is running and healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                self.log("✅ Server health check passed")
                return True
            else:
                self.log(f"❌ Server health check failed: {response.status_code}", "ERROR") 
                return False
        except Exception as e:
            self.log(f"❌ Server health check failed: {str(e)}", "ERROR")
            return False
    
    def upload_test_file(self, file_path: str, file_type: str = "csv") -> Optional[str]:
        """Upload a test file and return file_id"""
        try:
            if not os.path.exists(file_path):
                self.log(f"❌ Test file not found: {file_path}", "ERROR")
                return None
            
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'text/csv')}
                data = {'file_type': file_type}
                
                response = self.session.post(
                    f"{self.base_url}/upload",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                file_id = result.get('file_id')
                filename = os.path.basename(file_path)
                self.uploaded_files[filename] = file_id
                self.log(f"✅ Uploaded {filename} -> {file_id}")
                return file_id
            else:
                self.log(f"❌ Upload failed for {file_path}: {response.status_code}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"❌ Upload error for {file_path}: {str(e)}", "ERROR")
            return None
    
    def run_transformation(self, config: Dict[str, Any], file_ids: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Run a transformation with the given configuration"""
        try:
            # Replace file_id placeholders in source_files
            source_files = []
            for source_file in config.get('source_files', []):
                if source_file['file_id'] in file_ids:
                    source_file['file_id'] = file_ids[source_file['file_id']]
                    source_files.append(source_file)
                else:
                    self.log(f"❌ File ID not found: {source_file['file_id']}", "ERROR")
                    return None
            
            request_data = {
                "source_files": source_files,
                "transformation_config": config['transformation_config']
            }
            
            response = self.session.post(
                f"{self.base_url}/transformation/process",
                json=request_data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.log(f"❌ Transformation failed: {response.status_code} - {response.text}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"❌ Transformation error: {str(e)}", "ERROR")
            return None
    
    def validate_result(self, result_data: List[Dict], validation_rules: List[Dict]) -> TestResult:
        """Validate transformation results against rules"""
        validation_errors = []
        validation_warnings = []
        
        if not result_data:
            return TestResult(
                test_id="",
                name="Validation",
                status=TestStatus.FAILED,
                message="No result data to validate",
                details={},
                execution_time=0.0
            )
        
        for rule in validation_rules:
            field = rule['field']
            rule_type = rule['rule']
            expected_value = rule.get('value')
            required = rule.get('required', False)
            critical = rule.get('critical', False)
            
            # Count violations
            violations = 0
            samples = []
            
            for i, row in enumerate(result_data):
                field_value = row.get(field, "")
                violation = False
                
                if rule_type == "not_empty":
                    if not field_value or str(field_value).strip() == "":
                        violation = True
                        samples.append(f"Row {i}: empty value")
                
                elif rule_type == "equals":
                    if str(field_value) != str(expected_value):
                        violation = True
                        samples.append(f"Row {i}: '{field_value}' != '{expected_value}'")
                
                elif rule_type == "contains":
                    if expected_value not in str(field_value):
                        violation = True
                        samples.append(f"Row {i}: '{field_value}' doesn't contain '{expected_value}'")
                
                elif rule_type == "starts_with":
                    if not str(field_value).startswith(str(expected_value)):
                        violation = True
                        samples.append(f"Row {i}: '{field_value}' doesn't start with '{expected_value}'")
                
                elif rule_type == "is_numeric":
                    try:
                        float(field_value)
                    except (ValueError, TypeError):
                        violation = True
                        samples.append(f"Row {i}: '{field_value}' is not numeric")
                
                elif rule_type == "in_list":
                    if field_value not in expected_value:
                        violation = True
                        samples.append(f"Row {i}: '{field_value}' not in {expected_value}")
                
                if violation:
                    violations += 1
            
            # Record violations
            if violations > 0:
                error_msg = f"Field '{field}' validation failed: {violations}/{len(result_data)} rows violated '{rule_type}' rule"
                if samples:
                    error_msg += f". Samples: {samples[:3]}"
                
                if critical or required:
                    validation_errors.append(error_msg)
                else:
                    validation_warnings.append(error_msg)
        
        # Determine overall status
        if validation_errors:
            status = TestStatus.FAILED
            message = f"{len(validation_errors)} critical validation errors"
        elif validation_warnings:
            status = TestStatus.WARNING
            message = f"{len(validation_warnings)} validation warnings"
        else:
            status = TestStatus.PASSED
            message = "All validations passed"
        
        return TestResult(
            test_id="",
            name="Validation",
            status=status,
            message=message,
            details={
                "errors": validation_errors,
                "warnings": validation_warnings,
                "total_rows": len(result_data),
                "sample_data": result_data[:3] if result_data else []
            },
            execution_time=0.0
        )
    
    def run_account_summary_test(self) -> TestResult:
        """Specifically test the account_summary field generation issue"""
        start_time = time.time()
        
        self.log("🔍 Running CRITICAL account_summary test...")
        
        # Upload customer file
        file_id = self.upload_test_file("test_data/customers_test.csv")
        if not file_id:
            return TestResult(
                test_id="ACCOUNT_SUMMARY_TEST",
                name="Account Summary Critical Test",
                status=TestStatus.FAILED,
                message="Failed to upload test file",
                details={},
                execution_time=time.time() - start_time
            )
        
        # Configuration specifically for account_summary testing
        config = {
            "source_files": [
                {
                    "file_id": file_id,
                    "alias": "customers",
                    "purpose": "Test account_summary generation"
                }
            ],
            "transformation_config": {
                "name": "Account Summary Test",
                "description": "Critical test for account_summary field",
                "row_generation_rules": [
                    {
                        "id": "rule_account_summary",
                        "name": "Account Summary Generation",
                        "enabled": True,
                        "priority": 0,
                        "condition": "",
                        "output_columns": [
                            {
                                "id": "col_001",
                                "name": "customer_id",
                                "mapping_type": "direct",
                                "source_column": "Customer_ID"
                            },
                            {
                                "id": "col_002",
                                "name": "account_summary",
                                "mapping_type": "static",
                                "static_value": "{Account_Type} account with balance ${Balance}"
                            },
                            {
                                "id": "col_003",
                                "name": "debug_account_type",
                                "mapping_type": "direct",
                                "source_column": "Account_Type"
                            },
                            {
                                "id": "col_004",
                                "name": "debug_balance",
                                "mapping_type": "direct",
                                "source_column": "Balance"
                            }
                        ]
                    }
                ]
            }
        }
        
        # Run transformation
        result = self.run_transformation(config, {file_id: file_id})
        if not result:
            return TestResult(
                test_id="ACCOUNT_SUMMARY_TEST",
                name="Account Summary Critical Test",
                status=TestStatus.FAILED,
                message="Transformation execution failed",
                details={},
                execution_time=time.time() - start_time
            )
        
        # Extract result data
        result_data = []
        if 'data' in result:
            result_data = result['data']
        elif 'results' in result and len(result['results']) > 0 and 'data' in result['results'][0]:
            result_data = result['results'][0]['data']
        
        # Detailed validation for account_summary
        validation_details = {
            "total_rows": len(result_data),
            "account_summary_analysis": {},
            "debug_info": {},
            "sample_results": result_data[:5] if result_data else []
        }
        
        empty_count = 0
        valid_format_count = 0
        account_summary_samples = []
        
        for i, row in enumerate(result_data):
            account_summary = row.get("account_summary", "")
            debug_account_type = row.get("debug_account_type", "")
            debug_balance = row.get("debug_balance", "")
            
            if not account_summary or str(account_summary).strip() == "":
                empty_count += 1
            else:
                account_summary_samples.append(account_summary)
                if "account with balance $" in account_summary:
                    valid_format_count += 1
        
        validation_details["account_summary_analysis"] = {
            "empty_count": empty_count,
            "valid_format_count": valid_format_count,
            "samples": account_summary_samples[:10]
        }
        
        # Determine test result
        if empty_count > 0:
            status = TestStatus.FAILED
            message = f"CRITICAL: {empty_count}/{len(result_data)} account_summary fields are empty"
        elif valid_format_count < len(result_data):
            status = TestStatus.WARNING
            message = f"Format issues: only {valid_format_count}/{len(result_data)} have valid format"
        else:
            status = TestStatus.PASSED
            message = f"SUCCESS: All {len(result_data)} account_summary fields generated correctly"
        
        return TestResult(
            test_id="ACCOUNT_SUMMARY_TEST",
            name="Account Summary Critical Test",
            status=status,
            message=message,
            details=validation_details,
            execution_time=time.time() - start_time
        )
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all transformation tests"""
        self.log("🚀 Starting comprehensive transformation testing...")
        
        # Health check
        if not self.check_server_health():
            return [TestResult(
                test_id="HEALTH_CHECK",
                name="Server Health Check",
                status=TestStatus.FAILED,
                message="Server is not accessible",
                details={},
                execution_time=0.0
            )]
        
        # Upload test files
        self.log("📁 Uploading test files...")
        customers_file_id = self.upload_test_file("test_data/customers_test.csv")
        
        if not customers_file_id:
            return [TestResult(
                test_id="FILE_UPLOAD",
                name="Test File Upload",
                status=TestStatus.FAILED,
                message="Failed to upload required test files",
                details={},
                execution_time=0.0
            )]
        
        # Run critical account_summary test
        account_summary_result = self.run_account_summary_test()
        self.test_results.append(account_summary_result)
        
        # Load test scenarios
        try:
            with open("test_scenarios/advanced_transformation_tests.json", 'r') as f:
                test_scenarios = json.load(f)
        except Exception as e:
            self.log(f"❌ Could not load test scenarios: {str(e)}", "ERROR")
            return self.test_results
        
        # Run each test scenario
        for scenario in test_scenarios.get("test_scenarios", []):
            self.log(f"🧪 Running {scenario['name']} ({scenario['id']})...")
            
            start_time = time.time()
            
            # Prepare file mappings
            file_mappings = {"customers_file_id": customers_file_id}
            
            # Update source files in config
            config = scenario.copy()
            for source_file in config['transformation_config'].get('source_files', []):
                if source_file['file_id'] in file_mappings:
                    source_file['file_id'] = file_mappings[source_file['file_id']]
            
            # Run transformation
            result = self.run_transformation(config, file_mappings)
            
            if result:
                # Extract result data
                result_data = []
                if 'data' in result:
                    result_data = result['data']
                elif 'results' in result and len(result['results']) > 0:
                    result_data = result['results'][0].get('data', [])
                
                # Validate results
                validation_result = self.validate_result(result_data, scenario.get('validation_rules', []))
                validation_result.test_id = scenario['id']
                validation_result.name = scenario['name']
                validation_result.execution_time = time.time() - start_time
                
                self.test_results.append(validation_result)
            else:
                self.test_results.append(TestResult(
                    test_id=scenario['id'],
                    name=scenario['name'],
                    status=TestStatus.FAILED,
                    message="Transformation execution failed",
                    details={},
                    execution_time=time.time() - start_time
                ))
        
        return self.test_results
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report"""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE TRANSFORMATION TEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        passed = len([r for r in self.test_results if r.status == TestStatus.PASSED])
        failed = len([r for r in self.test_results if r.status == TestStatus.FAILED])
        warnings = len([r for r in self.test_results if r.status == TestStatus.WARNING])
        total = len(self.test_results)
        
        report.append(f"SUMMARY: {passed} PASSED | {failed} FAILED | {warnings} WARNINGS | {total} TOTAL")
        report.append("")
        
        # Critical account_summary test result
        account_summary_test = next((r for r in self.test_results if r.test_id == "ACCOUNT_SUMMARY_TEST"), None)
        if account_summary_test:
            report.append("🔥 CRITICAL ISSUE STATUS:")
            report.append(f"   Account Summary Test: {account_summary_test.status.value}")
            report.append(f"   Message: {account_summary_test.message}")
            if account_summary_test.status == TestStatus.FAILED:
                report.append("   ❌ THE ACCOUNT_SUMMARY FIELD GENERATION IS BROKEN!")
            else:
                report.append("   ✅ Account summary field generation working correctly")
            report.append("")
        
        # Detailed results
        report.append("DETAILED TEST RESULTS:")
        report.append("-" * 80)
        
        for result in self.test_results:
            status_icon = {
                TestStatus.PASSED: "✅",
                TestStatus.FAILED: "❌", 
                TestStatus.WARNING: "⚠️",
                TestStatus.SKIPPED: "⏭️"
            }.get(result.status, "❓")
            
            report.append(f"{status_icon} {result.name} ({result.test_id})")
            report.append(f"   Status: {result.status.value}")
            report.append(f"   Message: {result.message}")
            report.append(f"   Execution Time: {result.execution_time:.2f}s")
            
            if result.details:
                report.append("   Details:")
                for key, value in result.details.items():
                    if isinstance(value, list) and len(value) > 3:
                        report.append(f"     {key}: {value[:3]}... ({len(value)} total)")
                    else:
                        report.append(f"     {key}: {value}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main function to run all tests"""
    tester = TransformationTester()
    
    print("🧪 Comprehensive Transformation Testing Framework")
    print("=" * 60)
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Generate and display report
    report = tester.generate_report()
    print(report)
    
    # Save report to file
    with open("transformation_test_report.txt", "w") as f:
        f.write(report)
    
    print(f"\n📊 Full report saved to: transformation_test_report.txt")
    
    # Exit code based on test results
    failed_tests = [r for r in results if r.status == TestStatus.FAILED]
    if failed_tests:
        print(f"❌ {len(failed_tests)} tests failed!")
        exit(1)
    else:
        print("✅ All tests passed!")
        exit(0)


if __name__ == "__main__":
    main()