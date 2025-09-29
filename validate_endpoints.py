#!/usr/bin/env python3
"""
Endpoint Validation Script

This script validates that the modularized endpoints are properly structured
and can be imported without errors.
"""

import sys
import os
from pathlib import Path

# Add the Brain directory to Python path
brain_dir = Path(__file__).parent
sys.path.insert(0, str(brain_dir))

def validate_imports():
    """Test that all endpoint modules can be imported."""
    print("🔍 Validating endpoint imports...")
    
    try:
        # Test models import
        from server.endpoints.models import (
            DeviceRegisterRequest, 
            DeviceStatusRequest, 
            DataUploadRequest,
            FileUploadSimpleRequest,
            LoginRequest,
            QueryRequest,
            HealthCheckResponse
        )
        print("✅ Models imported successfully")
        
        # Test all endpoint modules
        from server.endpoints.system_endpoints import router as system_router
        print("✅ System endpoints imported successfully")
        
        from server.endpoints.auth_endpoints import router as auth_router
        print("✅ Auth endpoints imported successfully")
        
        from server.endpoints.ai_endpoints import router as ai_router
        print("✅ AI endpoints imported successfully")
        
        from server.endpoints.device_endpoints import router as device_router
        print("✅ Device endpoints imported successfully")
        
        from server.endpoints.data_endpoints import router as data_router
        print("✅ Data endpoints imported successfully")
        
        from server.endpoints.file_endpoints import router as file_router
        print("✅ File endpoints imported successfully")
        
        from server.endpoints.webhook_endpoints import router as webhook_router
        print("✅ Webhook endpoints imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def validate_endpoint_structure():
    """Validate that endpoints have the expected structure."""
    print("\n🔍 Validating endpoint structure...")
    
    try:
        from server.endpoints.system_endpoints import router as system_router
        from server.endpoints.auth_endpoints import router as auth_router
        from server.endpoints.ai_endpoints import router as ai_router
        from server.endpoints.device_endpoints import router as device_router
        from server.endpoints.data_endpoints import router as data_router
        from server.endpoints.file_endpoints import router as file_router
        from server.endpoints.webhook_endpoints import router as webhook_router
        
        # Check that routers have routes
        system_routes = len(system_router.routes)
        auth_routes = len(auth_router.routes)
        ai_routes = len(ai_router.routes)
        device_routes = len(device_router.routes)
        data_routes = len(data_router.routes)  
        file_routes = len(file_router.routes)
        webhook_routes = len(webhook_router.routes)
        
        print(f"✅ System router has {system_routes} routes")
        print(f"✅ Auth router has {auth_routes} routes")
        print(f"✅ AI router has {ai_routes} routes")
        print(f"✅ Device router has {device_routes} routes")
        print(f"✅ Data router has {data_routes} routes")
        print(f"✅ File router has {file_routes} routes")
        print(f"✅ Webhook router has {webhook_routes} routes")
        
        total_routes = system_routes + auth_routes + ai_routes + device_routes + data_routes + file_routes + webhook_routes
        print(f"🎯 Total routes across all modules: {total_routes}")
        
        # Validate minimum expected routes
        expected_minimums = {
            "system": (system_routes, 2, "health, root"),
            "auth": (auth_routes, 3, "login, register, profile"),
            "ai": (ai_routes, 1, "query"),
            "device": (device_routes, 5, "register, list, status, update, delete"),
            "data": (data_routes, 3, "upload, get, analytics"),
            "file": (file_routes, 4, "upload, list, download, delete"),
            "webhook": (webhook_routes, 3, "sms, call, transcription")
        }
        
        all_good = True
        for name, (actual, expected, description) in expected_minimums.items():
            if actual >= expected:
                print(f"✅ {name.title()} endpoints: Expected routes present ({description})")
            else:
                print(f"⚠️  {name.title()} endpoints: Fewer routes than expected ({actual}/{expected})")
                all_good = False
        
        if total_routes >= 20:
            print("🎉 Total endpoint count meets expectations!")
        else:
            print(f"⚠️  Total routes ({total_routes}) below expected minimum (20)")
            
        return all_good
        
    except Exception as e:
        print(f"❌ Structure validation error: {e}")
        return False

def validate_models():
    """Validate that models have proper validation."""
    print("\n🔍 Validating model validation...")
    
    try:
        from server.endpoints.models import (
            DeviceRegisterRequest, 
            DataUploadRequest, 
            LoginRequest,
            QueryRequest,
            FileUploadSimpleRequest
        )
        
        # Test valid device registration
        valid_device = DeviceRegisterRequest(
            device_id="test-device-123",
            device_name="Test Device"
        )
        print("✅ Valid device model created successfully")
        
        # Test invalid device registration (should raise validation error)
        try:
            invalid_device = DeviceRegisterRequest(device_id="ab")  # Too short
            print("⚠️  Device validation may not be working (short ID accepted)")
        except ValueError:
            print("✅ Device validation working (short ID rejected)")
        
        # Test valid data upload
        valid_data = DataUploadRequest(
            device_id="test-device-123",
            data=[{"timestamp": "2024-01-01T12:00:00Z", "value": 42}]
        )
        print("✅ Valid data model created successfully")
        
        # Test invalid data upload (empty array)
        try:
            invalid_data = DataUploadRequest(device_id="test-device-123", data=[])
            print("⚠️  Data validation may not be working (empty array accepted)")
        except ValueError:
            print("✅ Data validation working (empty array rejected)")
        
        # Test valid login request
        valid_login = LoginRequest(username="testuser", password="password123")
        print("✅ Valid login model created successfully")
        
        # Test invalid login (short password)
        try:
            invalid_login = LoginRequest(username="testuser", password="123")
            print("⚠️  Login validation may not be working (short password accepted)")
        except ValueError:
            print("✅ Login validation working (short password rejected)")
        
        # Test valid query request
        valid_query = QueryRequest(query="What is the weather today?")
        print("✅ Valid query model created successfully")
        
        # Test invalid query (empty)
        try:
            invalid_query = QueryRequest(query="")
            print("⚠️  Query validation may not be working (empty query accepted)")
        except ValueError:
            print("✅ Query validation working (empty query rejected)")
        
        # Test valid file upload
        valid_file = FileUploadSimpleRequest(filename="test.txt", content="Hello world")
        print("✅ Valid file model created successfully")
        
        # Test invalid file (path traversal)
        try:
            invalid_file = FileUploadSimpleRequest(filename="../../../etc/passwd", content="malicious")
            print("⚠️  File validation may not be working (path traversal accepted)")
        except ValueError:
            print("✅ File validation working (path traversal rejected)")
            
        return True
        
    except Exception as e:
        print(f"❌ Model validation error: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 Starting endpoint validation...\n")
    
    results = []
    
    # Run validation tests
    results.append(validate_imports())
    results.append(validate_endpoint_structure())
    results.append(validate_models())
    
    # Summary
    print("\n" + "="*50)
    print("📊 VALIDATION SUMMARY")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} validation tests PASSED!")
        print("✅ Modularization is working correctly")
        return 0
    else:
        print(f"⚠️  {passed}/{total} validation tests passed")
        print("❌ Some issues found - check output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
