#!/usr/bin/env python3
"""
Test script untuk memverifikasi implementasi scan logs.
"""
import os
import sys
import sqlite3
from datetime import datetime

# Add app directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_database_structure():
    """Test apakah table scan_logs ada dengan kolom yang benar"""
    print("=" * 60)
    print("TEST 1: Check scan_logs table structure")
    print("=" * 60)
    
    db_path = os.path.join(os.path.dirname(__file__), "database.db")
    
    if not os.path.exists(db_path):
        print("❌ Database file tidak ditemukan!")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check if scan_logs table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scan_logs'")
        if not cursor.fetchone():
            print("❌ Table scan_logs tidak ada!")
            return False
        
        print("✅ Table scan_logs ada")
        
        # Check columns
        cursor.execute("PRAGMA table_info(scan_logs)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        required_columns = ['timestamp', 'status', 'ip_address', 'nik', 'name', 'dob', 'address', 'age', 'message']
        
        for col in required_columns:
            if col in columns:
                print(f"  ✅ Column '{col}' exists ({columns[col]})")
            else:
                print(f"  ❌ Column '{col}' missing!")
                return False
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_imports():
    """Test apakah app.py bisa di-import"""
    print("\n" + "=" * 60)
    print("TEST 2: Check if app.py can be imported")
    print("=" * 60)
    
    try:
        import app
        print("✅ app.py imported successfully")
        
        # Check if log_scan_result function exists
        if hasattr(app, 'log_scan_result'):
            print("✅ Function log_scan_result exists")
        else:
            print("❌ Function log_scan_result not found!")
            return False
        
        # Check if constants exist
        if hasattr(app, 'MIN_VALID_FRAMES'):
            print(f"✅ MIN_VALID_FRAMES = {app.MIN_VALID_FRAMES}")
        else:
            print("❌ MIN_VALID_FRAMES not found!")
            return False
            
        if hasattr(app, 'LBPH_CONF_THRESHOLD'):
            print(f"✅ LBPH_CONF_THRESHOLD = {app.LBPH_CONF_THRESHOLD}")
        else:
            print("❌ LBPH_CONF_THRESHOLD not found!")
            return False
            
        if hasattr(app, 'model_lock'):
            print("✅ model_lock exists")
        else:
            print("❌ model_lock not found!")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error importing app: {e}")
        return False


def test_endpoint_exists():
    """Test apakah endpoint /api/scan_logs terdaftar"""
    print("\n" + "=" * 60)
    print("TEST 3: Check if /api/scan_logs endpoint exists")
    print("=" * 60)
    
    try:
        import app
        
        # Get all registered routes
        routes = [rule.rule for rule in app.app.url_map.iter_rules()]
        
        if '/api/scan_logs' in routes:
            print("✅ Endpoint /api/scan_logs is registered")
            return True
        else:
            print("❌ Endpoint /api/scan_logs not found!")
            print("\nAvailable /api/* endpoints:")
            for route in sorted(routes):
                if route.startswith('/api/'):
                    print(f"  - {route}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_manual_log():
    """Test mencatat log secara manual"""
    print("\n" + "=" * 60)
    print("TEST 4: Test manual log insertion")
    print("=" * 60)
    
    try:
        import app
        from flask import Flask
        
        # Create test context
        with app.app.test_request_context():
            # Test logging
            app.log_scan_result(
                status="success",
                nik="3571234567890123",
                name="Test User",
                dob="1990-01-15",
                address="Test Address",
                age="34 Tahun",
                message="Test log entry"
            )
            print("✅ Log entry created successfully")
        
        # Verify log was saved
        db_path = os.path.join(os.path.dirname(__file__), "database.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM scan_logs ORDER BY timestamp DESC LIMIT 1")
        log = cursor.fetchone()
        
        if log:
            print("\n📋 Latest log entry:")
            print(f"  Timestamp: {log['timestamp']}")
            print(f"  Status: {log['status']}")
            print(f"  NIK: {log['nik']}")
            print(f"  Name: {log['name']}")
            print(f"  Message: {log['message']}")
            return True
        else:
            print("❌ No log entries found!")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " SCAN LOGS IMPLEMENTATION TEST ".center(58) + "║")
    print("╚" + "=" * 58 + "╝\n")
    
    tests = [
        test_database_structure,
        test_imports,
        test_endpoint_exists,
        test_manual_log,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed! Scan logs implementation is working correctly.")
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please check the output above.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
