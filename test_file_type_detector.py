"""Test script for file type detection module."""

import sys
import os

# Add server to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.file_type_detector import (
    detect_file_type,
    DetectedFileType,
    CSI_HEADER_START,
    _is_csi_header,
)


def test_csi_detection():
    """Test CSI file detection by header."""
    print("\n=== Testing CSI Detection ===")
    
    # CSI header
    csi_header = "type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data"
    csi_data_row = "CSI_DATA,1,aa:bb:cc:dd:ee:ff,-50,11,0,0,0,6,1234567890,128,0,128,0,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128]"
    
    csi_content = f"{csi_header}\n{csi_data_row}\n{csi_data_row}\n".encode('utf-8')
    
    # Test with arbitrary filename (not csi_*)
    result = detect_file_type(csi_content, "my_random_data.csv")
    
    print(f"Filename: my_random_data.csv")
    print(f"Detected type: {result.detected_type.value}")
    print(f"Is CSI: {result.is_csi}")
    print(f"Confidence: {result.confidence}")
    print(f"Detection method: {result.detection_method}")
    print(f"CSI array length: {result.csi_array_length}")
    
    assert result.detected_type == DetectedFileType.CSI, f"Expected CSI, got {result.detected_type}"
    assert result.is_csi == True, "Expected is_csi=True"
    print("✓ CSI detection passed!")
    
    return True


def test_general_csv_detection():
    """Test general CSV file detection."""
    print("\n=== Testing General CSV Detection ===")
    
    # General CSV with arbitrary columns
    csv_content = """timestamp,temperature,humidity,pressure
2025-01-01 00:00:00,25.5,60.2,1013.25
2025-01-01 00:01:00,25.6,60.1,1013.30
2025-01-01 00:02:00,25.4,60.3,1013.20
""".encode('utf-8')
    
    result = detect_file_type(csv_content, "sensor_readings.csv")
    
    print(f"Filename: sensor_readings.csv")
    print(f"Detected type: {result.detected_type.value}")
    print(f"Is CSI: {result.is_csi}")
    print(f"Confidence: {result.confidence}")
    print(f"Header columns: {result.header_columns}")
    print(f"Column types: {result.statistics.get('column_types', {})}")
    
    assert result.detected_type == DetectedFileType.GENERAL_CSV, f"Expected GENERAL_CSV, got {result.detected_type}"
    assert result.is_csi == False, "Expected is_csi=False"
    assert result.header_columns == ['timestamp', 'temperature', 'humidity', 'pressure']
    print("✓ General CSV detection passed!")
    
    return True


def test_imu_json_detection():
    """Test IMU JSON file detection."""
    print("\n=== Testing IMU JSON Detection ===")
    
    imu_content = """[
        {"accel_x": 0.1, "accel_y": 0.2, "accel_z": 9.8, "gyro_x": 0.01, "gyro_y": 0.02, "gyro_z": 0.03},
        {"accel_x": 0.15, "accel_y": 0.25, "accel_z": 9.75, "gyro_x": 0.015, "gyro_y": 0.025, "gyro_z": 0.035}
    ]""".encode('utf-8')
    
    result = detect_file_type(imu_content, "motion_data.json")
    
    print(f"Filename: motion_data.json")
    print(f"Detected type: {result.detected_type.value}")
    print(f"Confidence: {result.confidence}")
    print(f"Statistics: {result.statistics}")
    
    assert result.detected_type == DetectedFileType.IMU, f"Expected IMU, got {result.detected_type}"
    print("✓ IMU JSON detection passed!")
    
    return True


def test_image_detection():
    """Test image file detection by extension."""
    print("\n=== Testing Image Detection ===")
    
    # PNG magic bytes
    png_content = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
    
    result = detect_file_type(png_content, "photo.png")
    
    print(f"Filename: photo.png")
    print(f"Detected type: {result.detected_type.value}")
    print(f"Confidence: {result.confidence}")
    
    assert result.detected_type == DetectedFileType.IMAGE, f"Expected IMAGE, got {result.detected_type}"
    print("✓ Image detection passed!")
    
    return True


def test_csi_header_matching():
    """Test CSI header matching function."""
    print("\n=== Testing CSI Header Matching ===")
    
    # Valid CSI header
    valid_header = ["type", "seq", "mac", "rssi", "rate", "noise_floor", "fft_gain", 
                   "agc_gain", "channel", "local_timestamp", "sig_len", "rx_state", 
                   "len", "first_word", "data"]
    
    assert _is_csi_header(valid_header) == True, "Should match valid CSI header"
    print("✓ Valid CSI header matched")
    
    # Invalid header (missing columns)
    invalid_header = ["timestamp", "temperature", "humidity"]
    assert _is_csi_header(invalid_header) == False, "Should not match invalid header"
    print("✓ Invalid header rejected")
    
    # Partial CSI header (not enough columns)
    partial_header = ["type", "seq", "mac", "rssi"]
    assert _is_csi_header(partial_header) == False, "Should not match partial header"
    print("✓ Partial header rejected")
    
    print("✓ CSI header matching passed!")
    return True


def test_arbitrary_filename():
    """Test that detection works regardless of filename."""
    print("\n=== Testing Arbitrary Filename Detection ===")
    
    # CSI content with non-standard filename
    csi_header = "type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data"
    csi_data = "CSI_DATA,1,aa:bb:cc:dd:ee:ff,-50,11,0,0,0,6,1234567890,128,0,128,0,[" + ",".join(str(i) for i in range(128)) + "]"
    csi_content = f"{csi_header}\n{csi_data}\n".encode('utf-8')
    
    # Test with various arbitrary filenames
    test_filenames = [
        "experiment_2025_01_24.csv",
        "data_collection_session_1.csv",
        "wifi_sensing_raw.csv",
        "user_uploaded_file.csv",
        "123456.csv",
    ]
    
    for filename in test_filenames:
        result = detect_file_type(csi_content, filename)
        print(f"  {filename}: {result.detected_type.value} (CSI={result.is_csi})")
        assert result.detected_type == DetectedFileType.CSI, f"Expected CSI for {filename}"
        assert result.is_csi == True
    
    print("✓ Arbitrary filename detection passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("File Type Detector Tests")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_csi_header_matching()
        all_passed &= test_csi_detection()
        all_passed &= test_general_csv_detection()
        all_passed &= test_imu_json_detection()
        all_passed &= test_image_detection()
        all_passed &= test_arbitrary_filename()
        
        print("\n" + "=" * 60)
        if all_passed:
            print("All tests PASSED! ✓")
        else:
            print("Some tests FAILED! ✗")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
