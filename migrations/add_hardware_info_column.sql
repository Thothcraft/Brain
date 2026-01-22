-- Migration: Add hardware_info column to device table
-- This column stores JSON data about device type, sensors, and capabilities
-- Date: 2026-01-21

-- Add hardware_info column to device table if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'device' AND column_name = 'hardware_info'
    ) THEN
        ALTER TABLE device ADD COLUMN hardware_info TEXT;
        COMMENT ON COLUMN device.hardware_info IS 'JSON string containing device type, sensors, and hardware capabilities';
    END IF;
END $$;

-- Example of what hardware_info JSON looks like:
-- {
--   "device_type": "laptop",  -- laptop, desktop, mobile, raspberry_pi, unknown
--   "is_raspberry_pi": false,
--   "raspberry_pi_model": null,
--   "sensors": [
--     {
--       "sensor_type": "camera",
--       "name": "Camera",
--       "available": true,
--       "device_path": "/dev/video0",
--       "capabilities": {"width": 1920, "height": 1080, "fps": 30}
--     },
--     {
--       "sensor_type": "microphone",
--       "name": "Microphone",
--       "available": true,
--       "capabilities": {"channels": 1, "sample_rate": 44100}
--     },
--     {
--       "sensor_type": "imu",
--       "name": "IMU",
--       "available": false,
--       "error": "No IMU/motion sensors detected"
--     }
--   ],
--   "system": "Windows",
--   "processor": "Intel64 Family 6 Model 142",
--   "cpu_count": 8,
--   "memory": {"total": 17179869184, "available": 8589934592, "percent": 50.0},
--   "hostname": "DESKTOP-ABC123"
-- }
