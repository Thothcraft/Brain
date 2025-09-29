import os
import pytest
import logging
import requests
import uuid
import shutil

logger = logging.getLogger(__name__)

# Define server URLs
LOCAL_URL = "http://localhost:7050"
# Canonical Railway backend deployment URL
RAILWAY_URL = "https://web-production-d7d37.up.railway.app"

# Get the target environment from environment variable
# To run tests locally, use: TEST_LOCAL=1 pytest test.py
@pytest.fixture(scope="session", autouse=True)
def base_url():
    use_local = os.environ.get("TEST_LOCAL", "") in ("1", "true", "yes")
    url = LOCAL_URL if use_local else RAILWAY_URL
    logger.info(f"Testing against: {url} {'(LOCAL)' if use_local else '(VERCEL)'}")
    return url

def test_server_health(base_url):
    """Test if the server is reachable and responds to a simple request"""
    logger.info("[test_server_health] Testing server health")
    
    # Test the profile endpoint which should be static and not require DB access
    resp = requests.get(f"{base_url}/profile")
    logger.info(f"[test_server_health] Profile endpoint response: {resp.status_code}, {resp.text}")
    
    # Try to get the root endpoint
    root_resp = requests.get(f"{base_url}/")
    logger.info(f"[test_server_health] Root endpoint response: {root_resp.status_code}, {root_resp.text[:100]}")
    
    # If we get here without exceptions, the server is responding
    logger.info("[test_server_health] Server is responding to requests")

# For local file handling, we still need some imports
from server.routes import ASSETS_FOLDER

import uuid

def getUniqueUser(with_phone_number=False):
    username = f"testuser_{uuid.uuid4().hex[:8]}"
    user_data = {"username": username, "password": "testpass123"}
    if with_phone_number:
        # Generate a unique-ish phone number string, then convert to int. Ensure it's a plausible length.
        # Using a large number to test BigInteger, ensuring it doesn't start with 0.
        phone_digits = f"{uuid.uuid4().int % 10**9}" # up to 9 digits
        user_data["phone_number"] = int(f"1{phone_digits.zfill(9)}") # Make it a 10-digit number starting with 1
    return user_data

TEST_FILE_NAME = "testfile.txt"
TEST_FILE_CONTENT = b"Hello, LMS!"

import shutil

def setup_module(module):
    # Clean up assets folder before tests
    if os.path.exists(ASSETS_FOLDER):
        for f in os.listdir(ASSETS_FOLDER):
            path = os.path.join(ASSETS_FOLDER, f)
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)

def registerUser(base_url, user_payload=None):
    logger.info(f"[registerUser] Registering user with payload: {user_payload}")
    if user_payload is None:
        user_payload = getUniqueUser()
    
    logger.info(f"[registerUser] Sending request to {base_url}/register with user data: {user_payload}")
    resp = requests.post(f"{base_url}/register", json=user_payload)
    logger.info(f"[registerUser] Response status: {resp.status_code}, raw response: {resp.text}")
    
    if resp.status_code != 200:
        # For duplicate phone number test, we might expect a different error code
        # but for general registration, 200 is success.
        # Let the test itself assert specific non-200 codes if expected.
        logger.warning(f"[registerUser] Registration did not return 200. Status: {resp.status_code}, Response: {resp.text}")
        # We return the response object so the test can inspect it
        return user_payload, resp 
        
    try:
        json_resp = resp.json()
        logger.info(f"[registerUser] JSON response: {json_resp}")
        assert json_resp["message"] == "Registered successfully"
        return user_payload, resp # Return the original payload and the response
    except Exception as e:
        logger.error(f"[registerUser] Error parsing response: {e}")

def loginUser(base_url, user_payload_dict):
    logger.info(f"[loginUser] Logging in user: {user_payload_dict['username']}") # user_payload_dict is the dict
    login_data = {"username": user_payload_dict["username"], "password": user_payload_dict["password"]}
    resp = requests.post(
        f"{base_url}/token", 
        data=login_data, 
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    logger.info(f"[loginUser] Response status: {resp.status_code}, body: {resp.json()}")
    assert resp.status_code == 200, f"[loginUser] Login failed: {resp.text}"
    assert "access_token" in resp.json()
    return resp.json()["access_token"]

def test_upload_file(base_url):
    logger.info("[test_upload_file] START")
    user_payload, reg_resp = registerUser(base_url)
    assert reg_resp.status_code == 200, f"Registration failed: {reg_resp.text}"
    token = loginUser(base_url, user_payload) # Pass the dict part
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": (TEST_FILE_NAME, TEST_FILE_CONTENT, "text/plain")}
    logger.info(f"[test_upload_file] Uploading file: {TEST_FILE_NAME}")
    resp = requests.post(f"{base_url}/upload", headers=headers, files=files)
    logger.info(f"[test_upload_file] Response status: {resp.status_code}, body: {resp.json()}")
    assert resp.status_code == 200, f"[test_upload_file] Upload failed: {resp.text}"
    assert resp.json()["filename"] == TEST_FILE_NAME
    assert resp.json()["size"] == len(TEST_FILE_CONTENT)
    # Clean up: delete user
    logger.info("[test_upload_file] END")
    requests.delete(f"{base_url}/user/{user_payload['username']}", headers=headers)

def test_active_url_success(base_url):
    logger.info("[test_active_url_success] START")
    payload = {"url": "https://example.com", "title": "Example Page"}
    logger.info(f"[test_active_url_success] Payload: {payload}")
    resp = requests.post(f"{base_url}/active_url", json=payload)
    logger.info(f"[test_active_url_success] Response status: {resp.status_code}, body: {resp.json()}")
    assert resp.status_code == 200, f"[test_active_url_success] Failed: {resp.text}"
    assert resp.json() == {"data": {"status": "success"}}
    logger.info("[test_active_url_success] END")

def test_active_url_missing_url(base_url):
    logger.info("[test_active_url_missing_url] START")
    payload = {"title": "Example Page"}
    logger.info(f"[test_active_url_missing_url] Payload: {payload}")
    resp = requests.post(f"{base_url}/active_url", json=payload)
    logger.info(f"[test_active_url_missing_url] Response status: {resp.status_code}, body: {resp.json()}")
    assert resp.status_code == 400, f"[test_active_url_missing_url] Failed: {resp.text}"
    assert "error" in resp.json()
    logger.info("[test_active_url_missing_url] END")

def test_profile(base_url):
    logger.info("[test_profile] START")
    user_payload_with_phone = getUniqueUser(with_phone_number=True)
    registered_user_with_phone, reg_resp_wp = registerUser(base_url, user_payload_with_phone)
    assert reg_resp_wp.status_code == 200, f"Registration failed for user with phone: {reg_resp_wp.text}"
    token_wp = loginUser(base_url, registered_user_with_phone)
    headers_wp = {"Authorization": f"Bearer {token_wp}"}

    resp_wp = requests.get(f"{base_url}/profile", headers=headers_wp)
    logger.info(f"[test_profile] Profile response (with phone): {resp_wp.status_code}, body: {resp_wp.json()}")
    assert resp_wp.status_code == 200
    data_wp = resp_wp.json()
    assert data_wp["username"] == registered_user_with_phone["username"]
    assert "userId" in data_wp
    assert "max_file_size" in data_wp
    assert "role" in data_wp
    assert "phone_number" in data_wp
    assert data_wp["phone_number"] == registered_user_with_phone["phone_number"]

    # Test with a user without a phone number
    user_payload_no_phone = getUniqueUser(with_phone_number=False)
    registered_user_no_phone, reg_resp_np = registerUser(base_url, user_payload_no_phone)
    assert reg_resp_np.status_code == 200, f"Registration failed for user without phone: {reg_resp_np.text}"
    token_np = loginUser(base_url, registered_user_no_phone)
    headers_np = {"Authorization": f"Bearer {token_np}"}

    resp_np = requests.get(f"{base_url}/profile", headers=headers_np)
    logger.info(f"[test_profile] Profile response (no phone): {resp_np.status_code}, body: {resp_np.json()}")
    assert resp_np.status_code == 200
    data_np = resp_np.json()
    assert data_np["username"] == registered_user_no_phone["username"]
    assert data_np["phone_number"] is None

    # Clean up
    requests.delete(f"{base_url}/user/{registered_user_with_phone['username']}", headers=headers_wp)
    requests.delete(f"{base_url}/user/{registered_user_no_phone['username']}", headers=headers_np)
    logger.info("[test_profile] END")

def testQueryEndpoint(base_url):
    """
    Test the AI query endpoint for various scenarios.
    """
    logger.info("[testQueryEndpoint] START")
    user_payload, reg_resp = registerUser(base_url)
    assert reg_resp.status_code == 200, f"Registration failed: {reg_resp.text}"
    token = loginUser(base_url, user_payload) # Pass the dict part
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    url = f"{base_url}/query"
    
    # Case 1: Missing JSON body
    logger.info("[testQueryEndpoint] Case 1: Missing JSON body")
    resp = requests.post(url, headers=headers)
    logger.info(f"[testQueryEndpoint] Case 1 Response: {resp.status_code}, {resp.text}")
    assert resp.status_code in (400, 422), f"[testQueryEndpoint] Case 1 failed: {resp.text}"

    # Case 2: Missing 'query' field
    logger.info("[testQueryEndpoint] Case 2: Missing 'query' field")
    resp = requests.post(url, headers=headers, json={"chatId": "chat1"})
    logger.info(f"[testQueryEndpoint] Case 2 Response: {resp.status_code}, {resp.text}")
    assert resp.status_code == 400, f"[testQueryEndpoint] Case 2 failed: {resp.text}"
    assert resp.json()["error"] == "No query provided"

    # Case 3: Missing 'chatId' field
    logger.info("[testQueryEndpoint] Case 3: Missing 'chatId' field")
    resp = requests.post(url, headers=headers, json={"query": "what IS GAD?"})
    logger.info(f"[testQueryEndpoint] Case 3 Response: {resp.status_code}, {resp.text}")
    assert resp.status_code == 400, f"[testQueryEndpoint] Case 3 failed: {resp.text}"
    assert resp.json()["error"] == "No chat ID provided"

    # Case 4: Valid request
    logger.info("[testQueryEndpoint] Case 4: Valid request")
    resp = requests.post(url, headers=headers, json={"query": "I have changed my phone number to 807-555-5555", "chatId": "chat1", "pageContent": "Some content"})
    logger.info(f"[testQueryEndpoint] Case 4 Response: {resp.status_code}, {resp.text}")
    assert resp.status_code in (200, 500), f"[testQueryEndpoint] Case 4 failed: {resp.text}"
    if resp.status_code == 200:
        assert "response" in resp.json()
    else:
        assert "error" in resp.json()
    # Clean up: delete user
    requests.delete(f"{base_url}/user/{user_payload['username']}", headers=headers)
    logger.info("[testQueryEndpoint] END")

def test_twilio_incoming_sms_known_number(base_url):
    logger.info("[test_twilio_incoming_sms_known_number] START")
    import random
    phone_number_int = random.randint(1000000000, 9999999999) # Generate a unique phone number
    phone_number_str_twilio = f"+1{phone_number_int}" # Twilio format
    
    user_payload = getUniqueUser()
    user_payload["phone_number"] = phone_number_int
    
    registered_user, reg_resp = registerUser(base_url, user_payload)
    assert reg_resp.status_code == 200, f"Registration with phone number failed: {reg_resp.text}"
    
    twilio_payload = {"From": phone_number_str_twilio, "Body": "Hello from Twilio test"}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    
    logger.info(f"[test_twilio_incoming_sms_known_number] Sending Twilio webhook simulation: {twilio_payload}")
    resp = requests.post(f"{base_url}/api/webhooks/twilio/incoming-message", data=twilio_payload, headers=headers)
    
    logger.info(f"[test_twilio_incoming_sms_known_number] Response: {resp.status_code}, Content-Type: {resp.headers.get('Content-Type')}, Text: {resp.text[:200]}")
    assert resp.status_code == 200
    assert "application/xml" in resp.headers.get("Content-Type", "").lower()
    assert "<Response><Message>" in resp.text
    assert "Sorry, your phone number is not recognized" not in resp.text # Should be an AI response or similar
    
    # Clean up
    cleanup_token = loginUser(base_url, registered_user) 
    auth_headers_cleanup = {"Authorization": f"Bearer {cleanup_token}"}
    requests.delete(f"{base_url}/user/{registered_user['username']}", headers=auth_headers_cleanup)
    logger.info("[test_twilio_incoming_sms_known_number] END")

def test_twilio_incoming_sms_unknown_number(base_url):
    logger.info("[test_twilio_incoming_sms_unknown_number] START")
    unknown_phone_number_twilio = "+19998887777"
    
    twilio_payload = {"From": unknown_phone_number_twilio, "Body": "Who dis?"}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    logger.info(f"[test_twilio_incoming_sms_unknown_number] Sending Twilio webhook simulation for unknown number: {twilio_payload}")
    resp = requests.post(f"{base_url}/api/webhooks/twilio/incoming-message", data=twilio_payload, headers=headers)
    
    logger.info(f"[test_twilio_incoming_sms_unknown_number] Response: {resp.status_code}, Content-Type: {resp.headers.get('Content-Type')}, Text: {resp.text[:200]}")
    assert resp.status_code == 200
    assert "application/xml" in resp.headers.get("Content-Type", "").lower()
    assert "<Response><Message>Sorry, your phone number is not recognized in our system.</Message></Response>" in resp.text
    logger.info("[test_twilio_incoming_sms_unknown_number] END")

def test_register_duplicate_phone_number(base_url):
    logger.info("[test_register_duplicate_phone_number] START")
    
    # Step 1: Create and register the first user with a unique phone number.
    # This user will establish the phone number to be tested for duplication.
    user1_payload = getUniqueUser(with_phone_number=True)
    phone_to_test_duplication = user1_payload["phone_number"]
    
    # This first registration MUST succeed.
    # registerUser returns (payload_passed_in, response_object)
    _, reg_resp1 = registerUser(base_url, user1_payload)
    assert reg_resp1.status_code == 200, f"Registration of user1 (with unique phone {phone_to_test_duplication}) failed: {reg_resp1.text}"
    logger.info(f"[test_register_duplicate_phone_number] User1 {user1_payload['username']} registered successfully with phone {phone_to_test_duplication}")

    try:
        # Step 2: Create a second user payload with a different username but the SAME phone number.
        user2_payload = getUniqueUser() # Ensures different username from user1
        while user2_payload['username'] == user1_payload['username']:
            user2_payload = getUniqueUser() # Make sure username is truly different
        user2_payload["phone_number"] = phone_to_test_duplication # Use phone from user1

        # Step 3: Attempt to register the second user. This registration MUST FAIL.
        logger.info(f"[test_register_duplicate_phone_number] Attempting to register user2 {user2_payload['username']} with duplicate phone {phone_to_test_duplication}")
        _, reg_resp2 = registerUser(base_url, user2_payload)
        
        logger.info(f"[test_register_duplicate_phone_number] Response for user2 (attempting duplicate phone): {reg_resp2.status_code}, {reg_resp2.text}")
        
        assert reg_resp2.status_code == 400, f"Expected 400 error for duplicate phone, got {reg_resp2.status_code}"
        response_json = reg_resp2.json()
        assert "detail" in response_json, "Error response JSON should contain 'detail' key"
        assert response_json["detail"].lower() == "phone number already registered", \
               f"Unexpected error detail: {response_json['detail']}"
        logger.info(f"[test_register_duplicate_phone_number] Correctly received 400 for user2 with duplicate phone.")

    finally:
        # Step 4: Clean up - delete the first user (user1).
        # user1_payload contains the username and password for login.
        logger.info(f"[test_register_duplicate_phone_number] Cleaning up user1: {user1_payload['username']}")
        token1 = loginUser(base_url, user1_payload) # user1_payload is the dict with 'username' and 'password'
        if token1:
            headers1 = {"Authorization": f"Bearer {token1}"}
            delete_resp = requests.delete(f"{base_url}/user/{user1_payload['username']}", headers=headers1)
            logger.info(f"[test_register_duplicate_phone_number] Cleanup delete response for {user1_payload['username']}: {delete_resp.status_code}")
            if delete_resp.status_code != 200:
                 logger.warning(f"[test_register_duplicate_phone_number] Cleanup of user {user1_payload['username']} failed with status {delete_resp.status_code}: {delete_resp.text}")       
        else:
            logger.warning(f"[test_register_duplicate_phone_number] Could not obtain token for cleanup of user: {user1_payload['username']}")

    logger.info("[test_register_duplicate_phone_number] END")


def test_delete_user(base_url):
    """
    Test the delete user endpoint for success, forbidden, and not found cases.
    """
    logger.info("[test_delete_user] START")
    # Use a unique user for this test
    user1_payload, reg_resp1 = registerUser(base_url, getUniqueUser())
    assert reg_resp1.status_code == 200, f"user1 registration failed: {reg_resp1.text}"
    token1 = loginUser(base_url, user1_payload)
    headers1 = {"Authorization": f"Bearer {token1}"}

    # Register user2 (will attempt to delete user1)
    user2_payload, reg_resp2 = registerUser(base_url, getUniqueUser())
    assert reg_resp2.status_code == 200, f"user2 registration failed: {reg_resp2.text}"
    token2 = loginUser(base_url, user2_payload)
    headers2 = {"Authorization": f"Bearer {token2}"}
    username = user1_payload["username"]
    url = f"{base_url}/user/{username}"

    # Success: user deletes themselves
    logger.info(f"[test_delete_user] Deleting user {username} (self-delete)")
    resp = requests.delete(url, headers=headers1)
    logger.info(f"[test_delete_user] Delete response: {resp.status_code}, {resp.text}")
    assert resp.status_code == 200, f"[test_delete_user] Delete failed: {resp.text}"
    assert resp.json()["message"].startswith(f"User '{username}' deleted successfully.")

    # Not found: try deleting again
    logger.info(f"[test_delete_user] Try deleting user {username} again (should be not found/unauthorized)")
    resp = requests.delete(url, headers=headers1)
    logger.info(f"[test_delete_user] Second delete response: {resp.status_code}, {resp.text}")
    assert resp.status_code == 401, f"[test_delete_user] Second delete failed: {resp.text}"

    # Forbidden: another user tries to delete test user
    logger.info(f"[test_delete_user] Second user attempts to delete {username}")
    resp = requests.delete(url, headers=headers2)
    logger.info(f"[test_delete_user] Forbidden delete response: {resp.status_code}, {resp.text}")
    assert resp.status_code == 403, f"[test_delete_user] Forbidden delete failed: {resp.text}"
    assert resp.json()["detail"] == "You can only delete your own account."
    logger.info("[test_delete_user] END")

def test_list_files(base_url):
    logger.info("[test_list_files] START")
    user_payload, reg_resp = registerUser(base_url)
    assert reg_resp.status_code == 200, f"Registration failed: {reg_resp.text}"
    token = loginUser(base_url, user_payload) # Pass the dict part
    headers = {'Authorization': f'Bearer {token}'}
    
    # Upload a file first to ensure there's something to list
    files = {"file": (TEST_FILE_NAME, TEST_FILE_CONTENT, "text/plain")}
    upload_resp = requests.post(f"{base_url}/upload", headers=headers, files=files)
    assert upload_resp.status_code == 200, f"[test_list_files] Upload failed: {upload_resp.text}"
    
    logger.info("[test_list_files] Listing files")
    resp = requests.get(f"{base_url}/files", headers=headers)
    logger.info(f"[test_list_files] Response status: {resp.status_code}, body: {resp.json()}")
    assert resp.status_code == 200, f"[test_list_files] Failed: {resp.text}"
    # New response format is a dict with 'files' array
    response_data = resp.json()
    assert isinstance(response_data, dict)
    assert "files" in response_data
    assert isinstance(response_data["files"], list)
    # Clean up: delete user
    requests.delete(f"{base_url}/user/{user_payload['username']}", headers=headers)
    logger.info("[test_list_files] END")

def test_download_file(base_url):
    logger.info("[test_download_file] START")
    user_payload, reg_resp = registerUser(base_url)
    assert reg_resp.status_code == 200, f"Registration failed: {reg_resp.text}"
    token = loginUser(base_url, user_payload) # Pass the dict part
    headers = {'Authorization': f'Bearer {token}'}
    
    # Upload a file to download
    files = {"file": (TEST_FILE_NAME, TEST_FILE_CONTENT, "text/plain")}
    upload_resp = requests.post(f"{base_url}/upload", headers=headers, files=files)
    assert upload_resp.status_code == 200, f"[test_download_file] Upload failed: {upload_resp.text}"
    
    # Get the fileId from the upload response
    upload_data = upload_resp.json()
    logger.info(f"[test_download_file] Upload response: {upload_data}")
    assert "fileId" in upload_data, "Upload response doesn't include fileId"
    file_id = upload_data["fileId"]
    
    # Download the file using fileId
    logger.info(f"[test_download_file] Downloading file with ID: {file_id}")
    resp = requests.get(f"{base_url}/download/{file_id}", headers=headers)
    logger.info(f"[test_download_file] Response status: {resp.status_code}, length: {len(resp.content)}")
    assert resp.status_code == 200, f"[test_download_file] Download failed: {resp.text}"
    assert resp.content == TEST_FILE_CONTENT
    
    # Clean up by deleting the user which should also handle file cleanup if implemented
    requests.delete(f"{base_url}/user/{user_payload['username']}", headers=headers)
    logger.info("[test_download_file] END")

def test_delete_file(base_url):
    logger.info("[test_delete_file] START")
    user_payload, reg_resp = registerUser(base_url)
    assert reg_resp.status_code == 200, f"Registration failed: {reg_resp.text}"
    token = loginUser(base_url, user_payload) # Pass the dict part
    headers = {'Authorization': f'Bearer {token}'}

    # Upload a file to delete
    files = {"file": (TEST_FILE_NAME, TEST_FILE_CONTENT, "text/plain")}
    upload_resp = requests.post(f"{base_url}/upload", headers=headers, files=files)
    assert upload_resp.status_code == 200, f"[test_delete_file] Upload failed: {upload_resp.text}"
    
    # Get the fileId from the upload response
    upload_data = upload_resp.json()
    logger.info(f"[test_delete_file] Upload response: {upload_data}")
    assert "fileId" in upload_data, "Upload response doesn't include fileId"
    file_id = upload_data["fileId"]
    
    # Delete the file using fileId
    logger.info(f"[test_delete_file] Deleting file with ID: {file_id}")
    resp = requests.delete(f"{base_url}/delete/{file_id}", headers=headers)
    logger.info(f"[test_delete_file] Delete response: {resp.status_code}, {resp.json()}")
    assert resp.status_code == 200, f"[test_delete_file] Delete failed: {resp.text}"
    resp_json = resp.json()
    
    # Accept message or detail containing 'deleted'
    msg = resp_json.get("message") or resp_json.get("detail")
    assert msg is not None and "deleted" in msg
    
    # Clean up: delete user
    requests.delete(f"{base_url}/user/{user_payload['username']}", headers=headers)
    
    # Confirm file is gone by checking the files list
    resp = requests.get(f"{base_url}/files", headers=headers)
    logger.info(f"[test_delete_file] File list after delete: {resp.json()}")
    files_data = resp.json()
    
    # Check that the file with our fileId is not in the list
    if "files" in files_data and isinstance(files_data["files"], list):
        file_ids = [file.get("fileId") for file in files_data["files"]]
        assert file_id not in file_ids, f"File with ID {file_id} was not deleted"
    
    logger.info("[test_delete_file] END")
