import os
import logging
import requests
import pytest
import sys

# import warnings
# from urllib3.exceptions import InsecureRequestWarning

# warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

RAILWAY_URL = "https://web-production-d7d37.up.railway.app"
LOCAL_URL = "http://localhost:7050"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test1.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
# Get the root logger and set its level
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Add handlers to the root logger
root_logger.handlers = [
    logging.FileHandler('logs/test1.log'),
    logging.StreamHandler(sys.stdout)
]

logger = logging.getLogger(__name__)

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
    print("response:", root_resp)
    logger.info(f"[test_server_health] Root endpoint response: {root_resp.status_code}, {root_resp.text[:100]}")
    
    # If we get here without exceptions, the server is responding
    logger.info("[test_server_health] Server is responding to requests")







# _____________________________________________
#  helper functions 
# _____________________________________________




def registerUser(base_url, user_payload=None):
    logger.info(f"[registerUser] Registering user with payload: {user_payload}")
    if user_payload is None:
        # user_payload = getUniqueUser()
        user_payload = {"username": "testuser1", "password": "testpass123", "phone_number": 18073587137}
    
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












# _____________________________________________
# Test the AI query endpoint
# _____________________________________________






def testQueryEndpoint(base_url):
    """
    Test the AI query endpoint for various scenarios.
    """
    logger.info("[testQueryEndpoint] START")
    # user_payload, reg_resp = registerUser(base_url)
    user_payload = {"username": "testuser1", "password": "testpass123", "phone_number": 18073587137}
    # assert reg_resp.status_code == 200, f"Registration failed: {reg_resp.text}"
    token = loginUser(base_url, user_payload) # Pass the dict part
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    url = f"{base_url}/query"
    

    # Case 4: Valid request
    logger.info("[testQueryEndpoint] Case 4: Valid request")
    resp = requests.post(url, headers=headers, json={"query": "where do I live?", "chatId": "chat1"})
    # resp = requests.post(url, headers=headers, json={"query": "what was our last conversation about?", "chatId": "chat1", "pageContent": "Some content"})
    logger.info(f"[testQueryEndpoint] Case 4 Response: {resp.status_code}, {resp.text}")
    assert resp.status_code in (200, 500), f"[testQueryEndpoint] Case 4 failed: {resp.text}"
    
    if resp.status_code == 200:
        assert "response" in resp.json()
    else:
        assert "error" in resp.json()

