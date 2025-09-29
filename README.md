# AI-Agent Backend API


## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/backend.git
   cd backend
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Set up virtual environment and install dependencies**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   # source venv/bin/activate
   
   pip install -r requirements.txt
   ```

4. **Run the server locally**
   ```bash
   python -m server.main
   ```

5. **Run tests**
   ```bash
   TEST_LOCAL=1 pytest -s test1.py
   # For all tests:
   # pytest
   ```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:

- **Interactive API Docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative API Docs (ReDoc)**: http://localhost:8000/redoc

## Environment Variables

See [.env.example](.env.example) for all available configuration options.

## Project Structure

```
backend/
├── server/               # Main application package
│   ├── __init__.py
│   ├── main.py           # FastAPI app initialization
│   ├── routes.py         # API endpoints
│   ├── auth.py           # Authentication utilities
│   ├── db.py             # Database models and session
│   ├── schemas.py        # Pydantic models
│   └── utils.py          # Helper functions
├── aiagent/             # AI-related functionality
│   ├── handler/         # Request handlers
│   ├── memory/          # Memory management
│   └── tools/           # AI tools and utilities
├── tests/               # Test files
├── .env.example         # Example environment variables
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Development

### Running Tests

```bash
# Run a specific test file
TEST_LOCAL=1 pytest -s test1.py

# Run all tests
pytest

# Run with coverage report
pytest --cov=server --cov-report=html
```
