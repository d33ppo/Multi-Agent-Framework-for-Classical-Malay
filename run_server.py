"""
Start the Classical Malay Pipeline API server.

    python run_server.py

Then open http://localhost:8000 in your browser.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
