# RASA Chatbot v3 - Quick Start Guide
 
## 🚀 Initial Setup (One-Time)
 
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
 
### 2. Train RASA Model
```bash
python -m rasa train
```
 
---
 
## ⚡ Running the Application
 
**IMPORTANT**: Set the environment variable in **EVERY terminal** before running commands.
 
### Windows Command Prompt
```cmd
set VECTOR_API_KEY=dev-local-key
```
 
### Windows PowerShell
```powershell
$env:VECTOR_API_KEY="dev-local-key"
```
 
### macOS/Linux
```bash
export VECTOR_API_KEY="dev-local-key"
```
 
---
 
## 🖥️ Required Terminals (Run in Order)
 
### **Terminal 1: RASA Actions Server**
```cmd
set VECTOR_API_KEY=dev-local-key
rasa run actions --port 5055
```
 
### **Terminal 2: RASA Server**
```cmd
set VECTOR_API_KEY=dev-local-key
rasa run --enable-api --cors "*" --port 5005
```
 
### **Terminal 3: Leave Calculator MCP Server**
```cmd
uvicorn mcp_utilities.leave_calculator:app --host 0.0.0.0 --port 8000 --reload
```
 
### **Terminal 4: Secure RAG Service**
```cmd
uvicorn secure_rag.main:app --host 0.0.0.0 --port 8001 --reload
```
*Note: Use `srvenv` or `secureragvenv` virtual environment if configured*
 
### **Terminal 5: Flask Web Application**
```cmd
python app.py
```
*Flask runs on port **5001** to avoid conflicts*
 
---
 
## 📡 Service Ports Summary
 
| Service | URL | Port |
|---------|-----|------|
| **Flask Web App** | http://localhost:5001 | 5001 |
| **RASA Server** | http://localhost:5005 | 5005 |
| **RASA Actions** | http://localhost:5055 | 5055 |
| **Leave Calculator MCP** | http://localhost:8000 | 8000 |
| **Secure RAG** | http://localhost:8001 | 8001 |
 
---
 
## 🔄 When to Retrain Model
 
Retrain the RASA model when:
- ✅ MCP/Leave Calculator updates
- ✅ New intents or actions added
- ✅ Domain or stories modified
 
```bash
python -m rasa train
```
 
---
 
## 📝 Notes
 
- **Environment Variable**: Must be set in each terminal session before running services
- **Startup Order**: Follow the terminal order above for smooth initialization
- **Port 5001**: Flask uses 5001 to avoid conflict with macOS AirPlay Receiver (port 5000)
- **Virtual Environments**: Activate appropriate venv for each service if configured