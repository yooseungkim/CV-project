import urllib.request
import json
import os

port = None
api_path = os.path.expanduser("~/.securecoder/api.json")
if os.path.exists(api_path):
    try:
        with open(api_path) as f:
            data = json.load(f)
            port = data.get("port")
    except Exception:
        pass

if not port:
    port = os.environ.get("SECURECODER_API_PORT")

if port:
    url = f"http://127.0.0.1:{port}/dependency/scan"
    req_data = json.dumps({
        "registry": "pypi",
        "packages": [{"package": "torchxrayvision"}]
    }).encode("utf-8")
    
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as res:
            print(res.read().decode("utf-8"))
    except Exception as e:
        print(f"Error: {e}")
else:
    print("SecureCoder API port not found.")
