import urllib.request
import json
import time

def test_retrieval():
    url = "http://127.0.0.1:8000/query"
    payload = {
        "query": "Take a look into this codebase and tell what is this project about?",
        "final_top_k": 8,
        "include_llm": True,       # Switch to True to see Kimi's thinking
        "stream": True             # Enabling stream for SSE support
    }

    print(f"📡 Sending retrieval request to: {url}...")
    headers = {"Content-Type": "application/json"}
    
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
        
        with urllib.request.urlopen(req) as response:
            if payload.get("stream"):
                print("\n✨ Receiving streaming response (NVIDIA Kimi Thinking...)\n")
                print("-" * 50)
                
                # Handling Server-Sent Events (SSE)
                for line in response:
                    line_decoded = line.decode('utf-8').strip()
                    if not line_decoded or line_decoded == "data: [DONE]":
                        continue
                        
                    if line_decoded.startswith("data: "):
                        try:
                            # Parse the JSON portion of the SSE line
                            event_data = json.loads(line_decoded[6:])
                            
                            # Support for OpenAI-style streaming format
                            if "choices" in event_data:
                                content = event_data["choices"][0]["delta"].get("content", "")
                                if content:
                                    print(content, end="", flush=True)
                            
                            # Support for our summary/error messages
                            elif "error" in event_data:
                                print(f"\n❌ Error: {event_data['error']}")
                                
                        except json.JSONDecodeError:
                            # It might be a [START] or special message
                            pass
                print("\n" + "-" * 50)
            else:
                raw_data = response.read().decode('utf-8')
                data = json.loads(raw_data)
                print("\n✅ Response Received (Non-streaming):")
                print(json.dumps(data, indent=2))

    except Exception as e:
        print(f"\n❌ Request failed: {e}")

if __name__ == "__main__":
    test_retrieval()