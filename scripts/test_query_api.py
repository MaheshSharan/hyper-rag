import urllib.request
import urllib.parse
import json
import sys
import os

def test_query():
    url = "http://127.0.0.1:8000/query"
    payload = {
        "query": "What is in sample.py?",
        "final_top_k": 10,
        "include_llm": False,
        "stream": False
    }
    
    print(f"📡 Sending request to {url}...")
    try:
        data = json.dumps(payload).encode('utf-8')
        # Fix: Only use positional or supported keyword arguments
        req = urllib.request.Request(url, data=data)
        req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode('utf-8')
            res_data = json.loads(res_body)
            
            print("\n✅ Response Received!")
            print(f"🔍 Found {len(res_data['ranked_results'])} unique chunks.")
            
            for i, res in enumerate(res_data['ranked_results'][:3], 1):
                score = res.get('final_score') or res.get('graph_score', 0)
                print(f"\n[{i}] {res['source']} (Score: {score:.4f})")
                text_preview = res['text'][:200].replace('\n', ' ')
                print(f"   {text_preview}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Ensure src is in python path
    test_query()
