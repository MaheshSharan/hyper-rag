import urllib.request
import json

def test_retrieval():
    url = "http://127.0.0.1:8000/query"
    payload = {
        "query": "What is this project about? Summarize the main purpose.",
        "final_top_k": 8,
        "include_llm": False,      # Set True only if you have valid OpenAI key
        "stream": False
    }
    
    print("📡 Sending retrieval request...\n")
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    req.add_header('Content-Type', 'application/json')
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            
            print("✅ Retrieval Successful!\n")
            print(f"Found {len(result['ranked_results'])} relevant chunks\n")
            
            for i, item in enumerate(result['ranked_results'][:5], 1):
                score = item.get('final_score', 0)
                source = item.get('source', 'unknown')
                text_preview = item['text'][:180].replace('\n', ' ')
                print(f"{i}. [{score:.3f}] {source}")
                print(f"   {text_preview}...\n")
                
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_retrieval()