from flask import Flask, request, jsonify, send_from_directory, session, Response, stream_with_context
import requests
import os
import json
import time
from datetime import datetime
from modules import get_all_models, is_model_available, get_model_info

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'ng-ai-agent-secret-key-2025')

DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
VENICE_API_URL = "https://outerface.venice.ai/api/inference/chat"
CHATFREE_API_URL = "https://chatfreeai.com/api/chat"

HISTORY_FILE = "/tmp/history.txt"
DEEPINFRA_API_KEY = os.environ.get('DEEPINFRA_API_KEY', '')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')

DEEPINFRA_HEADERS = {
    "Content-Type": "application/json",
    "X-Deepinfra-Source": "python-client",
    "Accept": "application/json"
}

if DEEPINFRA_API_KEY:
    DEEPINFRA_HEADERS["Authorization"] = f"Bearer {DEEPINFRA_API_KEY}"

OPENROUTER_HEADERS = {
    "Content-Type": "application/json",
    "HTTP-Referer": "https://ng-ai-agent.repl.co",
    "X-Title": "NG AI Agent"
}

if OPENROUTER_API_KEY:
    OPENROUTER_HEADERS["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"

VENICE_HEADERS = {
    'authority': 'outerface.venice.ai',
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.9',
    'content-type': 'application/json',
    'origin': 'https://venice.ai',
    'referer': 'https://venice.ai/',
    'sec-ch-ua': '"Chromium";v="137", "Not/A)Brand";v="24"',
    'sec-ch-ua-mobile': '?1',
    'sec-ch-ua-platform': '"Android"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'user-agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Mobile Safari/537.36',
    'x-venice-version': 'interface@20250626.212124+945291c'
}

CHATFREE_HEADERS = {
    "authority": "chatfreeai.com",
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "content-type": "application/json",
    "origin": "https://chatfreeai.com",
    "referer": "https://chatfreeai.com/",
    "user-agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Mobile Safari/537.36"
}

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        except:
            pass
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def call_deepinfra(model, message, conversation_history=None):
    messages = []
    
    if conversation_history and isinstance(conversation_history, list):
        messages = conversation_history
    else:
        messages = [{"role": "user", "content": message}]
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048
    }
    
    try:
        response = requests.post(DEEPINFRA_API_URL, headers=DEEPINFRA_HEADERS, json=payload, timeout=90)
        
        if response.status_code == 200:
            try:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]
                    if content:
                        return content
            except json.JSONDecodeError as e:
                print(f"DeepInfra JSON decode error: {e}")
                print(f"Response text: {response.text[:500]}")
        
        error_msg = f"DeepInfra API returned status {response.status_code}"
        try:
            error_data = response.json()
            if "error" in error_data:
                if isinstance(error_data["error"], dict):
                    error_msg = error_data["error"].get("message", str(error_data["error"]))
                else:
                    error_msg = str(error_data["error"])
        except:
            pass
        
        print(f"DeepInfra API Error: {error_msg}")
        print(f"Response: {response.text[:500]}")
        raise Exception(error_msg)
        
    except requests.exceptions.Timeout:
        raise Exception("Request timeout - the AI model took too long to respond")
    except requests.exceptions.ConnectionError:
        raise Exception("Connection error - cannot reach DeepInfra server")
    except Exception as e:
        if "API" in str(e) or "error" in str(e) or "status" in str(e).lower():
            raise
        raise Exception(f"Error: {str(e)}")

def call_venice(model_name, message):
    payload = {
        'requestId': f'req{int(time.time())}',
        'conversationType': 'text',
        'type': 'text',
        'modelId': 'dolphin-3.0-mistral-24b',
        'modelName': 'Venice Uncensored',
        'modelType': 'text',
        'prompt': [{'role': 'user', 'content': message}],
        'systemPrompt': '',
        'messageId': f'msg{int(time.time())}',
        'includeVeniceSystemPrompt': True,
        'isCharacter': False,
        'userId': f'user_anon_{int(time.time())}',
        'simpleMode': False,
        'characterId': '',
        'id': '',
        'textToSpeech': {
            'voiceId': 'af_sky',
            'speed': 1,
        },
        'webEnabled': True,
        'reasoning': True,
        'temperature': 0.3,
        'topP': 1,
        'clientProcessingTime': 11
    }
    response = requests.post(VENICE_API_URL, headers=VENICE_HEADERS, json=payload, timeout=60)
    if response.status_code == 200:
        full_text = ''
        for line in response.text.strip().splitlines():
            try:
                line_data = json.loads(line.strip())
                if isinstance(line_data, dict):
                    full_text += line_data.get("content", "")
            except json.JSONDecodeError:
                pass
        if full_text:
            return full_text
    raise Exception(f"API error: {response.status_code}")

def looks_cutoff(text):
    if not text:
        return True
    s = text.strip().lower()
    if "</html>" in s or "</body>" in s:
        return False
    if len(s.split()) < 30:
        return False
    if s.count("```") % 2 == 1:
        return True
    if not s.endswith((".", "?", "!", "}", ">", ")", "]", ";", ":", '"', "'")):
        return True
    return False

def extract_chatfree_reply(response):
    try:
        data = response.json()
    except:
        return response.text.strip()
    
    if isinstance(data, dict):
        for k in ("response", "message", "reply", "output"):
            if k in data:
                v = data[k]
                if isinstance(v, str):
                    return v.strip()
                elif isinstance(v, list) and v:
                    if isinstance(v[0], dict):
                        return v[0].get("text", str(v[0])).strip()
                    return str(v[0]).strip()
                elif isinstance(v, dict):
                    return (v.get("text") or str(v)).strip()
        if "choices" in data:
            ch = data["choices"][0]
            if "text" in ch:
                return ch["text"].strip()
            if "message" in ch and isinstance(ch["message"], dict):
                return ch["message"].get("content", "").strip()
    return str(data).strip()

def call_openrouter(model, message):
    clean_model = model.replace("openrouter/", "")
    
    if OPENROUTER_API_KEY:
        payload = {
            "model": clean_model,
            "messages": [{"role": "user", "content": message}]
        }
        response = requests.post(OPENROUTER_API_URL, headers=OPENROUTER_HEADERS, json=payload, timeout=60)
        if response.status_code == 200:
            try:
                data = response.json()
                if "choices" in data and data["choices"]:
                    return data["choices"][0]["message"]["content"]
            except:
                pass
        raise Exception(f"API error: {response.status_code}")
    else:
        history = [{"role": "user", "parts": [{"text": message}]}]
        payload = {
            "model": clean_model,
            "message": message,
            "history": history
        }
        response = requests.post(CHATFREE_API_URL, headers=CHATFREE_HEADERS, json=payload, timeout=60)
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")
        
        reply = extract_chatfree_reply(response)
        full_reply = reply
        history.append({"role": "assistant", "parts": [{"text": reply}]})
        
        tries = 0
        max_tries = 8
        
        while looks_cutoff(full_reply) and tries < max_tries and "</html>" not in full_reply.lower():
            tries += 1
            print(f"🔁 Auto-continuing chunk {tries}/{max_tries}...")
            
            continue_prompt = "Continue exactly from where you stopped. Do not repeat previous text. Just continue."
            history.append({"role": "user", "parts": [{"text": continue_prompt}]})
            
            time.sleep(1.0)
            
            cont_payload = {
                "model": clean_model,
                "message": continue_prompt,
                "history": history
            }
            
            try:
                cont_response = requests.post(CHATFREE_API_URL, headers=CHATFREE_HEADERS, json=cont_payload, timeout=60)
                if cont_response.status_code == 200:
                    cont = extract_chatfree_reply(cont_response)
                    if not cont or len(cont) < 20:
                        break
                    full_reply += "\n" + cont
                    history.append({"role": "assistant", "parts": [{"text": cont}]})
                else:
                    break
            except:
                break
        
        return full_reply

@app.route('/')
def index():
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/chat')
def chat_page():
    if 'user_id' not in session:
        return '<script>window.location.href="/"</script>'
    with open('ui.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/background')
def background_page():
    with open('background.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    
    if username and password:
        session['user_id'] = username
        session['logged_in'] = True
        return jsonify({"success": True, "message": "Login successful"})
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"success": True})

@app.route('/api/models', methods=['GET'])
def get_models():
    models = get_all_models()
    return jsonify({
        "success": True,
        "models": models,
        "total": len(models)
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    data = request.json
    message = data.get('message', '').strip()
    model_id = data.get('model', '')
    conversation_history = data.get('conversation_history', None)
    
    if not message:
        return jsonify({"success": False, "error": "Message is required"}), 400
    
    if not model_id:
        return jsonify({"success": False, "error": "Model is required"}), 400
    
    model_info = get_model_info(model_id)
    if not model_info:
        return jsonify({"success": False, "error": "Invalid model"}), 400
    
    provider = model_info['provider']
    
    try:
        print(f"🚀 Processing request for model: {model_id}")
        
        if provider == "deepinfra":
            reply = call_deepinfra(model_id, message, conversation_history)
        elif provider == "venice":
            reply = call_venice(model_id, message)
        elif provider == "openrouter":
            reply = call_openrouter(model_id, message)
        else:
            return jsonify({"success": False, "error": "Unknown provider"}), 400
        
        print(f"✅ Response generated successfully ({len(reply)} chars)")
        
        history = load_history()
        history.append({
            "user_id": session.get('user_id'),
            "timestamp": datetime.now().isoformat(),
            "model": model_id,
            "message": message,
            "response": reply
        })
        save_history(history)
        
        return jsonify({
            "success": True,
            "model": model_id,
            "message": message,
            "response": reply,
            "timestamp": datetime.now().isoformat()
        })
            
    except requests.exceptions.Timeout:
        return jsonify({
            "success": False,
            "error": "Request timeout. The AI model took too long to respond."
        }), 504
    except requests.exceptions.RequestException as e:
        return jsonify({
            "success": False,
            "error": f"Network error: {str(e)}"
        }), 503
    except Exception as e:
        print(f"❌ Error in chat: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Error: {str(e)}"
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    user_id = session.get('user_id')
    all_history = load_history()
    user_history = [h for h in all_history if h.get('user_id') == user_id]
    
    sanitized_history = []
    for item in user_history:
        sanitized_item = {k: v for k, v in item.items() if k != 'provider'}
        sanitized_history.append(sanitized_item)
    
    return jsonify({
        "success": True,
        "history": sanitized_history
    })

@app.route('/api/history/delete/<int:index>', methods=['DELETE'])
def delete_history_item(index):
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    user_id = session.get('user_id')
    all_history = load_history()
    user_history = [h for h in all_history if h.get('user_id') == user_id]
    
    if 0 <= index < len(user_history):
        item_to_delete = user_history[index]
        all_history.remove(item_to_delete)
        save_history(all_history)
        return jsonify({"success": True})
    
    return jsonify({"success": False, "error": "Invalid index"}), 400

@app.route('/api/history/clear', methods=['DELETE'])
def clear_history():
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    user_id = session.get('user_id')
    all_history = load_history()
    filtered_history = [h for h in all_history if h.get('user_id') != user_id]
    save_history(filtered_history)
    
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
