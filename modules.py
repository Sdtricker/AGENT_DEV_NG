import requests
import json
from datetime import datetime, timedelta

DEEPINFRA_MODELS = [
    "deepseek-ai/DeepSeek-Prover-V2-671B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-V3.1",
    "deepseek-ai/DeepSeek-V3-0324-Turbo",
    "allenai/olmOCR-7B-0725-FP8",
    "google/gemma-2-27b-it",
    "allenai/olmOCR-7B-0825",
    "Qwen/QwQ-32B",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "NovaSky-AI/Sky-T1-32B-Preview",
    "microsoft/WizardLM-2-7B",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "google/gemma-2-9b-it",
    "google/gemma-3-12b-it",
    "deepseek-ai/DeepSeek-V3-0324",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "microsoft/phi-4",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "Qwen/Qwen3-32B",
    "moonshotai/Kimi-K2-Instruct-0905",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "mistralai/Devstral-Small-2505",
    "microsoft/Phi-4-multimodal-instruct",
    "NousResearch/Hermes-3-Llama-3.1-70B",
    "Qwen/QVQ-72B-Preview",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "Qwen/Qwen3-30B-A3B",
    "lizpreciatior/lzlv_70b_fp16_hf",
    "deepseek-ai/DeepSeek-V3.2-Exp",
    "meta-llama/Llama-Guard-4-12B",
    "openai/gpt-oss-120b",
    "google/gemmai-3-4b-it",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "mistralai/Devstral-Small-2507",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "google/gemma-3-27b-it",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-Turbo",
    "zai-org/GLM-4.5",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "microsoft/phi-4-reasoning-plus",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1-0528",
    "google/gemma-1.1-7b-it",
    "deepseek-ai/DeepSeek-V3.1-Terminus"
]

VENICE_MODELS = [
    "venice/worm-gpt",
    "venice/dolphin-3.0-mistral-24b"
]

OPENROUTER_MODELS_CACHE = []
OPENROUTER_CACHE_TIME = None
CACHE_DURATION = timedelta(hours=1)

def fetch_openrouter_models():
    global OPENROUTER_MODELS_CACHE, OPENROUTER_CACHE_TIME
    
    if OPENROUTER_MODELS_CACHE and OPENROUTER_CACHE_TIME:
        if datetime.now() - OPENROUTER_CACHE_TIME < CACHE_DURATION:
            return OPENROUTER_MODELS_CACHE
    
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            model_ids = [f"openrouter/{m.get('id')}" for m in models if m.get("id")]
            OPENROUTER_MODELS_CACHE = model_ids
            OPENROUTER_CACHE_TIME = datetime.now()
            return model_ids
    except Exception as e:
        print(f"Error fetching OpenRouter models: {e}")
    
    return OPENROUTER_MODELS_CACHE

def get_all_models():
    venice_list = [{"id": m, "name": m.replace("venice/", "")} for m in VENICE_MODELS]
    deepinfra_list = [{"id": m, "name": m} for m in DEEPINFRA_MODELS]
    
    openrouter_models = fetch_openrouter_models()
    openrouter_list = [{"id": m, "name": m.replace("openrouter/", "")} for m in openrouter_models]
    
    all_models = venice_list + deepinfra_list + openrouter_list
    return all_models

def get_model_info(model_id):
    if model_id in DEEPINFRA_MODELS:
        return {"provider": "deepinfra", "name": model_id}
    elif model_id in VENICE_MODELS:
        return {"provider": "venice", "name": model_id.replace("venice/", "")}
    elif model_id.startswith("openrouter/"):
        return {"provider": "openrouter", "name": model_id.replace("openrouter/", "")}
    return None

def is_model_available(model_id):
    if model_id in DEEPINFRA_MODELS or model_id in VENICE_MODELS:
        return True
    if model_id.startswith("openrouter/"):
        openrouter_models = fetch_openrouter_models()
        return model_id in openrouter_models
    return False
