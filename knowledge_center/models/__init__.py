import json
import os


def USE_CLOUD_MODELS() -> bool:
    if not os.path.exists("ollama_option.json"):
        with open("ollama_option.json", "w") as f:
            json.dump({"use_ollama": True}, f)
    with open("ollama_option.json", "r") as f:
        ollama_option = json.load(f)
    return ollama_option.get("use_ollama", True)
