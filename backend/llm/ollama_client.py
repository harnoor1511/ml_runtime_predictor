import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"


def call_ollama(prompt, model=DEFAULT_MODEL, timeout=600, num_predict=None):
    payload = {"model": model, "prompt": prompt, "stream": False}
    if num_predict:
        payload["options"] = {"num_predict": num_predict}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        return {"ok": True, "text": r.json().get("response", "")}
    except requests.exceptions.ReadTimeout:
        return {
            "ok": False,
            "text": "",
            "error": f"Local model timed out after {timeout}s. This can happen on CPU with longer "
                     f"generations (e.g. full HTML reproduction). Try again, use a smaller/faster "
                     f"model, or reduce the number of source files being fed in.",
        }
    except requests.RequestException as e:
        return {
            "ok": False,
            "text": "",
            "error": f"Could not reach local Ollama server at {OLLAMA_URL}. "
                     f"Make sure `ollama serve` is running and you've pulled the model "
                     f"(`ollama pull {model}`). Details: {e}",
        }
