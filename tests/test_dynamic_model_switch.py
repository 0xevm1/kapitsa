#!/usr/bin/env python3
"""Tests for dynamic client-side model selection and on-demand model switching."""
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

PORT = 8932
BASE = f"http://127.0.0.1:{PORT}"
BINARY = os.path.abspath("build/kapitsa")

def req(path, data=None, method=None):
    url = f"{BASE}{path}"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data is not None else None
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:
            content = resp.read().decode("utf-8")
            return resp.status, content
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")

def check(name, condition, extra="", proc=None):
    if condition:
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name} {extra}")
        if proc:
            proc.terminate()
            out, err = proc.communicate(timeout=5)
            print("SERVER STDOUT:\n", out)
            print("SERVER STDERR:\n", err)
        sys.exit(1)

def wait_for_server(proc, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited prematurely with code {proc.returncode}")
        try:
            st, res = req("/health")
            if st == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError("Server did not become healthy in time")

def main():
    print(f"Starting Kapitsa server on port {PORT} with initial model 4b...")
    proc = subprocess.Popen(
        [BINARY, "serve", "--model", "4b", "--port", str(PORT), "--allow-switch"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        wait_for_server(proc)
        print("Server is up and healthy.")

        # 1. Test GET / (Chat UI HTML)
        status, html = req("/")
        check("GET / returns 200", status == 200)
        check("HTML has title Kapitsa", "<title>Kapitsa</title>" in html)
        check("HTML has model-select dropdown", 'id="model-select"' in html)
        check("HTML has model-status indicator", 'id="model-status"' in html)
        check("HTML has switchModel implementation", "switchModel" in html)
        check("HTML has /v1/chat/completions endpoint", "/v1/chat/completions" in html)

        # 2. Test GET /v1/models
        status, body = req("/v1/models")
        check("GET /v1/models returns 200", status == 200)
        models_data = json.loads(body)
        check("Response has object list", models_data.get("object") == "list")
        check("Response has data array", isinstance(models_data.get("data"), list) and len(models_data["data"]) > 0)

        first_model = models_data["data"][0]
        check("First model is active", first_model.get("active") is True)
        check("First model is 4b", "4b" in first_model.get("id"))
        check("First model has context_length", first_model.get("context_length") > 0)
        check("First model has slots", first_model.get("slots") >= 1)

        # Check other downloaded models are listed with downloaded: true
        downloaded = [m for m in models_data["data"] if m.get("downloaded")]
        check("Multiple downloaded models listed", len(downloaded) >= 2)
        print(f"  Found {len(downloaded)} ready models and {len(models_data['data'])} total models.")

        # 3. Test GET /v1/models/{model_id}
        status, single_body = req(f"/v1/models/{first_model['id']}")
        check("GET /v1/models/:id returns 200", status == 200)
        single_data = json.loads(single_body)
        check("Single model matches ID", single_data.get("id") == first_model["id"])

        # 4. Test Chat Completion on current model (4b)
        print("Testing chat completion on current model...")
        status, chat_resp = req("/v1/chat/completions", {
            "model": "qwen3.5-4b:4bit",
            "messages": [{"role": "user", "content": "Respond with the single word: PONG"}],
            "max_tokens": 16,
            "stream": False
        })
        check("Chat completion on 4b succeeds", status == 200, chat_resp)
        chat_data = json.loads(chat_resp)
        check("Chat completion model is 4b", "4b" in chat_data.get("model", ""))

        # 5. Test explicit switch endpoint: POST /v1/models/switch to 9b
        print("Testing explicit switch to 9b via POST /v1/models/switch...")
        status, switch_resp = req("/v1/models/switch", {"model": "9b"})
        check("POST /v1/models/switch returns 200", status == 200, switch_resp)
        switch_data = json.loads(switch_resp)
        check("Switch status is ok", switch_data.get("status") == "ok", json.dumps(switch_data), proc)
        check("New model is 9b", "9b" in switch_data.get("model", ""), json.dumps(switch_data), proc)
        check("Previous model was 4b", "4b" in switch_data.get("previous_model", ""), json.dumps(switch_data))

        # Verify /v1/models now reflects 9b as active
        status, body2 = req("/v1/models")
        models_data2 = json.loads(body2)
        active2 = models_data2["data"][0]
        check("Active model is now 9b", "9b" in active2.get("id") and active2.get("active") is True)

        # 6. Test on-demand dynamic model switching via Chat Completion request!
        # Send a request with model: "4b" while 9b is currently active
        print("Testing on-demand switching back to 4b via Chat Completion request...")
        status, chat_switch_resp = req("/v1/chat/completions", {
            "model": "4b",
            "messages": [{"role": "user", "content": "Respond with the single word: OK"}],
            "max_tokens": 16,
            "stream": False
        })
        check("On-demand chat completion returns 200", status == 200, chat_switch_resp)
        chat_switch_data = json.loads(chat_switch_resp)
        check("Chat completion switched back to 4b", "4b" in chat_switch_data.get("model", ""))

        # Verify /v1/models reflects 4b active again
        status, body3 = req("/v1/models")
        models_data3 = json.loads(body3)
        check("Active model is back to 4b", "4b" in models_data3["data"][0]["id"])

        # 7. Test on-demand dynamic model switching via Anthropic Messages endpoint!
        print("Testing on-demand switching to 9b via Anthropic Messages endpoint...")
        status, msg_resp = req("/v1/messages", {
            "model": "9b",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "max_tokens": 16
        })
        check("Anthropic messages returns 200", status == 200, msg_resp)
        msg_data = json.loads(msg_resp)
        check("Anthropic response model is 9b", "9b" in msg_data.get("model", ""))

        status, body4 = req("/v1/models")
        models_data4 = json.loads(body4)
        check("Active model is now 9b after Anthropic message", "9b" in models_data4["data"][0]["id"])

        # 8. Test count_tokens with model
        status, tok_resp = req("/v1/messages/count_tokens", {
            "model": "4b",
            "messages": [{"role": "user", "content": "Count these tokens please"}]
        })
        check("Count tokens returns 200", status == 200, tok_resp)
        tok_data = json.loads(tok_resp)
        check("Count tokens returned input_tokens", tok_data.get("input_tokens", 0) > 0)

        # 9. Test streaming chat completion with on-demand switch back to 4b
        print("Testing streaming chat completion with on-demand switch back to 4b...")
        url = f"{BASE}/v1/chat/completions"
        s_req = urllib.request.Request(
            url,
            data=json.dumps({
                "model": "4b",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "max_tokens": 8
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(s_req, timeout=30) as s_resp:
            stream_body = s_resp.read().decode("utf-8")
            check("Streaming returns 200", s_resp.status == 200)
            check("Streaming has data: events", "data: " in stream_body)
            check("Streaming has [DONE]", "[DONE]" in stream_body)

        status, body5 = req("/v1/models")
        models_data5 = json.loads(body5)
        check("Active model switched to 4b via streaming", "4b" in models_data5["data"][0]["id"])

        # 10. A switch that fails leaves the server on its model, and still answering
        status, bad = req("/v1/models/switch", {"model": "no-such-model-xyz"})
        check("Switch to an unknown model fails with 400", status == 400, bad)
        status, body6 = req("/v1/models")
        check("Active model unchanged after a failed switch", "4b" in json.loads(body6)["data"][0]["id"])
        status, ok = req("/v1/chat/completions", {
            "model": "default", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]})
        check("Server still answers after a failed switch", status == 200, ok)

        print("\nALL DYNAMIC MODEL SELECTION AND SWITCHING TESTS PASSED SUCCESSFULLY!")

    finally:
        print("Terminating test server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

if __name__ == "__main__":
    main()
