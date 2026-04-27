import json
import sys
import os
import time
import traceback
import argparse

_parser = argparse.ArgumentParser(description="Run HuggingGPT on a JSONL benchmark file")
_parser.add_argument("input", nargs="?", default="BLINK_100sample.jsonl", help="Input JSONL file")
_parser.add_argument("output", nargs="?", default="result.json", help="Output JSON file")
_parser.add_argument("--config", type=str, default="configs/config.default.yaml",
                     help="Path to HuggingGPT config YAML (use configs/config.multiround.yaml for multi-round)")
_cli_args = _parser.parse_args()

input_file = _cli_args.input
output_file = _cli_args.output

os.environ["AWESOME_CHAT_CONFIG"] = _cli_args.config
sys.argv = [sys.argv[0], "--mode", "cli"]

from awesome_chat import chat_huggingface, API_KEY, API_TYPE, API_ENDPOINT

def load_jsonl(path):
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if not samples:
        with open(path, "r") as f:
            content = f.read().strip()
            if content:
                samples.append(json.loads(content))
    return samples

def build_message(sample):
    images = sample.get("image", [])
    conversation = sample["conversations"]
    question = next(c["value"] for c in conversation if c["from"] == "human")
    if images:
        image_refs = ", ".join(images)
        content = f"Given the image(s): {image_refs}\n\n{question}"
    else:
        content = question
    return [{"role": "user", "content": content}]

def _extract_error_message(resp):
    """Pull out an error message from any of the shapes send_request/chat_huggingface may return."""
    if not isinstance(resp, dict):
        return ""
    # Shape 1: top-level "error" (from send_request raw error dict)
    err = resp.get("error")
    if isinstance(err, dict):
        return str(err.get("message", err))
    if isinstance(err, str):
        return err
    return ""

def is_failed(entry):
    """Return True if an entry has an error and should be retried."""
    resp = entry.get("response", {})
    if not isinstance(resp, dict):
        return True
    # Hard error: send_request returned an error dict (no user-facing "message")
    if "error" in resp and "message" not in resp:
        return True
    # Soft error: LLM reported inference failure - retry
    msg = resp.get("message", "")
    failure_phrases = [
        "error when running the model inference",
        "no results were produced",
        "i cannot provide a direct answer",
        "i am unable to provide",
        "i can't make it",
    ]
    if any(p in msg.lower() for p in failure_phrases):
        return True
    # Upstream API error that bubbled up as a message
    err_msg = _extract_error_message(resp) or msg
    api_failures = ["upstream error", "do request failed", "openai_error",
                    "network error", "request failed after", "http 5"]
    if any(p in err_msg.lower() for p in api_failures):
        return True
    return False

def main():
    samples = load_jsonl(input_file)
    print(f"Loaded {len(samples)} samples from {input_file}")

    # Load existing results and index by id for resume/retry
    existing = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            try:
                for entry in json.load(f):
                    existing[entry["id"]] = entry
            except Exception:
                pass
        failed = sum(1 for e in existing.values() if is_failed(e))
        skipped = len(existing) - failed
        print(f"Resuming: {skipped} already done, {failed} failed (will retry)")

    results = []
    for i, sample in enumerate(samples):
        sample_id = sample.get("id", i)
        gt_answer = next(
            (c["value"] for c in sample["conversations"] if c["from"] == "gpt"), None
        )

        # Skip if already successfully processed
        if sample_id in existing and not is_failed(existing[sample_id]):
            results.append(existing[sample_id])
            print(f"[{i+1}/{len(samples)}] Skipping {sample_id} (already done)")
            continue

        messages = build_message(sample)
        print(f"[{i+1}/{len(samples)}] Processing {sample_id} ...")
        # send_request() in awesome_chat.py does per-call retries with backoff
        # (up to ~78s), so the outer loop only needs a light safety-net retry
        # in case a whole pipeline run still fails end-to-end.
        retries = 2
        answer = None
        for attempt in range(1, retries + 1):
            try:
                answer = chat_huggingface(messages, API_KEY, API_TYPE, API_ENDPOINT)
                msg = ""
                if isinstance(answer, dict):
                    msg = answer.get("message", "") or _extract_error_message(answer)
                api_failures = ["upstream error", "do request failed", "openai_error",
                                "network error", "request failed after", "http 5"]
                if any(p in msg.lower() for p in api_failures):
                    raise RuntimeError(f"API error (attempt {attempt}): {msg[:160]}")
                break
            except Exception as e:
                print(f"  Attempt {attempt}/{retries} failed: {e}")
                if attempt < retries:
                    wait = 10  # inner layer already backed off; short pause here
                    print(f"  Waiting {wait}s before full-pipeline retry...")
                    time.sleep(wait)
                else:
                    traceback.print_exc()
                    if answer is None:
                        answer = {"error": str(e)}

        result_entry = {
            "id": sample_id,
            "task": sample.get("task"),
            "ground_truth": gt_answer,
            "input": messages[-1]["content"],
            "response": answer,
        }
        results.append(result_entry)
        status = "FAILED" if is_failed(result_entry) else "OK"
        print(f"  [{status}] Response: {json.dumps(answer)[:200]}")

        # Save incrementally
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    failed_count = sum(1 for r in results if is_failed(r))
    print(f"\nDone. {len(results)} results saved to {output_file} ({failed_count} failed)")

if __name__ == "__main__":
    main()
