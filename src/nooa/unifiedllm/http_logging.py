# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from nooa.tracing._secret_scrubber import REDACTED, _is_sensitive_key, scrub_value


def enable_http_request_logging(
    output_dir: str | Path = ".",
    url_filter: str | None = None,
    save_responses: bool = False,
    errors_only: bool = False,
    verbose: bool = True,
    force_httpx: bool = True,
):
    """Enable logging of HTTP requests to JSON files for debugging.

    This function patches httpx.HTTPTransport and httpx.AsyncHTTPTransport to intercept
    and log all HTTP requests made through httpx (used by OpenAI SDK, litellm, etc.).
    Each request is saved to a JSON file with a sequential counter.

    IMPORTANT: By default, litellm uses aiohttp transport instead of httpx. This function
    automatically sets DISABLE_AIOHTTP_TRANSPORT=True to force litellm to use httpx,
    which can be intercepted. Set force_httpx=False to disable this behavior.

    Args:
        output_dir: Directory to save request/response JSON files (default: current directory)
        url_filter: Optional substring to filter URLs (only log requests matching this)
        save_responses: If True, also save response bodies to separate files
        errors_only: If True, only log requests that result in HTTP errors (status >= 400)
        verbose: If True, print summary info for each captured request
        force_httpx: If True, sets DISABLE_AIOHTTP_TRANSPORT=True so litellm uses httpx (default: True)

    Returns:
        A function that when called, disables the logging and restores original behavior

    Example:
        ```python
        from nooa.unifiedllm.http_logging import enable_http_request_logging

        # Start logging - MUST be called BEFORE creating any LLM clients
        disable_logging = enable_http_request_logging(output_dir="debug_logs")

        # Now create LLM client and make API calls...
        # (requests will be saved to debug_logs/request_1.json, request_2.json, etc.)

        # Stop logging
        disable_logging()
        ```

    Example (errors only):
        ```python
        # Only capture failed requests to JSONL file
        disable_logging = enable_http_request_logging(
            output_dir="eval_errors",
            errors_only=True,
            save_responses=True,
            verbose=False
        )
        # Errors saved to: eval_errors/llm_errors.jsonl
        ```
    """
    # Force litellm to use httpx instead of aiohttp (which we can't easily intercept)
    original_disable_aiohttp = os.environ.get("DISABLE_AIOHTTP_TRANSPORT")
    if force_httpx:
        os.environ["DISABLE_AIOHTTP_TRANSPORT"] = "True"
        if verbose:
            print("✓ Set DISABLE_AIOHTTP_TRANSPORT=True to force litellm to use httpx")

    try:
        import httpx
    except ImportError as e:
        raise ImportError(
            "httpx is required for request logging. Install it with: pip install httpx"
        ) from e

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    request_counter = [0]

    # Determine output file based on mode
    jsonl_file = output_path / "llm_errors.jsonl" if errors_only else None

    # Store original methods for both sync and async transports
    original_sync_send = httpx.HTTPTransport.handle_request
    original_async_send = httpx.AsyncHTTPTransport.handle_async_request

    def _redact_headers(headers: dict) -> dict:
        """Redact sensitive headers like Authorization."""
        redacted = {}
        for key, value in headers.items():
            if _is_sensitive_key(key):
                redacted[key] = "***REDACTED***"
            else:
                redacted[key], _ = scrub_value(value)
        return redacted

    def _redact_body(body):
        """Recursively scrub secrets before a parsed HTTP body is logged."""
        scrubbed, _ = scrub_value(body)
        if isinstance(scrubbed, dict) and scrubbed.get("grant_type") == "authorization_code":
            # "code" is too generic for global key matching, but is a credential
            # in an OAuth authorization-code exchange.
            if "code" in scrubbed:
                scrubbed["code"] = REDACTED
        return scrubbed

    def _write_jsonl_entry(entry: dict):
        """Append a JSON entry to the JSONL error file."""
        if jsonl_file is None:
            return
        with open(jsonl_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _log_request(request, counter):
        """Helper to log request details."""
        model_name = "unknown"
        try:
            if request.content:
                body_dict = json.loads(request.content.decode("utf-8"))
                model_name = body_dict.get("model", "unknown").replace("/", "_").replace(":", "_")

                filename = output_path / f"request_{counter}_{model_name}.json"

                with open(filename, "w") as f:
                    json.dump(_redact_body(body_dict), f, indent=2)

                if verbose:
                    print(f"\n💾 Saved HTTP request to: {filename}")
                    print(f"   URL: {request.url}")
                    print(f"   Method: {request.method}")
                    if "model" in body_dict:
                        print(f"   Model: {body_dict['model']}")
                    if "input" in body_dict:
                        print(f"   Input messages: {len(body_dict['input'])}")
                    if "messages" in body_dict:
                        print(f"   Messages: {len(body_dict['messages'])}")
        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to log request: {e}")
        return model_name

    def _save_response_file(response_dict, counter, model_name):
        """Save response dict to file."""
        response_filename = output_path / f"response_{counter}_{model_name}.json"
        with open(response_filename, "w") as f:
            json.dump(response_dict, f, indent=2)
        if verbose:
            print(f"   Response status: {response_dict['status_code']}")
            print(f"   💾 Saved response to: {response_filename}")

    def _log_response_sync(response, request, counter, model_name):
        """Helper to log response details (sync version)."""
        try:
            response_dict = {
                "status_code": response.status_code,
                "headers": _redact_headers(dict(response.headers)),
            }

            try:
                response_body = response.read()

                if response_body:
                    try:
                        response_dict["body"] = json.loads(response_body.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        response_dict["body"] = response_body.decode("utf-8", errors="ignore")
                else:
                    response_dict["body"] = "[Empty response body]"

                # Recreate response with the body we read
                response = httpx.Response(
                    status_code=response.status_code,
                    headers=response.headers,
                    content=response_body,
                    request=request,
                )

            except Exception as read_error:
                response_dict["body"] = f"[Error reading response: {read_error}]"

            response_dict["body"] = _redact_body(response_dict["body"])

            _save_response_file(response_dict, counter, model_name)
        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to log response: {e}")
        return response

    async def _log_response_async(response, request, counter, model_name):
        """Helper to log response details (async version)."""
        try:
            response_dict = {
                "status_code": response.status_code,
                "headers": _redact_headers(dict(response.headers)),
            }

            try:
                # Use async read for async responses
                response_body = await response.aread()

                if response_body:
                    try:
                        response_dict["body"] = json.loads(response_body.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        response_dict["body"] = response_body.decode("utf-8", errors="ignore")
                else:
                    response_dict["body"] = "[Empty response body]"

                # Recreate response with the body we read
                response = httpx.Response(
                    status_code=response.status_code,
                    headers=response.headers,
                    content=response_body,
                    request=request,
                )

            except Exception as read_error:
                response_dict["body"] = f"[Error reading response: {read_error}]"

            response_dict["body"] = _redact_body(response_dict["body"])

            _save_response_file(response_dict, counter, model_name)
        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to log response: {e}")
        return response

    # Sync transport handler
    def logging_send(self, request):
        should_log = url_filter is None or url_filter in str(request.url)

        if not should_log:
            return original_sync_send(self, request)

        request_counter[0] += 1
        counter = request_counter[0]

        # Capture request details before sending
        request_data = None
        model_name = "unknown"
        if errors_only:
            try:
                if request.content:
                    body_dict = json.loads(request.content.decode("utf-8"))
                    model_name = (
                        body_dict.get("model", "unknown").replace("/", "_").replace(":", "_")
                    )
                    request_data = {
                        "url": str(request.url),
                        "method": request.method,
                        "headers": _redact_headers(dict(request.headers)),
                        "body": _redact_body(body_dict),
                    }
            except Exception as e:
                if verbose:
                    print(f"⚠️  Failed to capture request: {e}")
        else:
            model_name = _log_request(request, counter)

        # Send request
        response = original_sync_send(self, request)

        # Handle errors_only mode
        if errors_only:
            if response.status_code >= 400 and request_data:
                # Read and capture error response
                try:
                    response_body = response.read()

                    # Try to parse response body as JSON
                    body_content = None
                    if response_body:
                        try:
                            body_content = json.loads(response_body.decode("utf-8"))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            # If not JSON, store as string
                            body_content = response_body.decode("utf-8", errors="replace")

                    response_data = {
                        "status_code": response.status_code,
                        "headers": _redact_headers(dict(response.headers)),
                        "body": _redact_body(body_content),
                    }

                    # Write JSONL entry
                    entry = {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "counter": counter,
                        "model": model_name,
                        "request": request_data,
                        "response": response_data,
                    }
                    _write_jsonl_entry(entry)

                    if verbose:
                        print(
                            f"🔴 LLM Error captured (#{counter}): {response.status_code} - {model_name}"
                        )
                        print(f"   Saved to: {jsonl_file}")

                    # Recreate response with the body we read
                    response = httpx.Response(
                        status_code=response.status_code,
                        headers=response.headers,
                        content=response_body,
                        request=request,
                    )
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Failed to log error response: {e}")
        elif save_responses:
            response = _log_response_sync(response, request, counter, model_name)

        return response

    # Async transport handler
    async def async_logging_send(self, request):
        should_log = url_filter is None or url_filter in str(request.url)

        if not should_log:
            return await original_async_send(self, request)

        request_counter[0] += 1
        counter = request_counter[0]

        # Capture request details before sending
        request_data = None
        model_name = "unknown"
        if errors_only:
            try:
                if request.content:
                    body_dict = json.loads(request.content.decode("utf-8"))
                    model_name = (
                        body_dict.get("model", "unknown").replace("/", "_").replace(":", "_")
                    )
                    request_data = {
                        "url": str(request.url),
                        "method": request.method,
                        "headers": _redact_headers(dict(request.headers)),
                        "body": _redact_body(body_dict),
                    }
            except Exception as e:
                if verbose:
                    print(f"⚠️  Failed to capture request: {e}")
        else:
            model_name = _log_request(request, counter)

        # Send request
        response = await original_async_send(self, request)

        # Handle errors_only mode
        if errors_only:
            if response.status_code >= 400 and request_data:
                # Read and capture error response
                try:
                    response_body = await response.aread()

                    # Try to parse response body as JSON
                    body_content = None
                    if response_body:
                        try:
                            body_content = json.loads(response_body.decode("utf-8"))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            # If not JSON, store as string
                            body_content = response_body.decode("utf-8", errors="replace")

                    response_data = {
                        "status_code": response.status_code,
                        "headers": _redact_headers(dict(response.headers)),
                        "body": _redact_body(body_content),
                    }

                    # Write JSONL entry
                    entry = {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "counter": counter,
                        "model": model_name,
                        "request": request_data,
                        "response": response_data,
                    }
                    _write_jsonl_entry(entry)

                    if verbose:
                        print(
                            f"🔴 LLM Error captured (#{counter}): {response.status_code} - {model_name}"
                        )
                        print(f"   Saved to: {jsonl_file}")

                    # Recreate response with the body we read
                    response = httpx.Response(
                        status_code=response.status_code,
                        headers=response.headers,
                        content=response_body,
                        request=request,
                    )
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Failed to log error response: {e}")
        elif save_responses:
            response = await _log_response_async(response, request, counter, model_name)

        return response

    # Patch both sync and async transports
    httpx.HTTPTransport.handle_request = logging_send
    httpx.AsyncHTTPTransport.handle_async_request = async_logging_send

    if verbose:
        print("✓ HTTP logging patched (sync + async transports)")

    def disable():
        httpx.HTTPTransport.handle_request = original_sync_send
        httpx.AsyncHTTPTransport.handle_async_request = original_async_send
        # Restore original env var
        if force_httpx:
            if original_disable_aiohttp is None:
                os.environ.pop("DISABLE_AIOHTTP_TRANSPORT", None)
            else:
                os.environ["DISABLE_AIOHTTP_TRANSPORT"] = original_disable_aiohttp
        if verbose:
            print(
                f"\n✅ HTTP request logging disabled. Logged {request_counter[0]} requests to {output_path}"
            )

    return disable
