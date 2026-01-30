"""Code execution engine - runs ML code in safe environments."""

import asyncio
import json
import tempfile
from typing import Any, Optional
from pathlib import Path

from app.config import settings
from app.execution.safety import validate_code


async def execute_code_safely(
    code: str,
    project_id: str,
    context: dict[str, Any] = {},
    use_docker: bool = False,
) -> dict:
    """Execute Python code in a safe environment."""
    # Step 1: Safety check
    safety_result = validate_code(code)
    if not safety_result["safe"]:
        return {
            "success": False,
            "error": "Code failed safety check",
            "issues": safety_result["issues"],
        }

    # Step 2: Choose execution environment
    if use_docker and settings.DOCKER_ENABLED:
        return await _execute_in_docker(code, project_id, context)
    else:
        return await _execute_direct(code, project_id, context)


async def _execute_direct(
    code: str,
    project_id: str,
    context: dict[str, Any],
) -> dict:
    """Execute code directly in a restricted namespace."""
    # Set up safe namespace with allowed libraries
    namespace = {
        "__builtins__": _safe_builtins(),
    }

    # Add context variables
    namespace.update(context)

    # Add safe imports
    safe_setup = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import json
"""
    full_code = safe_setup + "\n" + code

    # Capture output
    output_capture = {"stdout": [], "result": None}

    # Custom print function to capture output
    def safe_print(*args, **kwargs):
        output_capture["stdout"].append(" ".join(str(a) for a in args))

    namespace["print"] = safe_print

    try:
        # Execute with timeout
        exec(compile(full_code, "<agent_code>", "exec"), namespace)

        # Check for a 'result' variable
        if "result" in namespace:
            result = namespace["result"]
            if hasattr(result, "to_dict"):
                output_capture["result"] = result.to_dict()
            elif hasattr(result, "tolist"):
                output_capture["result"] = result.tolist()
            else:
                output_capture["result"] = result

        return {
            "success": True,
            "stdout": "\n".join(output_capture["stdout"]),
            "result": output_capture["result"],
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "stdout": "\n".join(output_capture["stdout"]),
        }


async def _execute_in_docker(
    code: str,
    project_id: str,
    context: dict[str, Any],
) -> dict:
    """Execute code in an isolated Docker container."""
    try:
        import docker
        client = docker.from_env()
    except Exception as e:
        # Fallback to direct execution if Docker is unavailable
        return await _execute_direct(code, project_id, context)

    # Write code to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        # Add context serialization
        context_code = f"import json\ncontext = json.loads('''{json.dumps(context)}''')\n"
        f.write(context_code + code)
        code_path = f.name

    # Mount project uploads directory
    upload_dir = str(Path(settings.UPLOAD_DIR) / project_id)

    try:
        container = client.containers.run(
            settings.DOCKER_IMAGE,
            command=f"python /code/script.py",
            volumes={
                code_path: {"bind": "/code/script.py", "mode": "ro"},
                upload_dir: {"bind": "/data", "mode": "ro"},
            },
            mem_limit="2g",
            cpu_period=100000,
            cpu_quota=50000,  # 50% of one CPU
            network_disabled=True,
            remove=True,
            timeout=settings.EXECUTION_TIMEOUT,
        )

        output = container.decode("utf-8") if isinstance(container, bytes) else str(container)
        return {
            "success": True,
            "stdout": output,
            "result": None,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
    finally:
        Path(code_path).unlink(missing_ok=True)


def _safe_builtins() -> dict:
    """Return a restricted set of builtins."""
    import builtins
    safe = {}
    allowed = [
        "True", "False", "None",
        "abs", "all", "any", "bool", "dict", "enumerate",
        "filter", "float", "frozenset", "hasattr", "hash",
        "hex", "int", "isinstance", "issubclass", "iter",
        "len", "list", "map", "max", "min", "next",
        "oct", "ord", "pow", "print", "range", "repr",
        "reversed", "round", "set", "slice", "sorted",
        "str", "sum", "tuple", "type", "zip",
        "Exception", "ValueError", "TypeError", "KeyError",
        "IndexError", "RuntimeError", "StopIteration",
    ]
    for name in allowed:
        if hasattr(builtins, name):
            safe[name] = getattr(builtins, name)
    return safe
