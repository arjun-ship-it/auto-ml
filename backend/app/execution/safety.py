"""Code safety checker - validates generated code before execution."""

import ast
import re
from typing import Optional


FORBIDDEN_IMPORTS = [
    "os", "subprocess", "shutil", "sys", "importlib",
    "socket", "http", "urllib", "requests",
    "ctypes", "signal", "threading", "multiprocessing",
]

FORBIDDEN_BUILTINS = [
    "exec", "eval", "compile", "__import__",
    "open",  # Only allow through safe wrappers
    "globals", "locals", "vars",
    "getattr", "setattr", "delattr",
]

ALLOWED_MODULES = [
    "pandas", "numpy", "sklearn", "scipy", "statsmodels",
    "xgboost", "lightgbm", "torch", "matplotlib", "seaborn",
    "json", "math", "datetime", "collections", "itertools",
    "functools", "typing", "dataclasses", "prophet",
]


class CodeSafetyChecker:
    """Validates Python code for safety before execution."""

    def __init__(self):
        self.issues: list[str] = []

    def check(self, code: str) -> dict:
        """Run all safety checks on code."""
        self.issues = []

        self._check_syntax(code)
        if self.issues:
            return {"safe": False, "issues": self.issues}

        self._check_imports(code)
        self._check_forbidden_calls(code)
        self._check_file_access(code)
        self._check_network_access(code)
        self._check_system_commands(code)

        return {
            "safe": len(self.issues) == 0,
            "issues": self.issues,
        }

    def _check_syntax(self, code: str):
        """Check if code is valid Python."""
        try:
            ast.parse(code)
        except SyntaxError as e:
            self.issues.append(f"Syntax error: {e}")

    def _check_imports(self, code: str):
        """Check for forbidden imports."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in FORBIDDEN_IMPORTS:
                        self.issues.append(f"Forbidden import: {module}")
                    elif module not in ALLOWED_MODULES:
                        self.issues.append(f"Unknown module: {module} (not in allowed list)")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module in FORBIDDEN_IMPORTS:
                        self.issues.append(f"Forbidden import: {module}")

    def _check_forbidden_calls(self, code: str):
        """Check for forbidden function calls."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in FORBIDDEN_BUILTINS:
                        self.issues.append(f"Forbidden builtin: {node.func.id}()")

    def _check_file_access(self, code: str):
        """Check for unauthorized file access."""
        dangerous_patterns = [
            r"open\s*\(",
            r"Path\s*\(\s*['\"/]",
            r"\.write\s*\(",
            r"\.unlink\s*\(",
            r"rmtree",
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                self.issues.append(f"Potential file system access: {pattern}")

    def _check_network_access(self, code: str):
        """Check for network access attempts."""
        network_patterns = [
            r"requests\.", r"urllib\.", r"http\.",
            r"socket\.", r"urlopen", r"fetch",
        ]
        for pattern in network_patterns:
            if re.search(pattern, code):
                self.issues.append(f"Network access attempt: {pattern}")

    def _check_system_commands(self, code: str):
        """Check for system command execution."""
        system_patterns = [
            r"os\.system", r"os\.popen", r"subprocess",
            r"Popen", r"call\(", r"check_output",
        ]
        for pattern in system_patterns:
            if re.search(pattern, code):
                self.issues.append(f"System command attempt: {pattern}")


def validate_code(code: str) -> dict:
    """Convenience function to validate code safety."""
    checker = CodeSafetyChecker()
    return checker.check(code)
