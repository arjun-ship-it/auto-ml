"""Docker sandbox management for ML training jobs."""

from pathlib import Path
from typing import Optional

from app.config import settings


SANDBOX_DOCKERFILE = """
FROM python:3.11-slim

# Install ML dependencies
RUN pip install --no-cache-dir \\
    pandas==2.2.3 \\
    numpy==1.26.4 \\
    scikit-learn==1.5.2 \\
    xgboost==2.1.1 \\
    lightgbm==4.5.0 \\
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu \\
    statsmodels==0.14.4 \\
    prophet==1.1.6 \\
    matplotlib==3.9.2 \\
    seaborn==0.13.2 \\
    scipy==1.14.0

# Create non-root user
RUN useradd -m -s /bin/bash sandbox
USER sandbox

WORKDIR /code
"""


class SandboxManager:
    """Manages Docker sandbox environments for ML execution."""

    def __init__(self):
        self._docker_client = None

    @property
    def docker_client(self):
        if self._docker_client is None:
            try:
                import docker
                self._docker_client = docker.from_env()
            except Exception:
                self._docker_client = None
        return self._docker_client

    async def ensure_image_exists(self) -> bool:
        """Build the sandbox Docker image if it doesn't exist."""
        if not self.docker_client:
            return False

        try:
            self.docker_client.images.get(settings.DOCKER_IMAGE)
            return True
        except Exception:
            pass

        # Build the image
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                dockerfile_path = Path(tmpdir) / "Dockerfile"
                dockerfile_path.write_text(SANDBOX_DOCKERFILE)

                self.docker_client.images.build(
                    path=tmpdir,
                    tag=settings.DOCKER_IMAGE,
                    rm=True,
                )
            return True
        except Exception as e:
            print(f"Failed to build sandbox image: {e}")
            return False

    async def cleanup_old_containers(self, max_age_hours: int = 1):
        """Remove old sandbox containers."""
        if not self.docker_client:
            return

        try:
            containers = self.docker_client.containers.list(
                all=True,
                filters={"ancestor": settings.DOCKER_IMAGE},
            )
            for container in containers:
                if container.status in ["exited", "dead"]:
                    container.remove()
        except Exception:
            pass

    def is_available(self) -> bool:
        """Check if Docker is available."""
        return self.docker_client is not None


sandbox_manager = SandboxManager()
