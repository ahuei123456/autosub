"""Tests for Docker configuration files."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
GCLOUDIGNORE = ROOT / ".gcloudignore"
COMPOSEFILE = ROOT / "docker-compose.yml"


# ── Dockerfile ──────────────────────────────────────────────────────


class TestDockerfile:
    def test_exists(self):
        assert DOCKERFILE.exists()

    def test_base_image_is_python_slim(self):
        content = DOCKERFILE.read_text()
        assert "python:3.12-slim" in content

    def test_installs_ffmpeg(self):
        content = DOCKERFILE.read_text()
        assert "ffmpeg" in content

    def test_installs_uv(self):
        content = DOCKERFILE.read_text()
        assert "astral-sh/uv" in content

    def test_deps_installed_before_source(self):
        """uv sync for deps should run before COPY . . for layer caching."""
        content = DOCKERFILE.read_text()
        lines = content.splitlines()
        first_sync = next(
            i for i, l in enumerate(lines) if "uv sync" in l
        )
        copy_all = next(
            i for i, l in enumerate(lines) if l.strip() == "COPY . ."
        )
        assert first_sync < copy_all

    def test_no_dev_deps_in_entrypoint(self):
        content = DOCKERFILE.read_text()
        entrypoint_lines = [l for l in content.splitlines() if "ENTRYPOINT" in l]
        assert entrypoint_lines
        assert "--no-dev" in entrypoint_lines[0]

    def test_no_dev_deps_in_sync(self):
        content = DOCKERFILE.read_text()
        sync_lines = [l for l in content.splitlines() if "uv sync" in l]
        assert all("--no-dev" in l for l in sync_lines)


# ── .dockerignore ───────────────────────────────────────────────────


class TestDockerignore:
    @pytest.fixture()
    def patterns(self):
        return DOCKERIGNORE.read_text().splitlines()

    def test_exists(self):
        assert DOCKERIGNORE.exists()

    def test_excludes_git(self, patterns):
        assert ".git" in patterns

    def test_excludes_venv(self, patterns):
        assert ".venv" in patterns

    def test_excludes_projects(self, patterns):
        assert "projects/" in patterns

    def test_excludes_tests(self, patterns):
        assert "tests/" in patterns

    @pytest.mark.parametrize("ext", ["*.mp4", "*.mkv", "*.wav", "*.mp3"])
    def test_excludes_media(self, patterns, ext):
        assert ext in patterns

    def test_does_not_exclude_profiles(self, patterns):
        assert "profiles/" not in patterns
        assert "profiles/local/" not in patterns

    def test_does_not_exclude_prompts(self, patterns):
        assert "prompts/" not in patterns
        assert "prompts/local/" not in patterns


# ── .gcloudignore ───────────────────────────────────────────────────


class TestGcloudignore:
    @pytest.fixture()
    def patterns(self):
        return GCLOUDIGNORE.read_text().splitlines()

    def test_exists(self):
        assert GCLOUDIGNORE.exists()

    def test_does_not_exclude_profiles_local(self, patterns):
        """profiles/local must be included for Cloud Build to bake them in."""
        assert "profiles/local/" not in patterns
        assert "/profiles/local/" not in patterns

    def test_does_not_exclude_prompts_local(self, patterns):
        assert "prompts/local/" not in patterns
        assert "/prompts/local/" not in patterns

    def test_excludes_projects(self, patterns):
        assert "projects/" in patterns

    def test_excludes_tests(self, patterns):
        assert "tests/" in patterns

    @pytest.mark.parametrize("ext", ["*.mp4", "*.mkv", "*.wav", "*.mp3"])
    def test_excludes_media(self, patterns, ext):
        assert ext in patterns


# ── docker-compose.yml ──────────────────────────────────────────────


class TestComposeFile:
    @pytest.fixture()
    def content(self):
        return COMPOSEFILE.read_text()

    def test_exists(self):
        assert COMPOSEFILE.exists()

    def test_no_hardcoded_project(self, content):
        """Project ID should come from env vars, not be hardcoded."""
        assert "future-name-201021" not in content

    def test_no_hardcoded_paths(self, content):
        assert "/home/" not in content

    def test_image_is_configurable(self, content):
        assert "AUTOSUB_IMAGE" in content

    def test_project_dir_is_configurable(self, content):
        assert "AUTOSUB_PROJECTS_DIR" in content

    def test_mounts_projects_volume(self, content):
        assert "/projects" in content

    def test_passes_gcp_project_env(self, content):
        assert "GOOGLE_CLOUD_PROJECT" in content
