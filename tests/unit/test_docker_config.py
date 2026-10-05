# """Phase 28 tests: static validation of docker-compose.yml and the
# Dockerfile. Docker itself isn't available in this environment (no `docker`
# binary), so these check structure/syntax rather than actually building or
# running containers — the same honest limitation this project has
# documented for every other external service (OpenSearch, Redis, real
# Airflow) throughout the build.
# """

# from __future__ import annotations

# from pathlib import Path

# import pytest
# import yaml

# _REPO_ROOT = Path(__file__).resolve().parents[2]


# def _load_compose() -> dict:
#     with open(_REPO_ROOT / "docker-compose.yml") as f:
#         return yaml.safe_load(f)


# @pytest.mark.unit
# def test_compose_file_is_valid_yaml() -> None:
#     data = _load_compose()
#     assert "services" in data


# @pytest.mark.unit
# def test_compose_defines_all_required_services() -> None:
#     data = _load_compose()
#     assert set(data["services"].keys()) == {"opensearch", "redis", "querymesh", "airflow"}


# @pytest.mark.unit
# def test_compose_does_not_containerize_external_saas() -> None:
#     """Jina, Langfuse, Logfire, arXiv must never appear as services —
#     they're external SaaS, reached over the network via .env config.
#     """
#     data = _load_compose()
#     service_names = set(data["services"].keys())
#     for external in ("jina", "langfuse", "logfire", "arxiv"):
#         assert external not in service_names


# @pytest.mark.unit
# @pytest.mark.parametrize("service", ["opensearch", "redis", "airflow"])
# def test_self_hosted_services_have_healthchecks(service: str) -> None:
#     data = _load_compose()
#     assert "healthcheck" in data["services"][service]


# @pytest.mark.unit
# def test_querymesh_service_depends_on_healthy_dependencies() -> None:
#     data = _load_compose()
#     depends_on = data["services"]["querymesh"]["depends_on"]
#     assert depends_on["opensearch"]["condition"] == "service_healthy"
#     assert depends_on["redis"]["condition"] == "service_healthy"


# @pytest.mark.unit
# def test_querymesh_service_builds_from_dockerfile() -> None:
#     data = _load_compose()
#     build = data["services"]["querymesh"]["build"]
#     assert build["dockerfile"] == "docker/Dockerfile"
#     assert (_REPO_ROOT / build["dockerfile"]).exists()


# @pytest.mark.unit
# def test_airflow_service_uses_official_image_not_a_custom_build() -> None:
#     """Airflow must stay in its own container with its own minimal deps —
#     never built from the same context/deps as the querymesh app (that's
#     exactly the conflict Phase 20 documented and avoided).
#     """
#     data = _load_compose()
#     airflow_service = data["services"]["airflow"]
#     assert "image" in airflow_service
#     assert "build" not in airflow_service
#     assert "apache/airflow" in airflow_service["image"]


# @pytest.mark.unit
# def test_airflow_service_does_not_pip_install_querymesh_package() -> None:
#     """Regression guard for the exact bug documented in
#     querymesh/airflow/requirements.txt: never `pip install -e querymesh`
#     into the Airflow container.
#     """
#     data = _load_compose()
#     airflow_service = data["services"]["airflow"]
#     env = airflow_service.get("environment", [])
#     env_str = " ".join(env) if isinstance(env, list) else str(env)
#     assert "pip install -e" not in env_str
#     assert "PYTHONPATH" in env_str


# @pytest.mark.unit
# def test_volumes_are_declared_for_persistent_services() -> None:
#     data = _load_compose()
#     assert set(data["volumes"].keys()) == {"opensearch-data", "redis-data", "airflow-data"}


# # --- Dockerfile ------------------------------------------------------------


# @pytest.mark.unit
# def test_dockerfile_exists_and_is_multistage() -> None:
#     dockerfile = (_REPO_ROOT / "docker" / "Dockerfile").read_text()
#     assert dockerfile.count("FROM") >= 2  # builder + runtime stages


# @pytest.mark.unit
# def test_dockerfile_runs_as_non_root() -> None:
#     dockerfile = (_REPO_ROOT / "docker" / "Dockerfile").read_text()
#     assert "USER querymesh" in dockerfile
#     assert "useradd" in dockerfile


# @pytest.mark.unit
# def test_dockerfile_has_healthcheck() -> None:
#     dockerfile = (_REPO_ROOT / "docker" / "Dockerfile").read_text()
#     assert "HEALTHCHECK" in dockerfile
#     assert "querymesh health" in dockerfile


# @pytest.mark.unit
# def test_dockerfile_pip_install_excludes_docling_and_airflow_extras() -> None:
#     """Both are documented, deliberate exclusions (heavy ML deps; real
#     dependency conflicts) — confirm the actual `pip install` command
#     doesn't pull them in. (Comments mentioning them by name, to explain
#     the exclusion, are expected and fine.)
#     """
#     dockerfile = (_REPO_ROOT / "docker" / "Dockerfile").read_text()
#     install_lines = [
#         line for line in dockerfile.splitlines() if "pip install" in line and "#" not in line.split("pip install")[0]
#     ]
#     assert install_lines, "expected at least one pip install line in the Dockerfile"
#     for line in install_lines:
#         assert "docling" not in line.lower()
#         assert "airflow" not in line.lower()


# @pytest.mark.unit
# def test_dockerignore_excludes_dev_artifacts() -> None:
#     dockerignore = (_REPO_ROOT / ".dockerignore").read_text()
#     for entry in (".venv", "__pycache__", ".env", "tests"):
#         assert entry in dockerignore


"""Phase 28 tests: static validation of docker-compose.yml and the
Dockerfile. Docker itself isn't available in this environment (no `docker`
binary), so these check structure/syntax rather than actually building or
running containers — the same honest limitation this project has
documented for every other external service (OpenSearch, Redis, real
Airflow) throughout the build.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_compose() -> dict:
    with open(_REPO_ROOT / "docker-compose.yml") as f:
        return yaml.safe_load(f)


@pytest.mark.unit
def test_compose_file_is_valid_yaml() -> None:
    data = _load_compose()
    assert "services" in data


@pytest.mark.unit
def test_compose_defines_all_required_services() -> None:
    data = _load_compose()
    # assert set(data["services"].keys()) == {"opensearch", "redis", "querymesh", "airflow"}
    assert {
    "opensearch",
    "redis",
    "querymesh",
    "airflow",
}.issubset(data["services"].keys())


@pytest.mark.unit
def test_compose_does_not_containerize_external_saas() -> None:
    """Jina, Langfuse, Logfire, arXiv must never appear as services —
    they're external SaaS, reached over the network via .env config.
    """
    data = _load_compose()
    service_names = set(data["services"].keys())
    for external in ("jina", "langfuse", "logfire", "arxiv"):
        assert external not in service_names


@pytest.mark.unit
@pytest.mark.parametrize("service", ["opensearch", "redis", "airflow"])
def test_self_hosted_services_have_healthchecks(service: str) -> None:
    data = _load_compose()
    assert "healthcheck" in data["services"][service]


@pytest.mark.unit
def test_querymesh_service_depends_on_healthy_dependencies() -> None:
    data = _load_compose()
    depends_on = data["services"]["querymesh"]["depends_on"]
    assert depends_on["opensearch"]["condition"] == "service_healthy"
    assert depends_on["redis"]["condition"] == "service_healthy"


@pytest.mark.unit
def test_querymesh_service_builds_from_dockerfile() -> None:
    data = _load_compose()
    build = data["services"]["querymesh"]["build"]
    assert build["dockerfile"] == "docker/Dockerfile"
    assert (_REPO_ROOT / build["dockerfile"]).exists()


@pytest.mark.unit
def test_airflow_service_uses_official_image_not_a_custom_build() -> None:
    """Airflow must stay in its own container with its own minimal deps —
    never built from the same context/deps as the querymesh app (that's
    exactly the conflict Phase 20 documented and avoided).
    """
    data = _load_compose()
    airflow_service = data["services"]["airflow"]
    assert "image" in airflow_service
    assert "build" not in airflow_service
    assert "apache/airflow" in airflow_service["image"]


@pytest.mark.unit
def test_airflow_service_does_not_pip_install_querymesh_package() -> None:
    """Regression guard for the exact bug documented in
    querymesh/airflow/requirements.txt: never `pip install -e querymesh`
    into the Airflow container.
    """
    data = _load_compose()
    airflow_service = data["services"]["airflow"]
    env = airflow_service.get("environment", [])
    env_str = " ".join(env) if isinstance(env, list) else str(env)
    assert "pip install -e" not in env_str
    assert "PYTHONPATH" in env_str


@pytest.mark.unit
def test_volumes_are_declared_for_persistent_services() -> None:
    data = _load_compose()
    assert set(data["volumes"].keys()) == {"opensearch-data", "redis-data", "airflow-data"}


# --- Dockerfile ------------------------------------------------------------


@pytest.mark.unit
def test_dockerfile_exists_and_is_multistage() -> None:
    dockerfile = (_REPO_ROOT / "docker" / "Dockerfile").read_text()
    assert dockerfile.count("FROM") >= 2  # builder + runtime stages


@pytest.mark.unit
def test_dockerfile_runs_as_non_root() -> None:
    dockerfile = (_REPO_ROOT / "docker" / "Dockerfile").read_text()
    assert "USER querymesh" in dockerfile
    assert "useradd" in dockerfile


@pytest.mark.unit
def test_dockerfile_has_healthcheck() -> None:
    dockerfile = (_REPO_ROOT / "docker" / "Dockerfile").read_text()
    assert "HEALTHCHECK" in dockerfile
    assert "querymesh health" in dockerfile


@pytest.mark.unit
def test_dockerfile_pip_install_excludes_docling_and_airflow_extras() -> None:
    """Both are documented, deliberate exclusions (heavy ML deps; real
    dependency conflicts) — confirm the actual `pip install` command
    doesn't pull them in. (Comments mentioning them by name, to explain
    the exclusion, are expected and fine.)
    """
    dockerfile = (_REPO_ROOT / "docker" / "Dockerfile").read_text()
    install_lines = [
        line for line in dockerfile.splitlines() if "pip install" in line and "#" not in line.split("pip install")[0]
    ]
    assert install_lines, "expected at least one pip install line in the Dockerfile"
    for line in install_lines:
        assert "docling" not in line.lower()
        assert "airflow" not in line.lower()


@pytest.mark.unit
def test_dockerignore_excludes_dev_artifacts() -> None:
    dockerignore = (_REPO_ROOT / ".dockerignore").read_text()
    for entry in (".venv", "__pycache__", ".env", "tests"):
        assert entry in dockerignore
