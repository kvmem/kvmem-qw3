# Preserve upstream dependencies and APIs; distinguish qualification status from
# a completed conversion. The original fixture has unconverted Qualified leads.
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 UV_DEFAULT_INDEX=https://pypi.org/simple UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY app ./app
RUN python - <<'PY'
from pathlib import Path
p=Path('app/main.py');s=p.read_text()
replacements={
    'if lead.get("status") == "Qualified":': 'if lead.get("isConverted") is True:',
    '        lead["status"] = "Qualified"\n        lead["modifiedDate"] = now': '        lead["status"] = "Qualified"\n        lead["isConverted"] = True\n        lead["modifiedDate"] = now',
}
for old,new in replacements.items():
    if s.count(old)!=1: raise RuntimeError('Unexpected upstream conversion implementation')
    s=s.replace(old,new)
p.write_text(s)
PY
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
