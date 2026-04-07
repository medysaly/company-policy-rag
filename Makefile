.PHONY: install lint test serve ui demo evaluate ingest

install:
	pip install -e ".[dev]"

lint:
	ruff check app/ tests/
	ruff format --check app/ tests/

format:
	ruff format app/ tests/

test:
	pytest -v

serve:
	uvicorn app.api.main:app --reload --port 8000

ui:
	streamlit run ui/streamlit_app.py --server.port 8501

ingest:
	python scripts/ingest_sample.py

evaluate:
	python scripts/run_evaluation.py
