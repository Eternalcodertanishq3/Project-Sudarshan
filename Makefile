.PHONY: run demo test cache-tle clean install

run:
	./scripts/demo_start.sh

demo:
	@echo "Opening demo in browser: http://localhost:5173"
	xdg-open http://localhost:5173 2>/dev/null || open http://localhost:5173 2>/dev/null || true

test:
	cd backend && python -m pytest tests/ -v --tb=short

cache-tle:
	python scripts/fetch_tle.py

install:
	pip install -r requirements.txt
	cd frontend && npm install --legacy-peer-deps

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

drdo-brief:
	@echo "Opening DRDO brief..."
	cat docs/DRDO_BRIEF.md
