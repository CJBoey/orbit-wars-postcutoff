# Convenience targets. The analysis + page build need no third-party packages;
# `crawl` needs the Kaggle API (`pip install -r requirements.txt`).

ARCHIVE ?= data
TOP_N   ?= 100
CUTOFF  ?= 2026-06-24T00:00:00Z

.PHONY: page analyze crawl serve all

# Build the static page from the committed analysis snapshot (fast, zero deps).
page:
	python site/build_static.py --input $(ARCHIVE)/postcutoff_analysis.json --output docs/index.html

# Re-run the analysis from the bundled raw data, then rebuild the page.
analyze:
	python -m analysis.postcutoff_analysis --archive-dir $(ARCHIVE) --top-n $(TOP_N) \
		--bootstrap-reps 200 --min-games 50 --cutoff-utc $(CUTOFF)
	$(MAKE) page

# Fetch fresh data from Kaggle (requires credentials), then analyze + build.
crawl:
	python crawl/fetch_data.py --top-n $(TOP_N) --out-dir $(ARCHIVE)
	$(MAKE) analyze

# Preview the built page locally.
serve:
	python -m http.server -d docs

all: analyze
