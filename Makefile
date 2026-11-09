PY ?= .venv/bin/python

.PHONY: setup glove experiments benchmark figures summary report results test lint

setup:            ## create .venv with pinned dependencies
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

glove:            ## download GloVe 6B 100d (~860 MB zip, keeps only the 100d file)
	$(PY) -m agnews.embeddings

experiments: glove ## all training runs -> results/runs.csv (resumable)
	$(PY) -m agnews.experiments

benchmark:        ## controlled timing -> results/timing.csv (run on an idle machine)
	$(PY) -m agnews.benchmark

figures:          ## all figures -> results/figures/
	$(PY) -m agnews.figures
	$(PY) -m agnews.attention_viz

summary:          ## tables -> results/summary.md, README.md, reports/report.md
	$(PY) -m agnews.summary
	$(PY) -m agnews.summary --readme reports/report.md

report: summary   ## reports/report.pdf (needs pandoc + xelatex)
	cd reports && pandoc report.md -o report.pdf --pdf-engine=xelatex --citeproc --bibliography=references.bib --resource-path=..

results: experiments benchmark figures report

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests
