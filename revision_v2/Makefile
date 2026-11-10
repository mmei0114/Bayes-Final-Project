.PHONY: all summarize check
PYTHON = conda run -n yellow python

all: report.pdf

check:
	$(PYTHON) src/validate.py

summarize:
	$(PYTHON) src/summarize.py

report.pdf: report.tex results/abstract_result.tex results/robustness_table.tex results/comparison_table.tex results/findings.tex results/mcmc_table.tex results/validation_text.tex figures/trajectories.pdf figures/diagnostics.pdf figures/mcmc_trace.pdf
	mkdir -p build
	xelatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex > build/compile-pass1.txt
	xelatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex > build/compile-pass2.txt
	cp build/report.pdf report.pdf
	pdftotext -layout report.pdf qa/report_text.txt
