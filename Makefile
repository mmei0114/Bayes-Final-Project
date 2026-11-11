.PHONY: all report summarize check verify convergence
PYTHON = conda run -n yellow python
export MPLCONFIGDIR = $(CURDIR)/build/matplotlib
export XDG_CACHE_HOME = $(CURDIR)/build/cache

all: report.pdf

report: report.pdf

check:
	$(PYTHON) src/validate.py

summarize:
	$(PYTHON) src/summarize.py

convergence:
	$(PYTHON) src/summarize_convergence.py

verify:
	shasum -a 256 -c MANIFEST.sha256
	cd data && shasum -a 256 -c SHA256SUMS
	cd data/convergence && shasum -a 256 -c SHA256SUMS

report.pdf: report.tex results/abstract_result.tex results/robustness_table.tex results/comparison_table.tex results/findings.tex results/mcmc_table.tex results/validation_text.tex figures/trajectories.pdf figures/diagnostics.pdf figures/mcmc_trace.pdf results/convergence_table.tex results/convergence_text.tex figures/parameter_convergence.pdf figures/likelihood_convergence.pdf
	mkdir -p build
	xelatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex > build/compile-pass1.txt
	xelatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex > build/compile-pass2.txt
	cp build/report.pdf report.pdf
	pdftotext -layout report.pdf qa/report_text.txt
