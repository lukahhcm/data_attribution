.PHONY: pdf open clean

LATEXMK ?= $(HOME)/Library/TinyTeX/bin/universal-darwin/latexmk
TEXBIN := $(dir $(LATEXMK))

pdf:
	mkdir -p build
	PATH="$(TEXBIN):$$PATH" "$(LATEXMK)" -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex

open: pdf
	open build/main.pdf

clean:
	find build -maxdepth 1 -type f \( -name '*.aux' -o -name '*.bbl' -o -name '*.blg' -o -name '*.log' -o -name '*.out' \) -delete
