.PHONY: pdf open clean

pdf:
	mkdir -p build
	tectonic --keep-logs --keep-intermediates --outdir build main.tex

open: pdf
	open build/main.pdf

clean:
	find build -maxdepth 1 -type f \( -name '*.aux' -o -name '*.bbl' -o -name '*.blg' -o -name '*.log' -o -name '*.out' \) -delete
