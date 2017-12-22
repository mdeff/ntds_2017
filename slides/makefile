TEX = $(wildcard *.tex)
PDF = $(TEX:.tex=.pdf)
BIB = $(wildcard *.bib)

all: $(PDF)

%.pdf: %.tex $(BIB)
	latexmk -pdf $<

clean:
	rm -f *.{aux,bbl,blg,fdb_latexmk,fls,log,out,lof,dvi}
	rm -f *.{bcf,run.xml}
	rm -f *.{toc,snm,nav}
	rm -f *.gz*

cleanall: clean
	rm -f $(PDF)

.PHONY: all clean cleanall
