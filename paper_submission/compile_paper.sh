#!/bin/bash
# Compile LaTeX paper for submission

echo "🔧 Compiling Brain LDM Paper..."
echo "================================"

cd latex_source/

# Clean previous builds
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls

# Compile with bibliography
echo "📄 First LaTeX pass..."
pdflatex -interaction=nonstopmode main_paper.tex

echo "📚 Processing bibliography..."
bibtex main_paper

echo "📄 Second LaTeX pass..."
pdflatex -interaction=nonstopmode main_paper.tex

echo "📄 Final LaTeX pass..."
pdflatex -interaction=nonstopmode main_paper.tex

# Move final PDF
if [ -f "main_paper.pdf" ]; then
    mv main_paper.pdf ../submission_files/Brain_LDM_Manuscript.pdf
    echo "✅ Success! PDF created: submission_files/Brain_LDM_Manuscript.pdf"
else
    echo "❌ Error: PDF compilation failed"
    exit 1
fi

# Clean up auxiliary files
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls

echo "🎉 Paper compilation complete!"
