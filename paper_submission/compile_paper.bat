@echo off
REM Compile LaTeX paper for submission (Windows version)

echo 🔧 Compiling Brain LDM Paper...
echo ================================

cd latex_source

REM Clean previous builds
del /Q *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls 2>nul

REM Compile with bibliography
echo 📄 First LaTeX pass...
pdflatex -interaction=nonstopmode main_paper.tex

echo 📚 Processing bibliography...
bibtex main_paper

echo 📄 Second LaTeX pass...
pdflatex -interaction=nonstopmode main_paper.tex

echo 📄 Final LaTeX pass...
pdflatex -interaction=nonstopmode main_paper.tex

REM Move final PDF
if exist "main_paper.pdf" (
    move main_paper.pdf ..\submission_files\Brain_LDM_Manuscript.pdf
    echo ✅ Success! PDF created: submission_files\Brain_LDM_Manuscript.pdf
) else (
    echo ❌ Error: PDF compilation failed
    pause
    exit /b 1
)

REM Clean up auxiliary files
del /Q *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls 2>nul

echo 🎉 Paper compilation complete!
pause
