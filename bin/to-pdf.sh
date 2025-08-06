export PANDOC_PATH="/C/Users/Khor Kean Teng/Downloads/pandoc-3.7.0.2-windows-x86_64/pandoc-3.7.0.2"
export PATH="$PATH:$PANDOC_PATH"

cd "docs"

# ask user for part number
read -p "Enter part number (e.g., 1, 2, 3, etc.): " number

# Generate PDF with LaTeX for A4 paper size
pandoc "part-${number}" -o "part-${number}.pdf" \
  --pdf-engine=pdflatex \
  -V geometry:a4paper \
  -V geometry:margin=0.5in \
  -V mainfont="Arial" \
  --include-in-header=watermark.tex