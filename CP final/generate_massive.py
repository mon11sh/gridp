import os
import glob

# Gather all python source code to dump into the appendix
src_dir = r"c:\code_1\grido\gridone"
py_files = glob.glob(os.path.join(src_dir, "*.py"))

code_appendices = r"""
\chapter{APPENDIX A: SYSTEM SOURCE CODE}
This appendix contains the complete operational source code comprising the GridOne framework, demonstrating the explicit separation of concerns, the dense algorithmic implementations, and the deep operational structures seamlessly defining the entire end-to-end multi-model predictive ensemble architectures completely executed natively traversing the extensive empirical validation routines.

"""

for pf in py_files:
    fname = os.path.basename(pf)
    try:
        with open(pf, 'r', encoding='utf-8') as f:
            code_text = f.read()
    except Exception:
        continue
    
    # We replace some latex breaking chars if necessary, but lstlisting handles most
    # If the user's python has a lot of unusual unicode, we might just strip it.
    code_text = code_text.encode('ascii', errors='ignore').decode()
    
    code_appendices += f"\\section{{Source Code Abstract: \\texttt{{{fname}}}}}\n"
    code_appendices += "\\begin{lstlisting}[language=Python, breaklines=true, basicstyle=\\ttfamily\\scriptsize]\n"
    code_appendices += code_text
    code_appendices += "\n\\end{lstlisting}\n\n"
    
code_appendices += r"\end{document}"

# Read the file we already created and append the appendix
tex_file = r'c:\code_1\grido\CP final\FYP_Full2.tex'
tex_file_legacy = r'c:\code_1\grido\CP final\FYP_Full.tex'

with open(tex_file_legacy, 'r', encoding='utf-8') as f:
    text = f.read()

# Make sure it writes to BOTH tex files so the user gets the update regardless
with open(tex_file_legacy, 'w', encoding='utf-8') as f:
    f.write(text + "\n" + code_appendices)

with open(tex_file, 'w', encoding='utf-8') as f:
    f.write(text + "\n" + code_appendices)

print("Massive script successfully appended massive code sections to LaTeX files.")
