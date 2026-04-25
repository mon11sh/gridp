import os
import glob
import re

TEX_FILE = r'c:\code_1\grido\CP final\FYP_Full2.tex'
TEX_FILE_ALT = r'c:\code_1\grido\CP final\FYP_Full.tex'
SRC_DIR = r'c:\code_1\grido\gridone'

def get_source_code_tex():
    py_files = [
        'app.py', 'config.py', 'data.py', 'training.py', 
        'forecasting.py', 'metrics.py', 'tirex_utils.py', 
        'ui_india.py', 'ui_us.py', 'visualization.py'
    ]
    tex_code = r"\chapter{Appendix B: Project Source Code}" + "\n"
    tex_code += r"This appendix contains the complete operational source code for the GridOne framework." + "\n\n"
    for fname in py_files:
        path = os.path.join(SRC_DIR, fname)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            content = content.encode('ascii', errors='ignore').decode()
            tex_code += r"\section{Source Code: \texttt{" + fname + r"}}" + "\n"
            tex_code += r"\begin{lstlisting}[language=Python, breaklines=true, basicstyle=\ttfamily\scriptsize, caption=" + fname + r"]" + "\n"
            tex_code += content + "\n" + r"\end{lstlisting}" + "\n\n"
    return tex_code

def get_states_table_tex():
    config_path = os.path.join(SRC_DIR, 'config.py')
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r'INDIA_STATES = \{(.*?)\}', content, re.DOTALL)
    if not match: return ''
    states_dict_str = match.group(1)
    lines = states_dict_str.strip().split('\n')
    tex_table = r"\chapter{Appendix A: Indian State Parameters}" + "\n"
    tex_table += r"\begin{table}[H]\centering\begin{tabularx}{\textwidth}{|X|c|c|c|}\hline \textbf{State/UT} & \textbf{Base Load (MU)} & \textbf{Noise Level} & \textbf{Region} \\ \hline" + "\n"
    for line in lines:
        m = re.search(r"'(.*?)':\s+\{'base':\s+(\d+),\s+'noise':\s+(\d+),\s+'region':\s+'(.*?)'\}", line)
        if m:
            state, base, noise, region = m.groups()
            tex_table += f"{state} & {base} & {noise} & {region.capitalize()} \\\\ \\hline\n"
    tex_table += r"\end{tabularx}\caption{Baseline configurations for synthetic India load data}\end{table}" + "\n"
    return tex_table

FRONT_MATTER = r"""\documentclass[12pt,a4paper,twoside,fleqn]{report}
\usepackage{graphicx}
\usepackage[a4paper,top=25mm, bottom=25mm, left=25mm, right=25mm]{geometry}
\usepackage{fancyhdr}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{algorithm2e}
\usepackage{color}
\usepackage{fancybox}
\usepackage{lastpage}
\usepackage{setspace}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
\usepackage{tocloft}
\usepackage[compact]{titlesec}
\usepackage{booktabs}
\usepackage{float}
\usepackage{tabularx}

\thisfancypage{
  \setlength{\fboxsep}{20pt}\doublebox}{}
\pagestyle{fancy}
\addtolength{\cftaftertoctitleskip}{-\baselineskip}
\makeatletter
\g@addto@macro{\normalsize}{
\setlength{\abovedisplayskip}{10pt}
\setlength{\abovedisplayshortskip}{10pt}
\setlength{\belowdisplayskip}{10pt}
\setlength{\belowdisplayshortskip}{10pt}}
\makeatother
\renewcommand{\baselinestretch}{1.5}
\pagenumbering{roman}

\fancypagestyle{combined}{
    \fancyhf{}
    \fancyhead[L]{\small GridOne: Framework for Power Grid Load Forecasting}
    \fancyfoot[C]{\protect\thepage}
    \renewcommand{\headrulewidth}{0.4pt}
    \renewcommand{\footrulewidth}{0pt}
}

\begin{document}
\pagestyle{empty}
\onehalfspacing

\begin{center}
    \huge\textbf{DAYANANDA SAGAR UNIVERSITY} \\
    \large\textbf{Devarakaggalahalli, Harohalli,
Kanakapura Road,
Bengaluru South Dt. – 562 112 }
\end{center}
\vspace{10mm}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.4\textwidth]{image/SCHOOL OF ENGINEERING_c.png}
\end{figure}
\vspace{10mm}
\begin{center}
    \Large\textbf{Bachelor of Technology} \\
    \normalsize\textbf{in} \\
    \large\textbf{COMPUTER SCIENCE AND ENGINEERING} \\
    \large\textbf{(ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING)}
\end{center}
\vspace{10mm}
\begin{center}
    \Large\textbf{A Project Report On} \\
    \vspace{5mm}
    \Large\textbf{ GridOne: An Integrated Multi-Region, Multi-Model Framework for Power Grid Load Forecasting}
\end{center}

\vfill
\begin{center}
    \small\textbf{By} \\
    \normalsize\textbf{PULKIT SINGH - ENG20AM0050} \\
    \normalsize\textbf{VISWAMBER PRASAD - ENG20AM0060} \\
    \normalsize\textbf{YASH M NARULE - ENG20AM0063} 
\end{center}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.11\textwidth]{image/Logo of AIML.jpg}
\end{figure}
\begin{center}
    \normalsize\textbf{Under the supervision of} \\
    \normalsize\textbf{Prof. UDAYA BHASKARA N} \\
    \normalsize\textbf{Assistant Professor}\\
    \normalsize\textbf{Computer Science \& Engineering (AI \& ML)}
\end{center}
\vfill
\begin{center}
    \normalsize\textbf{SCHOOL OF ENGINEERING}\\
    \normalsize\textbf{(2025 -- 2026)}
\end{center}

\newpage
\thisfancypage{
  \setlength{\fboxsep}{20pt}\doublebox}{}
\begin{center}
\textcolor{blue}{\LARGE\textbf{{DAYANANDA SAGAR UNIVERSITY}}} \\
\vspace{2mm}
\includegraphics[width=1\textwidth]{image/certificate.png} \\
\begin{center}
    \large\textbf{Department of Computer Science \& Engineering}\\
    \large\textbf{(ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING)}\\
    \small\textbf{Devarakaggalahalli, Harohalli,
Kanakapura Road,
Bengaluru South Dt. – 562 112\\}
    \small\textbf{Karnataka, India}
\end{center}
\vspace{5mm}
\Large{\underline{\textbf{CERTIFICATE}}} \\
  \end{center}
  \vspace{5mm}
\normalsize
This is to certify that the project entitled \textbf{“GridOne: An Integrated Multi-Region, Multi-Model Framework for Power Grid Load Forecasting”} is carried out by \textbf{PULKIT SINGH (ENG20AM0050), VISWAMBE RPRASAD (ENG20AM0060),YASH M NARULE (ENG20AM0063)}, 
bonafide students of Bachelor of Technology in Computer Science and Engineering at the School of Engineering, Dayananda Sagar University, Bangalore, in partial fulfillment for the award of a degree in Bachelor of Technology in Computer Science and Engineering, during the year 
\textbf{2025 \-- 2026}.\\

\vspace{15mm}
\noindent
\begin{minipage}[t]{0.31\textwidth}
\small
  \textbf{Prof. Udayabhaskara N}\\
  Assistant Professor\\
  Dept. of CSE (AI\&ML)\\
  School of Engineering\\
  Dayananda Sagar University\\
  \\
  Signature .........................\\
\end{minipage}%
\hfill
\begin{minipage}[t]{0.31\textwidth}
\small
  \textbf{Dr. Vinutha N \& Prof. Pradeep Kumar K}\\
  Project Co-ordinator\\
  Dept. of CSE (AI\&ML)\\
  School of Engineering\\
  Dayananda Sagar University\\
  \\
  Signature .........................\\
\end{minipage}%
\hfill
\begin{minipage}[t]{0.34\textwidth}
\small
  \textbf{Dr. Jayavrinda Vrindavanam}\\
  Professor \& Chairperson\\
  Dept. of CSE (AI\&ML)\\
  School of Engineering\\
  Dayananda Sagar University \\
  \\
  Signature .........................\\
\end{minipage}

\newpage
\pagestyle{plain}
\tableofcontents
\newpage
\listoffigures
\newpage
\listoftables
\newpage
\pagenumbering{arabic}
\pagestyle{combined}
"""

REPORT_CONTENT = r"""
\chapter{INTRODUCTION}
\section{Motivation and Background}
Energy load forecasting is the process of predicting future electricity demand based on historical data and various influencing factors such as weather, economic indicators, and seasonal patterns. As global energy systems transition from centralized, fossil-fuel-based generation to decentralized, renewable-heavy infrastructures, the stability of the grid has become increasingly dependent on accurate demand-side predictions. 

The complexity of the modern power grid necessitates a robust forecasting framework that can handle non-stationary demand patterns, multi-scale seasonality, and regional heterogeneity. For instance, the demand profile of a solar-rich state like California is significantly different from that of an industrial hub in Western India.

\section{Project Objectives}
The primary objective of \textbf{GridOne} is to provide an integrated platform for multi-region power grid load forecasting. The key goals include:
\begin{itemize}
    \item Unified interface for US and Indian power markets.
    \item Integration of seven diverse forecasting models ranging from classical SARIMA to Foundation models.
    \item Robust evaluation framework using specialized metrics like NSE and Diebold-Mariano.
    \item High-fidelity synthetic data generation for regions with data scarcity.
\end{itemize}

\chapter{SYSTEM ARCHITECTURE}
\section{Modular Design}
GridOne is built as a modular Python package to ensure scalability and maintainability. The core components include:
\begin{itemize}
    \item \textbf{Data Ingestion Layer}: Fetches real-time ISO data via \texttt{gridstatus} and generates synthetic data for Indian states.
    \item \textbf{Preprocessing & Featurization}: Implements cyclic temporal encoding, rolling statistics, and multi-step lag features.
    \item \textbf{Model Management}: Wrappers for XGBoost, LSTM, CNN-LSTM, SARIMA, TiRex, and DLinear.
    \item \textbf{Visualization Dashboard}: A Streamlit-based interactive interface for comparative analysis.
\end{itemize}

\chapter{METHODOLOGY}
\section{Forecasting Models}
We implement a diverse set of models to capture different signals in the load data.

\subsection{XGBoost (Extreme Gradient Boosting)}
XGBoost is used as a primary tabular model. It excels at capturing non-linear interactions between temporal features and historical lags.

\subsection{LSTM and CNN-LSTM}
For sequential modeling, we deploy LSTMs. In the Indian daily forecasting mode, we use a hybrid CNN-LSTM architecture where 1D convolutions extract local patterns before passing them to recurrent layers.

\subsection{DLinear and Foundation Models}
DLinear provides a decomposition-based approach (Trend + Seasonal), while TiRex (xLSTM-based) offers zero-shot forecasting capabilities as a foundation model.

\chapter{RESULTS AND ANALYSIS}
\section{Evaluation Framework}
Models are benchmarked using:
\begin{itemize}
    \item \textbf{Nash-Sutcliffe Efficiency (NSE)}: For hydrological-grade variance accuracy.
    \item \textbf{Diebold-Mariano Test}: For statistical significance in accuracy differentials.
    \item \textbf{MAPE and R2}: For relative and absolute accuracy metrics.
\end{itemize}

\chapter{CONCLUSION}
GridOne establishes a new standard for open-source grid load forecasting by providing a unified, multi-model sandbox. Future work will include weather API integration and graph-based spatial models.
"""

if __name__ == "__main__":
    with open(TEX_FILE, 'w', encoding='utf-8') as f:
        f.write(FRONT_MATTER)
        f.write(REPORT_CONTENT)
        f.write(get_states_table_tex())
        f.write(get_source_code_tex())
        f.write(r"\end{document}")
    
    with open(TEX_FILE_ALT, 'w', encoding='utf-8') as f:
        f.write(FRONT_MATTER)
        f.write(REPORT_CONTENT)
        f.write(get_states_table_tex())
        f.write(get_source_code_tex())
        f.write(r"\end{document}")

    print("Proper LaTeX report generated effectively.")
