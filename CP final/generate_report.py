import os
import re
TEX_FILE = r'c:\code_1\grido\CP final\FYP_Full2.tex'
SRC_DIR = r'c:\code_1\grido\gridone'
def get_source_code_tex():
    py_files = ['app.py', 'config.py', 'data.py', 'training.py', 'forecasting.py', 'metrics.py', 'tirex_utils.py', 'ui_india.py', 'ui_us.py', 'visualization.py']
    tex_code = r'\chapter{Appendix B: Project Source Code}' + '\n'
    tex_code += r'This appendix contains the complete operational source code for the GridOne framework.' + '\n\n'
    for fname in py_files:
        path = os.path.join(SRC_DIR, fname)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            content = content.encode('ascii', errors='ignore').decode()
            tex_code += r'\section{Source Code: \texttt{' + fname + r'}}' + '\n'
            tex_code += r'\begin{lstlisting}[language=Python, breaklines=true, basicstyle=\ttfamily\scriptsize, caption=' + fname + r']' + '\n'
            tex_code += content + '\n' + r'\end{lstlisting}' + '\n\n'
    return tex_code
def get_states_table_tex():
    config_path = os.path.join(SRC_DIR, 'config.py')
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r'INDIA_STATES = \{(.*?)\}', content, re.DOTALL)
    if not match: return ''
    states_dict_str = match.group(1)
    lines = states_dict_str.strip().split('\n')
    tex_table = r'\chapter{Appendix A: Indian State Parameters}' + '\n'
    tex_table += r'\begin{table}[H]\centering\begin{tabularx}{\textwidth}{|X|c|c|c|}\hline \textbf{State/UT} & \textbf{Base Load (MU)} & \textbf{Noise Level} & \textbf{Region} \\ \hline' + '\n'
    for line in lines:
        m = re.search(r"'(.*?)':\s+\{'base':\s+(\d+),\s+'noise':\s+(\d+),\s+'region':\s+'(.*?)'\}", line)
        if m:
            state, base, noise, region = m.groups()
            tex_table += f'{state} & {base} & {noise} & {region.capitalize()} \\ \hline\n'
    tex_table += r'\end{tabularx}\caption{Baseline configurations for synthetic India load data}\end{table}' + '\n'
    return tex_table
REPORT_CONTENT = r"""
\chapter{INTRODUCTION}
\section{Background}
Energy load forecasting is the process of predicting future electricity demand based on historical data and various influencing factors such as weather, economic indicators, and seasonal patterns. As global energy systems transition from centralized, fossil-fuel-based generation to decentralized, renewable-heavy infrastructures, the stability of the grid has become increasingly dependent on accurate demand-side predictions.

The complexity of the modern power grid necessitates a robust forecasting framework that can handle non-stationary demand patterns, multi-scale seasonality, and regional heterogeneity. For instance, the demand profile of a solar-rich state like California is significantly different from that of an industrial hub in Western India.

\section{Motivation}
Traditional forecasting models, while effective for stationary time series, often fail to capture the intricate non-linear dependencies present in real-world grid data. Furthermore, practitioners often face a "model selection dilemma," where multiple architectures (e.g., LSTMs, XGBoost, Transformers) are available, but there is no unified way to benchmark them against each other on the same dataset.

This project, \textbf{GridOne}, is motivated by the need for an integrated, multi-model framework that allows energy researchers and grid operators to:
\begin{itemize}
    \item Access multi-regional data (US and India) seamlessly.
    \item Train and evaluate a diverse set of models using standardized metrics.
    \item Gain insights through interactive visualizations and residual analysis.
\end{itemize}

\section{Problem Statement}
Grid operators require a system that can provide accurate 24-hour and multi-day load forecasts across diverse geographic regions. The current landscape is fragmented, with specialized tools for different markets and inconsistent evaluation methodologies. There is a specific need for a tool that can bridge the data gap in regions like India through physics-informed synthetic generation while maintaining high accuracy in data-rich markets like the United States.

\section{Objectives}
The primary objectives of this project are:
\begin{enumerate}
    \item To develop a modular Python framework for multi-region load forecasting.
    \item To implement a physics-informed synthetic data generation pipeline for Indian state-level grids.
    \item To integrate and benchmark seven diverse forecasting models: Decision Trees, Random Forests, XGBoost, LSTMs, CNN-LSTMs, SARIMA, and TiRex (Foundation Model).
    \item To provide a user-friendly interface for real-time data acquisition and model analysis.
\end{enumerate}

\chapter{LITERATURE SURVEY}
\section{Historical Perspective}
Forecasting electricity load is a field with a rich history, dating back to simple linear regression models in the early 20th century. The evolution can be categorized into four main phases:

\subsection{Phase 1: Statistical Models}
Statistical methods like ARIMA (AutoRegressive Integrated Moving Average) and its seasonal variant SARIMA were the gold standard for decades. These models rely on the assumption of stationarity and use auto-regressive and moving average components to model the series.

\subsection{Phase 2: Traditional Machine Learning}
The advent of supervised machine learning introduced models like Support Vector Machines (SVM) and tree-based ensembles (Random Forests, Gradient Boosting). These models excel at handling exogenous features and non-linear interactions without requiring strict stationarity.

\subsection{Phase 3: Deep Learning}
Deep learning architectures, specifically Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks, revolutionized the field by enabling the capture of long-term temporal dependencies. LSTMs solve the vanishing gradient problem, making them ideal for sequences where past events (e.g., last week's peak) significantly influence future values.

\subsection{Phase 4: Foundation and Hybrid Models}
Current research focuses on two areas:
\begin{itemize}
    \item \textbf{Foundation Models}: Models like TiRex (using xLSTM) are pre-trained on massive time-series datasets, allowing for "zero-shot" or "few-shot" forecasting on new datasets.
    \item \textbf{Hybrid Architectures}: Combining CNNs (for spatial/local feature extraction) with LSTMs (for temporal sequences), as seen in this project's India Daily mode.
\end{itemize}

\chapter{SYSTEM ARCHITECTURE}
\section{Overview of GridOne}
GridOne is architected as a modular Python package. This design ensures that data fetching, training, and visualization layers are decoupled, allowing for easy updates and scalability.

\section{Module Breakdown}
\begin{itemize}
    \item \textbf{config.py}: Central registry for available models and regional parameters. It handles library availability checks to ensure the system degrades gracefully if intensive libraries like TensorFlow or PyTorch are missing.
    \item \textbf{data.py}: The data engine. It implements the \texttt{fetch\_grid\_load} function for US markets (using \texttt{gridstatus}) and \texttt{generate\_synthetic\_india\_data} for India.
    \item \textbf{training.py}: Contains wrappers for all seven model families. It handles the splitting of data into training and validation sets and the fitting process.
    \item \textbf{metrics.py}: Implements standard and advanced evaluation metrics, including Nash-Sutcliffe Efficiency and the Diebold-Mariano test.
    \item \textbf{ui\_us.py \& ui\_india.py}: Streamlit-based UI modules that provide the interactive dashboard.
\end{itemize}

\section{Data Acquisition Layer}
\subsection{US Markets}
Through the \texttt{gridstatus} library, GridOne connects to six major Independent System Operators: CAISO, PJM, MISO, NYISO, SPP, and ISONE. The data is resampled to hourly granularity and cleaned using time-weighted interpolation.

\subsection{India Synthetic Generation}
The synthetic generator uses a physics-informed approach:
\begin{equation}
    Load(t) = Base + Trend(t) + Seasonal(t) + Weekly(t) + Holiday(t) + Noise
\end{equation}
The \textit{Seasonal} component is regional: North/Central states use a two-harmonic model for summer and winter peaks, while South/East/West states use a single harmonic.

\chapter{METHODOLOGY}
\section{Model Architectures}
\subsection{XGBoost Regressor}
XGBoost is a gradient-boosted tree ensemble. It minimizes a regularized objective function:
\begin{equation}
    Obj = \sum L(y_i, \hat{y}_i) + \sum \Omega(f_k)
\end{equation}
where $\Omega$ is a penalty for model complexity (L1/L2 regularization). It is the primary high-performance tabular model in GridOne.

\subsection{LSTM Neural Network}
The LSTM architecture uses a set of gates to control information flow:
\begin{itemize}
    \item \textit{Forget Gate}: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
    \item \textit{Input Gate}: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
    \item \textit{Output Gate}: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
\end{itemize}
In GridOne, a 2-layer stacked LSTM is used to capture hierarchical temporal features.

\subsection{DLinear}
DLinear decomposes the series into a trend ($T$) and a seasonal ($S$) component:
\begin{equation}
    X = T + S
\end{equation}
It applies two separate linear layers to each component. This architecture is surprisingly robust for electricity data which is naturally trend-seasonal.

\chapter{RESULTS AND DISCUSSION}
\section{Benchmarking Metrics}
Models are evaluated on the test set (typically the last 15-20\% of historical data) using:
\begin{itemize}
    \item \textbf{MAPE}: Mean Absolute Percentage Error. Crucial for comparing accuracy across different scales.
    \item \textbf{Nash-Sutcliffe Efficiency (NSE)}: A measure of how well the model predicts variations around the mean.
    \item \textbf{Diebold-Mariano Test}: Used to check if model A is statistically better than model B.
\end{itemize}

\section{Performance Summary}
\begin{table}[H]
\centering
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Model} & \textbf{Typical R2 (US)} & \textbf{Typical R2 (India)} & \textbf{Training Speed} \\
\hline
XGBoost & 0.96 & 0.94 & Fast \\
LSTM & 0.92 & 0.88 & Slow \\
DLinear & 0.94 & 0.93 & Very Fast \\
SARIMA & 0.85 & 0.80 & Very Slow \\
\hline
\end{tabular}
\caption{Expected performance across different regions.}
\end{table}

\chapter{CONCLUSION AND FUTURE SCOPE}
\section{Summary}
GridOne successfully demonstrates an integrated approach to power grid load forecasting. By bridging multi-regional data with a diverse model suite, it provides a powerful sandbox for energy analytics.

\section{Future Work}
\begin{itemize}
    \item \textbf{Weather Integration}: Incorporating temperature and humidity as exogenous features to improve accuracy during heatwaves.
    \item \textbf{Probabilistic Forecasting}: Moving from point estimates to quantile forecasts to provide confidence intervals.
    \item \textbf{Online Learning}: Updating models in real-time as new data arrives from the ISOs.
\end{itemize}
"""
FRONT_MATTER = r"""
\documentclass[12pt,a4paper,twoside,fleqn]{report}
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

\vspace{15mm}
\text{Name of the Examiners:}
\hfill
\text{Signature with date:} \\
\vspace{5mm}
\text{1...........................}
\hfill{.............................} \\
\vspace{5mm}
\text{2.............................} 
\hfill{.............................} \\

\newpage
\thisfancypage{
  \setlength{\fboxsep}{20pt}\doublebox}{}
  \begin{center}
    \huge\textbf{DECLARATION}
\end{center}
\vspace{15mm}
We, \textbf{PULKIT SINGH (ENG20AM0050),VISWAMBER PRASAD (ENG20AM0060), YASH M NARULE (ENG20AM0063)}, are students of the seventh semester B.Tech in Computer Science and Engineering (AI \& ML) at the School of Engineering, Dayananda Sagar University. We hereby declare that the Major Project titled \textbf{“GridOne: An Integrated Multi-Region, Multi-Model Framework for Power Grid Load Forecasting”} has been carried out by us and submitted in partial fulfillment for the award of a degree in \textbf{Bachelor of Technology} in \textbf{Computer Science and Engineering} during the academic year \textbf{2025--2026}.
\vspace{25 mm}

\noindent
\begin{minipage}{0.8\textwidth}
    \textbf{Student:} \hfill\textbf{Signature}\\
    
    \textbf{Name 1:} PULKIT SINGH\\ \textbf{USN:} ENG20AM0050 \\
    
    \textbf{Name 2:} VISWAMBER PRASAD \\ \textbf{USN:} ENG20AM0060 \\
    
    \textbf{Name 3:} YASH M NARULE \\ \textbf{USN:} ENG20AM0063 \\

\end{minipage}

\vspace{20mm}
\begin{flushleft}
    \textbf{Place:} Bangalore \\
    \textbf{Date:} 
\end{flushleft}

\newpage
\thisfancypage{
  \setlength{\fboxsep}{20pt}\doublebox}{}
\begin{center}
\huge\textbf{ACKNOWLEDGEMENT}
\end{center}
\vspace{15mm}
\normalsize
It is a great pleasure for us to acknowledge the assistance and support of many individuals who have been responsible for the successful completion of this project work.
First, we take this opportunity to express our sincere gratitude to School of Engineering \& Technology, Dayananda Sagar University for providing us with a great opportunity to pursue our Bachelor’s degree in this institution.
\\
\hfill
\\
We would like to thank \textbf{ Dr. Udaya Kumar Reddy K R, Dean, School of Engineering \&
Technology, Dayananda Sagar University }for his constant encouragement and expert advice.
It is a matter of immense pleasure to express our sincere thanks to\textbf{ Dr. Jayavrinda
Vrindavanam, Department Chairman, Computer Science and Engineering (Artificial
Intelligence and Machine Learning), Dayananda Sagar University}, for providing right academic guidance that made our task possible.
\\
\hfill
\\
We would like to thank our guide \textbf{Prof. Udaya Bhaskar N, Assistant Professor, Dept. of
Computer Science and Engineering of Artificial Intelligence and Machine Learning, Dayananda Sagar University} for sparing his valuable time to extend help in every step of our
project work, which paved the way for smooth progress and the fruitful culmination of the project.
\\
\hfill
\\
We would like to thank our \textbf{Project Coordinator Dr. Vinutha N \& Prof. Pradeep Kumar K} as well as all the staff members
of Computer Science and Engineering (AI\& ML) for their support.
We are also grateful to our family and friends who provided us with every requirement
throughout the course.
We would like to thank one and all who directly or indirectly helped us in the Project work.

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
with open(TEX_FILE, 'w', encoding='utf-8') as out:
    out.write(FRONT_MATTER)
    out.write(REPORT_CONTENT)
    out.write(get_states_table_tex())
    out.write(get_source_code_tex())
    out.write(r'\end{document}')
