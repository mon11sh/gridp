import os
import glob
import pandas as pd
from datetime import datetime

base_dir = r"c:\code_1\grido\gridone\data\Cleaned_Data2"
files = glob.glob(os.path.join(base_dir, "**\\*.xlsx"), recursive=True)

all_data = []
for f in files:
    filename = os.path.basename(f)
    # filename is like 01.04.23_NLDC_PSP.xlsx
    date_str = filename.split('_')[0]
    try:
        dt = datetime.strptime(date_str, "%d.%m.%y")
    except ValueError:
        try:
            dt = datetime.strptime(date_str, "%d.%m.%Y")
        except:
            continue

    try:
        df = pd.read_excel(f)
        df['Date'] = dt
        all_data.append(df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

if all_data:
    master_df = pd.concat(all_data, ignore_index=True)
    master_df.to_csv(r"c:\code_1\grido\gridone\data\india_master_data.csv", index=False)
    print("Compiled to india_master_data.csv")
else:
    print("No data compiled")
