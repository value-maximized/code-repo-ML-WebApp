import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load data from the Excel file
file_path = 'Income Statement.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')
#Remove & replace empty columns with 'NaN'
data = data.dropna(axis=1, how='all')
data = data.fillna('')
print(data)
#extract rows (ex., first row is index=0) from data frame into an numpy array
revenue = data.iloc[0]
cogs = data.iloc[5]
gross_profit = data.iloc[10]
ypoints_revenue =np.array(revenue)
ypoints_cogs =np.array(cogs)
ypoints_gross_profit =np.array(gross_profit)
#slicing array to drop the first value (a string called 'Revenue,COGS and GrossMargins')
ypoints_revenue = ypoints_revenue[1:]
ypoints_cogs = ypoints_cogs[1:]
ypoints_gross_profit = ypoints_gross_profit[1:]
year = 2024
xpoints = [year, year+1, year+2, year+3, year+4] 
# print(f"{ypoints}")
# print(f"{xpoints}")
#set up the plot
plt.xlabel("Year")
plt.title("5 year financial forecast - Gross Profits ($)")
plt.plot(xpoints, ypoints_gross_profit, label ='Gross Profit', color="#000000", marker ='o')
plt.bar(xpoints, ypoints_revenue, label ='Revenue', color ="#A8C8E8", width=0.6)
plt.bar(xpoints, ypoints_cogs, label = 'COGS', color ="#000080", width=0.4)
plt.grid()
plt.show()
