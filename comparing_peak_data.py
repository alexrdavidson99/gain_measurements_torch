import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple


# Read the CSV file
plot_handles = []
plot_handles2 = []
for j in range(1, 5):
    print(j)

    data_df = pd.read_csv(f'all_data_peaks_min_{j}.csv', index_col=0)
    
    data_df.columns = data_df.columns.astype(str)
    
    if j == 3:
    #    MIN_FILES_PREFIX = [f"F{i}" for i in range(1, 4)]
    #    for pref in MIN_FILES_PREFIX:
    #        peak_data_F1 = data_df.loc[f'{pref}', '95.0':'98.5']
    #        
            #plt.plot(peak_data_F1.index.astype(float), peak_data_F1.values, label=pref)
        continue

    if j == 4:
        MIN_FILES_PREFIX = [f"F{i}" for i in range(2, 4)]
        for pref in MIN_FILES_PREFIX:
            peak_data_F1 = data_df.loc[f'{pref}', '84.3':'89.3']
            print(peak_data_F1)
            h, = plt.plot(peak_data_F1.index.astype(float)+10.65, peak_data_F1.values, label=pref)
            plot_handles.append(h)
        
    else:
        MIN_FILES_PREFIX = [f"F{i}" for i in range(2, 4)]
        for pref in MIN_FILES_PREFIX:
            peak_data_F1 = data_df.loc[f'{pref}', '97.0':'100.9']
            print(peak_data_F1)
            h, = plt.plot(peak_data_F1.index.astype(float)-1.2, peak_data_F1.values, label=pref)
            plot_handles2.append(h)


plt.legend([tuple(plot_handles), tuple(plot_handles2)], ["500V", "1300V"], handler_map={tuple: HandlerTuple(ndivide=None),  }, fontsize='24', handlelength=5)



plt.show()
# Plot the data

