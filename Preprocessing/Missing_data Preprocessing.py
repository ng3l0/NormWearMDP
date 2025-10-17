import json
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

def preprocess_stress_data(json_file):
    # Leggere i dati dal file JSON
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Verifica se il JSON è una lista
    if isinstance(data, list):
        # Combina tutte le serie temporali in un unico DataFrame
        combined_df = pd.DataFrame()
        for entry in data:
            date = entry.get("date")
            time_series = entry.get("time_series", [])
            df = pd.DataFrame(time_series, columns=["timestamp", "stress"])
            df['date'] = date
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        # Rinomina e gestisce il timestamp
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], unit='ms')
        combined_df.set_index('timestamp', inplace=True)
    else:
        raise ValueError("Formato JSON non supportato.")

    # Identifica i valori negativi
    stress_values = combined_df['stress']
    is_negative = stress_values < 0

    # Identifica gruppi consecutivi di negativi
    groups = (is_negative != is_negative.shift()).cumsum()
    for group in np.unique(groups[is_negative]):
        indices = groups[groups == group].index
        if len(indices) > 1:  # Più di un valore negativo consecutivo
            combined_df.loc[indices, 'stress'] = 0  # Sostituisci con 0
        else:
            combined_df.loc[indices, 'stress'] = np.nan  # Lascia per l'interpolazione

    # Interpolazione lineare per riempire i valori mancanti
    combined_df['stress'] = combined_df['stress'].interpolate(method='linear')

    # Visualizzare i risultati
    plt.figure(figsize=(12, 6))
    plt.plot(combined_df.index, combined_df['stress'], label="Stress (Interpolated)", color='blue')
    plt.title("Stress Time Series with Conditional Handling")
    plt.xlabel("Time")
    plt.ylabel("Stress Level")
    plt.legend()
    plt.show()

    return combined_df


# Esempio di utilizzo
json_file = "/Users/angeloguarnerio/Documents/Python/garmin_env/stress_time_series/stress_time_series_1gennaio.json"  # Sostituisci con il percorso del tuo file JSON
preprocessed_data = preprocess_stress_data(json_file)

# Salva il risultato preprocessato (opzionale)
preprocessed_data.to_csv("preprocessed_stress_data1_1.csv")
