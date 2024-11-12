# data_processing.py
import pandas as pd
from tkinter import messagebox

def load_data(file_path):
    try:
        # Wczytujemy dane, pomijając puste kolumny
        data = pd.read_csv(file_path, sep=';', usecols=lambda x: x != '')
        return data
    except Exception as e:
        messagebox.showerror("Błąd", f"Nie udało się załadować danych: {e}")
        return None

def validate_data(data):
    if data is not None:
        # Sprawdź, czy wszystkie wymagane kolumny są obecne
        required_columns = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']
        if not all(column in data.columns for column in required_columns):
            messagebox.showerror("Błąd", "Dane nie zawierają wszystkich wymaganych kolumn.")
            return False

        # Sprawdź, czy nie ma brakujących wartości
        if data[required_columns].isnull().values.any():
            messagebox.showerror("Błąd", "Dane zawierają brakujące wartości.")
            return False

        # Sprawdź, czy liczby są w odpowiednim zakresie
        main_numbers_valid = data[['L1', 'L2', 'L3', 'L4', 'L5']].applymap(lambda x: 1 <= x <= 50).all().all()
        euro_numbers_valid = data[['L6', 'L7']].applymap(lambda x: 1 <= x <= 12).all().all()
        if not main_numbers_valid or not euro_numbers_valid:
            messagebox.showerror("Błąd", "Dane zawierają liczby poza odpowiednim zakresem.")
            return False

        return True
    else:
        messagebox.showerror("Błąd", "Dane nie zostały załadowane.")
        return False

def feature_engineering(data):
    # Dodaj kolumnę z sumą głównych liczb
    data['Sum_Main'] = data[['L1', 'L2', 'L3', 'L4', 'L5']].sum(axis=1)
    # Dodaj kolumnę z średnią głównych liczb
    data['Mean_Main'] = data[['L1', 'L2', 'L3', 'L4', 'L5']].mean(axis=1)
    # Dodaj kolumnę z odchyleniem standardowym głównych liczb
    data['Std_Main'] = data[['L1', 'L2', 'L3', 'L4', 'L5']].std(axis=1)
    return data
