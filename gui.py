# gui.py
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import pandas as pd
import numpy as np
import os
import random
import data_processing as dp
import models
import visualization as vz

class EurojackpotPredictionApp:
    MIN_SAMPLES_REQUIRED = 50  # Minimalna liczba próbek wymagana do trenowania modeli

    def __init__(self, root):
        self.root = root
        self.root.title("Eurojackpot Prediction App")
        self.xgb_model = None
        self.lstm_model = None
        self.classification_model = None
        self.scaler_X = None
        self.scaler_y = None
        self.scaler_X_lstm = None
        self.scaler_y_lstm = None
        self.data = None
        self.file_path = None

        # Ustawienia losowości
        self.seed = 42
        np.random.seed(self.seed)
        random.seed(self.seed)

        # GUI elements
        self.create_widgets()

        self.data_range = None

    def create_widgets(self):
        # Menu
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Załaduj Dane", command=self.load_data)
        file_menu.add_command(label="Aktualizuj Dane", command=self.update_data)
        file_menu.add_separator()
        file_menu.add_command(label="Wyjdź", command=self.root.quit)
        menubar.add_cascade(label="Plik", menu=file_menu)

        model_menu = tk.Menu(menubar, tearoff=0)
        model_menu.add_command(label="Zapisz Model XGBoost", command=self.save_xgb_model)
        model_menu.add_command(label="Wczytaj Model XGBoost", command=self.load_xgb_model)
        model_menu.add_command(label="Zapisz Model LSTM", command=self.save_lstm_model)
        model_menu.add_command(label="Wczytaj Model LSTM", command=self.load_lstm_model)
        model_menu.add_command(label="Zapisz Model Klasyfikacyjny", command=self.save_classification_model)
        model_menu.add_command(label="Wczytaj Model Klasyfikacyjny", command=self.load_classification_model)
        menubar.add_cascade(label="Model", menu=model_menu)

        self.root.config(menu=menubar)

        # Zakładki
        tab_control = ttk.Notebook(self.root)

        self.tab_data = ttk.Frame(tab_control)
        self.tab_models = ttk.Frame(tab_control)
        self.tab_predict = ttk.Frame(tab_control)
        self.tab_visualize = ttk.Frame(tab_control)

        tab_control.add(self.tab_data, text='Dane')
        tab_control.add(self.tab_models, text='Modele')
        tab_control.add(self.tab_predict, text='Predykcja')
        tab_control.add(self.tab_visualize, text='Wizualizacja')

        tab_control.pack(expand=1, fill='both')

        # Zakładka Dane
        self.load_data_btn = tk.Button(self.tab_data, text="Załaduj Dane", command=self.load_data)
        self.load_data_btn.pack(pady=10)

        self.update_data_btn = tk.Button(self.tab_data, text="Aktualizuj Dane", command=self.update_data, state=tk.NORMAL)
        self.update_data_btn.pack(pady=10)

        self.select_data_range_btn = tk.Button(self.tab_data, text="Wybierz Zakres Danych", command=self.select_data_range, state=tk.DISABLED)
        self.select_data_range_btn.pack(pady=10)

        self.search_data_btn = tk.Button(self.tab_data, text="Znajdź Optymalny Zakres Danych", command=self.search_data_range, state=tk.DISABLED)
        self.search_data_btn.pack(pady=10)

        # Zakładka Modele
        self.seed_var = tk.IntVar(value=self.seed)
        tk.Label(self.tab_models, text="Seed (ziarno losowości):").pack()
        tk.Entry(self.tab_models, textvariable=self.seed_var).pack()

        # Hiperparametry XGBoost
        self.xgb_params_frame = tk.LabelFrame(self.tab_models, text="Hiperparametry XGBoost")
        self.xgb_params_frame.pack(pady=10, fill="x")

        self.n_estimators_var = tk.IntVar(value=100)
        tk.Label(self.xgb_params_frame, text="n_estimators:").grid(row=0, column=0, sticky='e')
        tk.Entry(self.xgb_params_frame, textvariable=self.n_estimators_var).grid(row=0, column=1)

        self.learning_rate_var = tk.DoubleVar(value=0.1)
        tk.Label(self.xgb_params_frame, text="learning_rate:").grid(row=1, column=0, sticky='e')
        tk.Entry(self.xgb_params_frame, textvariable=self.learning_rate_var).grid(row=1, column=1)

        self.max_depth_var = tk.IntVar(value=5)
        tk.Label(self.xgb_params_frame, text="max_depth:").grid(row=2, column=0, sticky='e')
        tk.Entry(self.xgb_params_frame, textvariable=self.max_depth_var).grid(row=2, column=1)

        self.train_xgb_model_btn = tk.Button(self.tab_models, text="Trenuj Model XGBoost", command=self.train_xgb_model, state=tk.DISABLED)
        self.train_xgb_model_btn.pack(pady=10)

        # Hiperparametry LSTM
        self.lstm_params_frame = tk.LabelFrame(self.tab_models, text="Hiperparametry LSTM")
        self.lstm_params_frame.pack(pady=10, fill="x")

        self.lstm_epochs_var = tk.IntVar(value=50)
        tk.Label(self.lstm_params_frame, text="Epoki:").grid(row=0, column=0, sticky='e')
        tk.Entry(self.lstm_params_frame, textvariable=self.lstm_epochs_var).grid(row=0, column=1)

        self.lstm_batch_size_var = tk.IntVar(value=16)
        tk.Label(self.lstm_params_frame, text="Batch size:").grid(row=1, column=0, sticky='e')
        tk.Entry(self.lstm_params_frame, textvariable=self.lstm_batch_size_var).grid(row=1, column=1)

        self.lstm_dropout_var = tk.DoubleVar(value=0.2)
        tk.Label(self.lstm_params_frame, text="Dropout:").grid(row=2, column=0, sticky='e')
        tk.Entry(self.lstm_params_frame, textvariable=self.lstm_dropout_var).grid(row=2, column=1)

        self.train_lstm_model_btn = tk.Button(self.tab_models, text="Trenuj Model LSTM", command=self.train_lstm_model, state=tk.DISABLED)
        self.train_lstm_model_btn.pack(pady=10)

        # Model klasyfikacyjny
        self.train_classification_model_btn = tk.Button(self.tab_models, text="Trenuj Model Klasyfikacyjny", command=self.train_classification_model, state=tk.DISABLED)
        self.train_classification_model_btn.pack(pady=10)

        # Zakładka Predykcja
        self.priorities_frame = tk.LabelFrame(self.tab_predict, text="Priorytety Liczb")
        self.priorities_frame.pack(pady=10, fill="x")

        tk.Label(self.priorities_frame, text="Podaj liczby do priorytetyzacji (oddzielone przecinkami):").pack()
        self.prioritized_numbers_var = tk.StringVar()
        tk.Entry(self.priorities_frame, textvariable=self.prioritized_numbers_var).pack()

        self.num_predictions_var = tk.IntVar(value=1)
        tk.Label(self.tab_predict, text="Liczba przyszłych losowań do przewidzenia:").pack()
        tk.Entry(self.tab_predict, textvariable=self.num_predictions_var).pack()

        self.generate_numbers_btn = tk.Button(self.tab_predict, text="Wygeneruj Liczby", command=self.generate_numbers, state=tk.DISABLED)
        self.generate_numbers_btn.pack(pady=10)

        self.output_label = tk.Label(self.tab_predict, text="Predykcja Liczb: ")
        self.output_label.pack(pady=20)

        # Zakładka Wizualizacja
        self.show_chart_btn = tk.Button(self.tab_visualize, text="Pokaż Wykres Częstotliwości", command=self.show_frequency_chart, state=tk.DISABLED)
        self.show_chart_btn.pack(pady=10)

        # Pasek postępu
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack(pady=10)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.file_path = file_path
            self.data = dp.load_data(self.file_path)
            if dp.validate_data(self.data):
                self.data = dp.feature_engineering(self.data)
                messagebox.showinfo("Informacja", "Pomyślnie załadowano i przygotowano dane.")
                self.select_data_range_btn.config(state=tk.NORMAL)
                self.search_data_btn.config(state=tk.NORMAL)
                self.show_chart_btn.config(state=tk.NORMAL)
            else:
                self.data = None  # Reset danych w przypadku błędu

    def update_data(self):
        if self.file_path and os.path.exists(self.file_path):
            self.data = dp.load_data(self.file_path)
            if dp.validate_data(self.data):
                self.data = dp.feature_engineering(self.data)
                messagebox.showinfo("Informacja", "Dane zostały zaktualizowane.")
            else:
                self.data = None
        else:
            messagebox.showerror("Błąd", "Plik danych nie jest dostępny.")

    def select_data_range(self):
        if self.data is None:
            messagebox.showerror("Błąd", "Najpierw załaduj dane.")
            return

        total_samples = len(self.data)
        min_samples_required = self.MIN_SAMPLES_REQUIRED

        start = simpledialog.askinteger("Zakres Danych", f"Podaj początkowy indeks danych do szkolenia (0 - {total_samples - 1}):")
        if start is None:
            return
        end = simpledialog.askinteger("Zakres Danych", f"Podaj końcowy indeks danych do szkolenia ({start} - {total_samples - 1}):")
        if end is None:
            return

        if start < 0 or end >= total_samples or start > end:
            messagebox.showerror("Błąd", "Nieprawidłowy zakres danych. Upewnij się, że start ≤ end i indeksy są w odpowiednim zakresie.")
            return

        num_samples = end - start + 1
        if num_samples < min_samples_required:
            messagebox.showerror("Błąd", f"Wybrany zakres danych zawiera {num_samples} próbek. Wymagane jest co najmniej {min_samples_required} próbek.")
            return

        self.data_range = (start, end + 1)  # Dodajemy 1 do end, ponieważ w iloc ostatni indeks jest wyłączony
        messagebox.showinfo("Informacja", f"Wybrano zakres danych: od {start} do {end}.")
        self.train_xgb_model_btn.config(state=tk.NORMAL)
        self.train_lstm_model_btn.config(state=tk.NORMAL)
        self.train_classification_model_btn.config(state=tk.NORMAL)

    def search_data_range(self):
        if self.data is None:
            messagebox.showerror("Błąd", "Najpierw załaduj dane.")
            return

        target_numbers = simpledialog.askstring("Wyszukiwanie Zakresu Danych", "Podaj 5 głównych liczb i 2 Euronumery (oddzielone przecinkami, np. 1,2,3,4,5,1,2):")
        if target_numbers:
            try:
                target_numbers = [int(num.strip()) for num in target_numbers.split(',')]
                if len(target_numbers) != 7:
                    raise ValueError("Należy podać dokładnie 7 liczb (5 głównych i 2 Euronumery).")

                best_range = None
                max_matches = 0

                # Iterate over all possible ranges in the data
                for start in range(len(self.data) - self.MIN_SAMPLES_REQUIRED + 1):
                    end = start + self.MIN_SAMPLES_REQUIRED
                    if end > len(self.data):
                        break  # Nie mamy wystarczającej liczby danych od tego punktu

                    current_numbers = np.hstack((
                        self.data[['L1', 'L2', 'L3', 'L4', 'L5']].iloc[start:end].values.flatten(),
                        self.data[['L6', 'L7']].iloc[start:end].values.flatten()
                    ))
                    matches = len(set(target_numbers) & set(current_numbers))
                    if matches > max_matches and matches >= 3:
                        max_matches = matches
                        best_range = (start, end)

                if best_range:
                    self.data_range = best_range
                    messagebox.showinfo("Informacja", f"Znaleziono optymalny zakres danych: od {best_range[0]} do {best_range[1]-1}. Liczba trafień: {max_matches}")
                    self.train_xgb_model_btn.config(state=tk.NORMAL)
                    self.train_lstm_model_btn.config(state=tk.NORMAL)
                    self.train_classification_model_btn.config(state=tk.NORMAL)
                else:
                    messagebox.showinfo("Informacja", f"Nie znaleziono zakresu danych spełniającego kryteria z minimalną liczbą {self.MIN_SAMPLES_REQUIRED} próbek i co najmniej 3 trafieniami.")
            except ValueError as e:
                messagebox.showerror("Błąd", str(e))
            except Exception as e:
                messagebox.showerror("Błąd", f"Wystąpił błąd: {e}")

    def train_xgb_model(self):
        if self.data is None or self.data_range is None:
            messagebox.showerror("Błąd", "Najpierw załaduj dane i wybierz zakres.")
            return

        self.progress.start()
        try:
            # Ustawienie ziarna losowości
            self.seed = self.seed_var.get()

            # Preprocessing data
            start, end = self.data_range
            X, y = models.prepare_data_regression(self.data.iloc[start:end])
            params = {
                'n_estimators': self.n_estimators_var.get(),
                'learning_rate': self.learning_rate_var.get(),
                'max_depth': self.max_depth_var.get()
            }

            self.xgb_model, self.scaler_X, self.scaler_y, mse, average_mse = models.train_xgb_model(X, y, self.seed, params)

            messagebox.showinfo("Informacja", f"Model XGBoost został wytrenowany.\nMSE: {mse:.4f}\nŚrednie MSE w kroswalidacji: {average_mse:.4f}")
            self.generate_numbers_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił błąd podczas trenowania modelu XGBoost: {e}")
        finally:
            self.progress.stop()

    def save_xgb_model(self):
        if self.xgb_model is None:
            messagebox.showerror("Błąd", "Nie ma wytrenowanego modelu XGBoost do zapisania.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
        if file_path:
            models.save_xgb_model(self.xgb_model, self.scaler_X, self.scaler_y, file_path)
            messagebox.showinfo("Informacja", "Model XGBoost został zapisany.")

    def load_xgb_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if file_path:
            self.xgb_model, self.scaler_X, self.scaler_y = models.load_xgb_model(file_path)
            if self.xgb_model:
                messagebox.showinfo("Informacja", "Model XGBoost został wczytany.")
                self.generate_numbers_btn.config(state=tk.NORMAL)
            else:
                messagebox.showerror("Błąd", "Nie udało się wczytać modelu XGBoost.")

    def train_lstm_model(self):
        if self.data is None or self.data_range is None:
            messagebox.showerror("Błąd", "Najpierw załaduj dane i wybierz zakres.")
            return

        self.progress.start()
        try:
            # Ustawienie ziarna losowości
            self.seed = self.seed_var.get()

            # Preprocessing data
            start, end = self.data_range
            X, y = models.prepare_data_regression(self.data.iloc[start:end], lstm=True)
            params = {
                'epochs': self.lstm_epochs_var.get(),
                'batch_size': self.lstm_batch_size_var.get(),
                'dropout_rate': self.lstm_dropout_var.get()
            }

            self.lstm_model, self.scaler_X_lstm, self.scaler_y_lstm = models.train_lstm_model(X, y, self.seed, params)

            messagebox.showinfo("Informacja", "Model LSTM został wytrenowany.")
            self.generate_numbers_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił błąd podczas trenowania modelu LSTM: {e}")
        finally:
            self.progress.stop()

    def save_lstm_model(self):
        if self.lstm_model is None:
            messagebox.showerror("Błąd", "Nie ma wytrenowanego modelu LSTM do zapisania.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("HDF5 files", "*.h5")])
        if file_path:
            models.save_lstm_model(self.lstm_model, self.scaler_X_lstm, self.scaler_y_lstm, file_path)
            messagebox.showinfo("Informacja", "Model LSTM został zapisany.")

    def load_lstm_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
        if file_path:
            self.lstm_model, self.scaler_X_lstm, self.scaler_y_lstm = models.load_lstm_model(file_path)
            if self.lstm_model:
                messagebox.showinfo("Informacja", "Model LSTM został wczytany.")
                self.generate_numbers_btn.config(state=tk.NORMAL)
            else:
                messagebox.showerror("Błąd", "Nie udało się wczytać modelu LSTM.")

    def train_classification_model(self):
        if self.data is None or self.data_range is None:
            messagebox.showerror("Błąd", "Najpierw załaduj dane i wybierz zakres.")
            return

        self.progress.start()
        try:
            # Ustawienie ziarna losowości
            self.seed = self.seed_var.get()

            start, end = self.data_range
            X, y = models.prepare_data_classification(self.data.iloc[start:end])

            self.classification_model, test_hamming, f1 = models.train_classification_model(X, y, self.seed)

            messagebox.showinfo("Informacja", f"Model klasyfikacyjny został wytrenowany.\nHamming Loss na zbiorze testowym: {test_hamming:.4f}\nF1-score: {f1:.4f}")
            self.generate_numbers_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił błąd podczas trenowania modelu klasyfikacyjnego: {e}")
        finally:
            self.progress.stop()

    def save_classification_model(self):
        if self.classification_model is None:
            messagebox.showerror("Błąd", "Nie ma wytrenowanego modelu klasyfikacyjnego do zapisania.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
        if file_path:
            models.save_classification_model(self.classification_model, file_path)
            messagebox.showinfo("Informacja", "Model klasyfikacyjny został zapisany.")

    def load_classification_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if file_path:
            self.classification_model = models.load_classification_model(file_path)
            if self.classification_model:
                messagebox.showinfo("Informacja", "Model klasyfikacyjny został wczytany.")
                self.generate_numbers_btn.config(state=tk.NORMAL)
            else:
                messagebox.showerror("Błąd", "Nie udało się wczytać modelu klasyfikacyjnego.")

    def generate_numbers(self):
        if self.xgb_model is None and self.lstm_model is None and self.classification_model is None:
            messagebox.showerror("Błąd", "Najpierw wytrenuj przynajmniej jeden model.")
            return

        num_predictions = self.num_predictions_var.get()

        # Pobranie priorytetowych liczb od użytkownika
        prioritized_numbers_input = self.prioritized_numbers_var.get()
        if prioritized_numbers_input:
            try:
                prioritized_numbers = [int(num.strip()) for num in prioritized_numbers_input.split(',') if num.strip()]
                prioritized_numbers_main = [num for num in prioritized_numbers if 1 <= num <= 50]
                prioritized_numbers_euro = [num for num in prioritized_numbers if 1 <= num <= 12]
            except ValueError:
                messagebox.showerror("Błąd", "Nieprawidłowe liczby priorytetowe.")
                return
        else:
            prioritized_numbers_main = []
            prioritized_numbers_euro = []

        outputs = []

        # Ustawienie ziarna losowości
        import random
        if self.seed_var.get() is None:
            self.seed = random.randint(1, 1000000)
        else:
            self.seed = self.seed_var.get()
        np.random.seed(self.seed)
        random.seed(self.seed)

        for _ in range(num_predictions):
            xgb_output = "XGBoost: Model nie jest wytrenowany."
            lstm_output = "LSTM: Model nie jest wytrenowany."
            classification_output = "Klasyfikacja: Model nie jest wytrenowany."

            if self.xgb_model is not None:
                try:
                    recent_numbers = self.data[['L1', 'L2', 'L3', 'L4', 'L5']].iloc[-1].values
                    recent_euros = self.data[['L6', 'L7']].iloc[-1].values
                    recent_features = self.data[['Sum_Main', 'Mean_Main', 'Std_Main']].iloc[-1].values
                    recent_data_xgb = np.hstack((recent_numbers, recent_euros, recent_features)).reshape(1, -1)
                    recent_data_xgb_scaled = self.scaler_X.transform(recent_data_xgb)
                    xgb_prediction = self.xgb_model.predict(recent_data_xgb_scaled)
                    xgb_prediction_rescaled = self.scaler_y.inverse_transform(xgb_prediction)

                    # Dodanie losowego szumu
                    noise = np.random.normal(0, 1, xgb_prediction_rescaled.shape)
                    xgb_prediction_rescaled += noise

                    xgb_predicted_numbers = [max(1, min(50, int(round(num)))) for num in xgb_prediction_rescaled[0][:5]]
                    xgb_predicted_euros = [max(1, min(12, int(round(num)))) for num in xgb_prediction_rescaled[0][5:]]

                    xgb_predicted_numbers.extend(prioritized_numbers_main)
                    xgb_predicted_euros.extend(prioritized_numbers_euro)
                    xgb_output = f"XGBoost: Główne {sorted(set(xgb_predicted_numbers))[:5]}, Euronumery {sorted(set(xgb_predicted_euros))[:2]}"
                except Exception as e:
                    xgb_output = f"XGBoost: Błąd podczas generowania liczb ({e})"

            if self.lstm_model is not None:
                try:
                    window_size = 5
                    recent_data = self.data[['L1', 'L2', 'L3', 'L4', 'L5',
                                             'L6', 'L7', 'Sum_Main', 'Mean_Main', 'Std_Main']].iloc[-window_size:].values
                    X_input_main = recent_data[:, :5]
                    X_input_euro = recent_data[:, 5:7]
                    X_input_features = recent_data[:, 7:]
                    X_input = np.hstack((X_input_main, X_input_euro, X_input_features))
                    X_input = X_input.reshape(1, window_size, -1)
                    X_input_reshaped = X_input.reshape(-1, X_input.shape[2])
                    X_input_scaled = self.scaler_X_lstm.transform(X_input_reshaped)
                    X_input_scaled = X_input_scaled.reshape(1, window_size, -1)
                    lstm_prediction = self.lstm_model.predict(X_input_scaled)
                    lstm_prediction_rescaled = self.scaler_y_lstm.inverse_transform(lstm_prediction)
                    # Dodanie losowego szumu
                    noise = np.random.normal(0, 1, lstm_prediction_rescaled.shape)
                    lstm_prediction_rescaled += noise
                    # Ekstrakcja przewidywanych liczb
                    lstm_predicted_numbers = [max(1, min(50, int(round(num))))
                                              for num in lstm_prediction_rescaled[0][:5]]
                    lstm_predicted_euros = [max(1, min(12, int(round(num))))
                                            for num in lstm_prediction_rescaled[0][5:]]
                    lstm_predicted_numbers.extend(prioritized_numbers_main)
                    lstm_predicted_euros.extend(prioritized_numbers_euro)
                    lstm_output = f"LSTM: Główne {sorted(set(lstm_predicted_numbers))[:5]}, " \
                                  f"Euronumery {sorted(set(lstm_predicted_euros))[:2]}"
                except Exception as e:
                    lstm_output = f"LSTM: Błąd podczas generowania liczb ({e})"

            if self.classification_model is not None:
                try:
                    recent_numbers = self.data[['L1', 'L2', 'L3', 'L4', 'L5']].iloc[-5:].values.flatten()
                    recent_euros = self.data[['L6', 'L7']].iloc[-5:].values.flatten()
                    X_input = np.hstack((recent_numbers, recent_euros)).reshape(1, -1)
                    y_pred_proba = self.classification_model.predict_proba(X_input)

                    # Główne liczby
                    proba_main = np.array([proba[0][1] if len(proba[0]) > 1 else proba[0][0] for proba in y_pred_proba[:50]])
                    proba_main /= proba_main.sum()
                    predicted_numbers = np.random.choice(np.arange(1, 51), size=5, replace=False, p=proba_main)

                    # Euronumery
                    proba_euro = np.array([proba[0][1] if len(proba[0]) > 1 else proba[0][0] for proba in y_pred_proba[50:]])
                    proba_euro /= proba_euro.sum()
                    predicted_euros = np.random.choice(np.arange(1, 13), size=2, replace=False, p=proba_euro)

                    predicted_numbers = list(predicted_numbers) + prioritized_numbers_main
                    predicted_euros = list(predicted_euros) + prioritized_numbers_euro

                    classification_output = f"Klasyfikacja: Główne {sorted(set(predicted_numbers))[:5]}, " \
                                            f"Euronumery {sorted(set(predicted_euros))[:2]}"
                except Exception as e:
                    classification_output = f"Klasyfikacja: Błąd podczas generowania liczb ({e})"

            outputs.append(f"{xgb_output}\n{lstm_output}\n{classification_output}")

        self.output_label.config(text=f"Predykcja Liczb (pamiętaj o losowości wyników):\n" + "\n\n".join(outputs))

    def show_frequency_chart(self):
        if self.data is None:
            messagebox.showerror("Błąd", "Najpierw załaduj dane.")
            return
        vz.plot_frequency(self.data, self.tab_visualize)

if __name__ == "__main__":
    root = tk.Tk()
    app = EurojackpotPredictionApp(root)
    root.mainloop()
