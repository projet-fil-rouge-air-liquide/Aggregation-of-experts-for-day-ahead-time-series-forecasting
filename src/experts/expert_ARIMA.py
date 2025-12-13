# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os


class ARIMAExpert:
    def __init__(self, csv_path, features=None, order=(2,0,2)):
        if features is None:
                features = [
                    "speed_longitudinale_100m",
                    "speed_latitudinale_100m",
                    "2m_temperature",
                    "mean_sea_level_pressure",
                    "sea_surface_temperature",
                    "surface_pressure",
                    "Wind_Norm",
                    "Wind_Norm_Cubes",
                    "wind_std_3h",
                    "wind_cv_3h",
                    "Wind_Norm_lag_1h",
                    "Wind_Norm_lag_24h",
                    "Hour_sin","Hour_cos","Weekday_sin","Weekday_cos",
                    "Month_sin","Month_cos",
                    "P_curve","Wind_mean_3h","Air_density",
                    "Wind_Dir_Meteo_sin","Wind_Dir_Meteo_cos"
                ]
        self.csv_path = csv_path
        self.features = features
        self.order = order
        
        # Chargement & preprocessing
        self.data = pd.read_csv(csv_path)
        self._detect_datetime()

        # Target
        self.y = self.data["Eolien_MW"]

        # Split data
        self._split()

        self.model = None
        self.model_future = None
        self.res = None


    # ------------------------------------------------------------
    #   Détection de la colonne date
    # ------------------------------------------------------------
    def _detect_datetime(self):
        datetime_cols = [c for c in self.data.columns if "date" in c.lower() or "time" in c.lower()]
        if datetime_cols:
            col = datetime_cols[0]
            self.data[col] = pd.to_datetime(self.data[col])
            self.data = self.data.set_index(col)
            self.datetime_col = col
        else:
            self.datetime_col = None


    # ------------------------------------------------------------
    #   Split temporel (60% train / 20% valid / 20% test)
    # ------------------------------------------------------------
    def _split(self):
        n = len(self.data)
        train_end = int(n * 0.6)
        valid_end = int(n * 0.8)

        self.X_train = self.data[self.features].iloc[:train_end]
        self.y_train = self.y.iloc[:train_end]

        self.X_valid = self.data[self.features].iloc[train_end:valid_end]
        self.y_valid = self.y.iloc[train_end:valid_end]

        self.X_test = self.data[self.features].iloc[valid_end:]
        self.y_test = self.y.iloc[valid_end:]


    # ------------------------------------------------------------
    #   Entraînement ARIMA / SARIMAX
    # ------------------------------------------------------------
    def fit(self):
        print("Training ARIMAX...")
        self.model = SARIMAX(
            self.y_train,
            exog=self.X_train,
            order=self.order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.res = self.model.fit(disp=False)
        print("Done.")


    # ------------------------------------------------------------
    #   Prédiction sur les données de test
    # ------------------------------------------------------------
    def predict_test(self, plot=True):
        y_pred = self.res.predict(
            start=len(self.y_train),
            end=len(self.y_train) + len(self.X_test) - 1,
            exog=self.X_test
        )

        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        print("ARIMAX RMSE test :", rmse)

        if plot:
            plt.figure(figsize=(9,7))
            plt.hexbin(
                self.y_test.values, y_pred.values,
                gridsize=60, cmap="viridis",
                mincnt=1, norm=LogNorm()
            )
            plt.colorbar(label="Densité (log)")

            maxv = max(self.y_test.max(), y_pred.max())
            minv = min(self.y_test.min(), y_pred.min())
            plt.plot([minv,maxv],[minv,maxv],'r--')

            plt.xlabel("Réel (MW)")
            plt.ylabel("Prédit (MW)")
            plt.title("Réel vs Prédit — ARIMAX")
            plt.tight_layout()
            plt.show()

        return y_pred, rmse


    # ------------------------------------------------------------
    #   Prédiction ARIMAX auto-régressive 24h
    # ------------------------------------------------------------
    def predict_future_24h(self, N_FUTURE=24, plot=True):

        future_target = self.y.iloc[-N_FUTURE:]
        future_meteo = self.data[self.features].iloc[-N_FUTURE:]

        # Train sur tout sauf les dernières 24h
        self.model_future = SARIMAX(
            self.y.iloc[:-N_FUTURE],
            exog=self.data[self.features].iloc[:-N_FUTURE],
            order=self.order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res_future = self.model_future.fit(disp=False)

        start = len(self.y) - N_FUTURE
        future_pred = res_future.predict(
            start=start,
            end=start + N_FUTURE - 1,
            exog=future_meteo
        )

        rmse_24h = np.sqrt(mean_squared_error(future_target, future_pred))
        print("RMSE auto-régression 24h :", rmse_24h)

        if plot:
            plt.figure(figsize=(10,5))
            plt.plot(future_target.values, label="Vrai", marker='o')
            plt.plot(future_pred.values, label="Prédit", marker='x')
            plt.title("Prédiction auto-régressive ARIMAX — 24h")
            plt.legend()
            plt.grid()
            plt.show()

        self.future_pred = future_pred
        self.future_target = future_target

        return future_pred, rmse_24h


    # ------------------------------------------------------------
    #   Sauvegarde des prévisions dans un CSV
    # ------------------------------------------------------------
    def save_predictions_csv(self, output_csv="predictions_24h.csv"):
        
        N = len(self.future_pred)

        if self.datetime_col:
            dates = self.data.index[-N:].astype(str)
        else:
            dates = pd.Series(range(N)).astype(str)

        save_df = pd.DataFrame({
            "date": dates,
            "pred_ARIMA": self.future_pred.values
        })

        if os.path.exists(output_csv):
            existing = pd.read_csv(output_csv)

            if "date" not in existing.columns:
                print("⚠ Le CSV n’a pas de colonne 'date'. Recréation...")
                save_df.to_csv(output_csv, index=False)
                return

            existing["date"] = existing["date"].astype(str)
            merged = existing.merge(save_df, on="date", how="outer", suffixes=("", "_new"))

            if "pred_ARIMA_new" in merged:
                merged["pred_ARIMA"] = merged["pred_ARIMA_new"].combine_first(merged["pred_ARIMA"])
                merged.drop(columns=["pred_ARIMA_new"], inplace=True)

            merged.to_csv(output_csv, index=False)
            print(f"✔ Mise à jour de {output_csv}")

        else:
            save_df.to_csv(output_csv, index=False)
            print(f"✔ Création de {output_csv}")


exp = ARIMAExpert("Data/Processed_data/data_engineering_belgique.csv")

exp.fit()
pred_test, rmse_test = exp.predict_test()

pred_24h, rmse_24h = exp.predict_future_24h()

exp.save_predictions_csv("predictions_24h.csv")
