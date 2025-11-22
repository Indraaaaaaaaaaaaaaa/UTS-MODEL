import base64
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

APP_ROOT = Path(__file__).resolve().parent
DATA_PATH = APP_ROOT / "world_population.csv"
DEFAULT_COUNTRY = "Indonesia"

app = Flask(__name__)

df = pd.read_csv(DATA_PATH)
numeric_columns = [
    "1970 Population",
    "1980 Population",
    "1990 Population",
    "2000 Population",
    "2010 Population",
    "2015 Population",
    "2020 Population",
    "2022 Population",
]


def prepare_country_context(country_name: str) -> dict:
    if country_name not in df["Country/Territory"].values:
        country_name = DEFAULT_COUNTRY

    data = df[df["Country/Territory"] == country_name].iloc[0]
    years = np.array([1970, 1980, 1990, 2000, 2010, 2015, 2020, 2022])
    pop = data[numeric_columns].to_numpy(dtype=float)

    t = years - years[0]
    p0 = pop[0]
    r = float(np.polyfit(t, np.log(pop), 1)[0])

    future_years = np.arange(2022, 2072)
    t_future = future_years - years[0]
    pop_future_exp = p0 * np.exp(r * t_future)

    k = pop.max() * 3
    pop_log = []
    p = pop[-1]
    for _ in future_years:
        dp = r * p * (1 - p / k)
        p += dp
        pop_log.append(p)

    chart_data = build_chart(
        years,
        pop,
        future_years,
        pop_future_exp,
        pop_log,
        country_name,
    )

    forecast_points = [2030, 2040, 2050, 2060, 2071]
    forecast_table = []
    for year in forecast_points:
        if year < future_years[0]:
            continue
        idx = year - future_years[0]
        if idx >= len(future_years):
            continue
        forecast_table.append(
            {
                "year": year,
                "exp": f"{int(pop_future_exp[idx]):,}",
                "log": f"{int(pop_log[idx]):,}",
            }
        )

    final_year = forecast_table[-1]["year"] if forecast_table else future_years[-1]
    final_exp = forecast_table[-1]["exp"] if forecast_table else f"{int(pop_future_exp[-1]):,}"
    final_log = forecast_table[-1]["log"] if forecast_table else f"{int(pop_log[-1]):,}"

    forecast_summary = (
        f"Dalam horizon 50 tahun hingga {final_year}, model eksponensial memproyeksikan "
        f"populasi mencapai sekitar {final_exp} jiwa, sedangkan model logistic yang mempertimbangkan "
        f"kapasitas lingkungan memperkirakan sekitar {final_log} jiwa. Perbedaan ini membantu membaca "
        f"skenario optimistis versus realistis untuk setiap negara."
    )

    stats = {
        "Rank": int(data["Rank"]),
        "Capital": data["Capital"],
        "Continent": data["Continent"],
        "Area": f"{int(data['Area (km²)']):,} km²",
        "Density": f"{float(data['Density (per km²)']):.2f} people/km²",
        "Growth Rate": f"{float(data['Growth Rate']):.4f}",
        "World Share": f"{float(data['World Population Percentage']):.2f}%",
    }

    population_history = [
        (year, f"{int(value):,}") for year, value in zip(years[::-1], pop[::-1])
    ]

    return {
        "country": country_name,
        "stats": stats,
        "growth_rate": f"{r:.4f}",
        "chart": chart_data,
        "population_history": population_history,
        "forecast_table": forecast_table,
        "forecast_summary": forecast_summary,
    }


def build_chart(years, pop, future_years, pop_future_exp, pop_log, country):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, pop, "o-", label="Data Historis", color="#00d1ff")
    ax.plot(future_years, pop_future_exp, "--", label="Model Eksponensial", color="#ff6b6b")
    ax.plot(future_years, pop_log, "--", label="Model Logistic", color="#feca57")
    ax.set_title(f"Dynamic System Modelling: Populasi {country}")
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Jumlah Populasi")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.2)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
    country = request.form.get("country", DEFAULT_COUNTRY)
    context = prepare_country_context(country)
    countries = sorted(df["Country/Territory"].unique())
    return render_template("index.html", countries=countries, selected=country, **context)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

