"""Local Flask web application for infectious disease forecasting."""

from __future__ import annotations

import base64
import sys
from io import BytesIO
from pathlib import Path

from flask import Flask, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from src.utils import ensure_directory  # noqa: E402

UPLOAD_FOLDER = PROJECT_ROOT / "outputs" / "uploads"
APP_OUTPUT_DIR = PROJECT_ROOT / "outputs"
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "txt", "tsv"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(ensure_directory(UPLOAD_FOLDER))

LAST_OUTPUTS: dict[str, Path] = {}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def figure_to_base64(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result = None
    image_base64 = None

    if request.method == "POST":
        file = request.files.get("case_file")
        if not file or file.filename == "":
            error = "Please upload a CSV, XLSX, or XLS file."
        elif not allowed_file(file.filename):
            error = "Unsupported file type. Upload CSV, TXT/TSV, XLSX, or XLS."
        else:
            filename = secure_filename(file.filename)
            upload_path = UPLOAD_FOLDER / filename
            file.save(upload_path)
            try:
                config = PipelineConfig(
                    input_path=str(upload_path),
                    date_col=request.form.get("date_col") or None,
                    value_col=request.form.get("value_col") or None,
                    disease_col=request.form.get("disease_col") or None,
                    geography_col=request.form.get("geography_col") or None,
                    disease_value=request.form.get("disease_value") or None,
                    geography_value=request.form.get("geography_value") or None,
                    frequency=request.form.get("frequency", "weekly"),
                    horizon=int(request.form.get("horizon", 12)),
                    seasonal_period=int(request.form.get("seasonal_period", 52)),
                    train_test_split=float(request.form.get("train_test_split", 0.2)),
                    aggregation=request.form.get("aggregation", "sum"),
                    fill_method=request.form.get("fill_method", "zero"),
                    smooth_outliers=bool(request.form.get("smooth_outliers")),
                    ensemble_method=request.form.get("ensemble_method", "weighted_mean"),
                    sarima_weight=float(request.form.get("sarima_weight", 1.0)),
                    fourier_weight=float(request.form.get("fourier_weight", 1.0)),
                    mlp_weight=float(request.form.get("mlp_weight", 1.0)),
                    n_harmonics=int(request.form.get("n_harmonics", 5)),
                    lookback=int(request.form.get("lookback", 12)),
                    run_sarima=bool(request.form.get("run_sarima")),
                    run_fourier=bool(request.form.get("run_fourier")),
                    run_mlp=bool(request.form.get("run_mlp")),
                    output_dir=str(APP_OUTPUT_DIR),
                )
                result = run_pipeline(config)
                image_base64 = figure_to_base64(result["figure"])
                LAST_OUTPUTS["csv"] = Path(result["csv_path"])
                LAST_OUTPUTS["xlsx"] = Path(result["xlsx_path"])
                LAST_OUTPUTS["png"] = Path(result["png_path"])
            except Exception as exc:
                error = str(exc)

    return render_template("index.html", error=error, result=result, image_base64=image_base64)


@app.route("/download/<kind>")
def download(kind: str):
    path = LAST_OUTPUTS.get(kind)
    if not path or not path.exists():
        return "Output not available. Run a forecast first.", 404
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
