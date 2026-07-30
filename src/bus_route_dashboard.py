import json
import os
import threading
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from ui_theme import (
    apply_theme, AnimatedBanner, Card, ScrollableFrame,
    BG_APP, BG_CARD, BG_INPUT, FG_TEXT, FG_MUTED, ACCENT,
)
from spark_backend import SparkBackend, PROJECT_ROOT, OUTPUT_DIR

# --------------------------------------------------------------------------
MODEL_COMPARISON_CSV = OUTPUT_DIR / "model_comparison.csv"
FEATURE_IMPORTANCE_JSON = OUTPUT_DIR / "feature_importance.json"
CV_RESULTS_JSON = OUTPUT_DIR / "cv_results.json"
FEATURE_METADATA_JSON = OUTPUT_DIR / "feature_metadata.json"
FEATURE_CSV = OUTPUT_DIR / "final_feature_dataset.csv"
EVAL_DIR = OUTPUT_DIR / "eval_predictions"

plt.rcParams.update({
    "figure.facecolor": BG_CARD,
    "axes.facecolor": BG_CARD,
    "axes.edgecolor": FG_MUTED,
    "axes.labelcolor": FG_TEXT,
    "text.color": FG_TEXT,
    "xtick.color": FG_MUTED,
    "ytick.color": FG_MUTED,
    "grid.color": "#2a3d63",
    "font.size": 9,
    "legend.facecolor": BG_CARD,
    "legend.edgecolor": "none",
    "legend.labelcolor": FG_TEXT,
})
BAR_COLORS = ["#2f8fef", "#5fd0c0", "#f2b134", "#e0607e", "#8f7ee8"]


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def safe_model_filename(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "")


class DashboardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bus Route Performance Analytics -- Dashboard")
        self.geometry("1250x820")
        apply_theme(self)

        self.banner = AnimatedBanner(
            self, "Bus Route Performance Analytics"
        )
        self.banner.pack(fill="x")

        self.status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.status_var, style="Muted.TLabel", padding=(14, 3)).pack(fill="x")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        # Shared data, loaded once
        self.comparison_df = pd.read_csv(MODEL_COMPARISON_CSV) if MODEL_COMPARISON_CSV.exists() else None
        self.feature_importance = load_json(FEATURE_IMPORTANCE_JSON)
        self.cv_results = load_json(CV_RESULTS_JSON)
        self.feature_metadata = load_json(FEATURE_METADATA_JSON)
        self.label_names = (self.feature_importance or {}).get("label_names", [])
        self.model_names = self.comparison_df["Model"].tolist() if self.comparison_df is not None else []
        self.eval_predictions = self._load_eval_predictions()

        self._scrollable_frames = []
        tabs = [
            ("Overview", self._build_overview_tab),
            ("Model Comparison", self._build_comparison_tab),
            ("Confusion Matrices", self._build_confusion_tab),
            ("ROC Curves", self._build_roc_tab),
            ("Feature Importance", self._build_importance_tab),
            ("Cross-Validation", self._build_cv_tab),
        ]
        for label, builder in tabs:
            scroll = ScrollableFrame(self.notebook)
            self.notebook.add(scroll, text=label)
            self._scrollable_frames.append(scroll)
            try:
                builder(scroll.body)
            except Exception as e:
                ttk.Label(scroll.body, text=f"Could not build this tab:\n{e}",
                          style="Muted.TLabel").pack(pady=20, padx=20)

        self.predict_tab = ScrollableFrame(self.notebook)
        self.notebook.add(self.predict_tab, text="Predict Route")
        self._scrollable_frames.append(self.predict_tab)
        self._build_predict_placeholder(self.predict_tab.body)

        self.backend = None
        self._spark_starting = False
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # Bind wheel scrolling to whichever tab starts visible
        self._scrollable_frames[0].bind_wheel(self)

    def _on_close(self):
        if self.backend is not None:
            self.backend.stop()
        self.destroy()

    def _set_status(self, message: str):
        self.status_var.set(message)
        self.update_idletasks()

    def _on_tab_changed(self, event):
        selected_path = self.notebook.select()
        for scroll in self._scrollable_frames:
            if str(scroll) == selected_path:
                scroll.bind_wheel(self)
                break
        if selected_path == str(self.predict_tab) and self.backend is None and not self._spark_starting:
            self._start_spark_backend()

    # -------------------------------------------------------- Data loading --
    def _load_eval_predictions(self):
        """Reads Notebook per-model prediction parquet files with plain
        pandas. Powers the Confusion Matrix and ROC tabs."""
        predictions = {}
        if not EVAL_DIR.exists() or not self.model_names:
            return predictions
        for name in self.model_names:
            path = EVAL_DIR / f"{safe_model_filename(name)}.parquet"
            if path.exists():
                try:
                    predictions[name] = pd.read_parquet(path)
                except Exception as e:
                    print(f"Could not read {path}: {e}")
        return predictions

    # ---------------------------------------------------------- Overview --
    def _build_overview_tab(self, frame):
        metadata = self.feature_metadata
        row_count, class_counts = None, None
        if FEATURE_CSV.exists():
            try:
                target_col = (metadata or {}).get("target_column", "route_popularity")
                df = pd.read_csv(FEATURE_CSV, usecols=lambda c: c == target_col)
                row_count = len(df)
                class_counts = df[target_col].value_counts().to_dict() if target_col in df.columns else None
            except Exception:
                pass

        metrics_row = ttk.Frame(frame, style="TFrame")
        metrics_row.pack(fill="x", pady=(4, 14), padx=4)

        def metric_card(parent, label, value):
            card = Card(parent)
            card.pack(side="left", expand=True, fill="both", padx=6)
            ttk.Label(card, text=str(value), style="Metric.TLabel").pack(anchor="w")
            ttk.Label(card, text=label, style="CardMuted.TLabel").pack(anchor="w")

        metric_card(metrics_row, "Total records", f"{row_count:,}" if row_count else "N/A")
        metric_card(metrics_row, "Features used", len((metadata or {}).get("candidate_feature_columns", [])) or "N/A")
        best_row = self.comparison_df.iloc[self.comparison_df["f1"].idxmax()] if self.comparison_df is not None else None
        metric_card(metrics_row, "Best model", best_row["Model"] if best_row is not None else "N/A")
        metric_card(metrics_row, "Best F1 score", f"{best_row['f1']:.4f}" if best_row is not None else "N/A")

        bottom = ttk.Frame(frame, style="TFrame")
        bottom.pack(fill="both", expand=True, padx=4)

        chart_card = Card(bottom, title="Target class distribution (route_popularity)")
        chart_card.pack(side="left", fill="both", expand=True, padx=(0, 6))
        if class_counts:
            fig = Figure(figsize=(5, 3.0), dpi=100)
            ax = fig.add_subplot(111)
            classes = list(class_counts.keys())
            values = list(class_counts.values())
            ax.bar(classes, values, color=BAR_COLORS[: len(classes)])
            ax.set_ylabel("Count")
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=chart_card)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            ttk.Label(chart_card, text="final_feature_dataset.csv not found.", style="CardMuted.TLabel").pack()

        info_card = Card(bottom, title="Feature columns used (20)")
        info_card.pack(side="left", fill="both", expand=True, padx=(6, 0))
        text = tk.Text(info_card, height=14, bg=BG_APP, fg=FG_TEXT, relief="flat",
                        font=("Consolas", 9), wrap="word")
        text.pack(fill="both", expand=True)
        if metadata:
            for c in metadata.get("candidate_feature_columns", []):
                text.insert("end", f"\u2022 {c}\n")
            leakage = metadata.get("leakage_columns", [])
            if leakage:
                text.insert("end", f"\nExcluded as leakage: {', '.join(leakage)}\n")
        else:
            text.insert("end", "feature_metadata.json not found.")
        text.config(state="disabled")

    # ------------------------------------------------------ Model Comparison --
    def _build_comparison_tab(self, frame):
        if self.comparison_df is None:
            ttk.Label(frame, text=f"model_comparison.csv not found at:\n{MODEL_COMPARISON_CSV}").pack(pady=20)
            return
        df = self.comparison_df

        # One consistent color per model, reused across all four charts so
        # a reader can visually match models between them at a glance.
        model_colors = {m: BAR_COLORS[i % len(BAR_COLORS)] for i, m in enumerate(df["Model"])}

        metrics = [
            ("accuracy", "Accuracy"),
            ("weightedPrecision", "Precision (weighted)"),
            ("weightedRecall", "Recall (weighted)"),
            ("f1", "F1-score (weighted)"),
        ]

        grid = ttk.Frame(frame, style="TFrame")
        grid.pack(fill="both", expand=True, padx=4, pady=(4, 10))
        row1 = ttk.Frame(grid, style="TFrame")
        row2 = ttk.Frame(grid, style="TFrame")
        row1.pack(fill="both", expand=True, pady=(0, 8))
        row2.pack(fill="both", expand=True)

        rows = [row1, row1, row2, row2]
        for (column, title), row in zip(metrics, rows):
            if column not in df.columns:
                continue
            self._build_single_metric_chart(row, df, model_colors, column, title)

        time_card = Card(frame, title="Training time (seconds, incl. cross-validation)")
        time_card.pack(fill="both", expand=True, padx=4, pady=(0, 10))
        fig2 = Figure(figsize=(8.5, 2.3), dpi=100)
        ax2 = fig2.add_subplot(111)
        colors = [model_colors[m] for m in df["Model"]]
        bars2 = ax2.bar(df["Model"], df["Training Time (s)"], color=colors, edgecolor=BG_APP, linewidth=1)
        ax2.bar_label(bars2, fmt="%.1f s", padding=3, fontsize=8, color=FG_TEXT)
        ax2.set_ylabel("Seconds")
        ax2.tick_params(axis="x", rotation=12)
        ax2.grid(axis="y", alpha=0.25, linestyle="--")
        ax2.grid(axis="x", visible=False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        fig2.tight_layout()
        canvas2 = FigureCanvasTkAgg(fig2, master=time_card)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True)

        table_card = Card(frame, title="Raw results")
        table_card.pack(fill="both", expand=True, padx=4)
        tree = ttk.Treeview(table_card, columns=list(df.columns), show="headings", height=len(df) + 1)
        for c in df.columns:
            tree.heading(c, text=c)
            tree.column(c, width=115, anchor="center")
        for _, row in df.iterrows():
            values = [f"{v:.4f}" if isinstance(v, float) else v for v in row]
            tree.insert("", "end", values=values)
        tree.pack(fill="both", expand=True)

    def _build_single_metric_chart(self, parent_row, df, model_colors, column, title):
        card = Card(parent_row, title=title)
        card.pack(side="left", fill="both", expand=True, padx=6)

        fig = Figure(figsize=(4.6, 3.4), dpi=100)
        ax = fig.add_subplot(111)
        colors = [model_colors[m] for m in df["Model"]]
        bars = ax.bar(df["Model"], df[column], color=colors, edgecolor=BG_APP, linewidth=1.2, width=0.55)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8.5, color=FG_TEXT, fontweight="bold")

        ax.set_ylim(0, 1.12)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["Model"], rotation=15, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=model_colors[m]) for m in df["Model"]]
        ax.legend(legend_handles, df["Model"], loc="upper right", fontsize=6.5,
                  framealpha=0.85, facecolor=BG_CARD, edgecolor="none", labelcolor=FG_TEXT)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=card)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------ Confusion Matrices --
    def _build_confusion_tab(self, frame):
        if not self.eval_predictions or not self.label_names:
            ttk.Label(
                frame,
                text=f"No prediction files found under:\n{EVAL_DIR}\n\n"
                     "(Needs Notebook 05's eval_predictions/*.parquet files.)",
                style="Muted.TLabel",
            ).pack(pady=20, padx=20)
            return

        n_classes = len(self.label_names)
        card = Card(frame, title="Confusion matrices (from eval_predictions/*.parquet)")
        card.pack(fill="both", expand=True, padx=4, pady=4)

        n_models = len(self.eval_predictions)
        fig = Figure(figsize=(4.0 * n_models, 3.4), dpi=100)
        for i, (name, df) in enumerate(self.eval_predictions.items(), start=1):
            ax = fig.add_subplot(1, n_models, i)
            matrix = np.zeros((n_classes, n_classes), dtype=int)
            counts = df.groupby(["label", "prediction"]).size()
            for (l, p), c in counts.items():
                if int(l) < n_classes and int(p) < n_classes:
                    matrix[int(l), int(p)] = c
            im = ax.imshow(matrix, cmap="Blues")
            ax.set_title(name, fontsize=9)
            ax.set_xticks(range(n_classes)); ax.set_xticklabels(self.label_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(n_classes)); ax.set_yticklabels(self.label_names, fontsize=8)
            ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8)
            for r in range(n_classes):
                for c_ in range(n_classes):
                    ax.text(c_, r, matrix[r, c_], ha="center", va="center", fontsize=8,
                            color="white" if matrix[r, c_] > matrix.max() / 2 else "black")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=card)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------ ROC Curves --
    def _build_roc_tab(self, frame):
        if not self.eval_predictions or not self.label_names:
            ttk.Label(
                frame,
                text=f"No prediction files found under:\n{EVAL_DIR}",
                style="Muted.TLabel",
            ).pack(pady=20, padx=20)
            return

        try:
            from sklearn.metrics import roc_curve, auc
        except ImportError:
            ttk.Label(frame, text="scikit-learn not installed -- run:\npip install scikit-learn",
                       style="Muted.TLabel").pack(pady=20)
            return

        card = Card(frame, title="ROC curves per class (one-vs-rest), from eval_predictions/*.parquet")
        card.pack(fill="both", expand=True, padx=4, pady=4)

        n_models = len(self.eval_predictions)
        fig = Figure(figsize=(4.2 * n_models, 3.4), dpi=100)
        for i, (name, df) in enumerate(self.eval_predictions.items(), start=1):
            ax = fig.add_subplot(1, n_models, i)
            has_prob = "probability_array" in df.columns
            for class_idx, class_name in enumerate(self.label_names):
                y_true = (df["label"] == class_idx).astype(int)
                if has_prob:
                    y_score = df["probability_array"].apply(lambda arr: arr[class_idx])
                else:
                    y_score = (df["prediction"] == class_idx).astype(int)
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.3f})", linewidth=1.5)
            ax.plot([0, 1], [0, 1], "--", color=FG_MUTED, linewidth=0.8)
            suffix = "" if has_prob else "\n(approximate -- no probability column)"
            ax.set_title(f"{name}{suffix}", fontsize=9)
            ax.set_xlabel("False Positive Rate", fontsize=8)
            ax.set_ylabel("True Positive Rate", fontsize=8)
            ax.legend(fontsize=7)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=card)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------ Feature Importance --
    def _build_importance_tab(self, frame):
        data = self.feature_importance
        if not data:
            ttk.Label(frame, text=f"feature_importance.json not found at:\n{FEATURE_IMPORTANCE_JSON}").pack(pady=20)
            return

        assembler_inputs = data.get("assembler_inputs", [])
        tree_importances = data.get("tree_importances", {})
        lr_coefficients = data.get("lr_coefficients", {})

        tree_card = Card(frame, title="Tree-based feature importance (Decision Tree / Random Forest)")
        tree_card.pack(fill="both", expand=True, padx=4, pady=(4, 10))
        n_tree_models = len(tree_importances)
        if n_tree_models:
            fig = Figure(figsize=(4.6 * n_tree_models, 3.6), dpi=100)
            for i, (name, importances) in enumerate(tree_importances.items(), start=1):
                ax = fig.add_subplot(1, n_tree_models, i)
                pairs = sorted(zip(assembler_inputs, importances), key=lambda kv: kv[1])[-12:]
                labels = [k for k, _ in pairs]
                values = [v for _, v in pairs]
                ax.barh(labels, values, color=BAR_COLORS[(i - 1) % len(BAR_COLORS)])
                ax.set_title(name, fontsize=9)
                ax.tick_params(axis="y", labelsize=7)
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=tree_card)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            ttk.Label(tree_card, text="No tree_importances found.", style="CardMuted.TLabel").pack()

        if lr_coefficients:
            lr_card = Card(frame, title="Logistic Regression coefficients by class (red = negative, blue = positive)")
            lr_card.pack(fill="both", expand=True, padx=4)
            classes = list(lr_coefficients.keys())
            fig2 = Figure(figsize=(4.6 * len(classes), 3.6), dpi=100)
            for i, class_name in enumerate(classes, start=1):
                ax = fig2.add_subplot(1, len(classes), i)
                coefs = lr_coefficients[class_name]
                pairs = sorted(zip(assembler_inputs, coefs), key=lambda kv: abs(kv[1]))[-12:]
                labels = [k for k, _ in pairs]
                values = [v for _, v in pairs]
                colors = ["#e0607e" if v < 0 else "#2f8fef" for v in values]
                ax.barh(labels, values, color=colors)
                ax.set_title(f"Class: {class_name}", fontsize=9)
                ax.tick_params(axis="y", labelsize=7)
                ax.axvline(0, color=FG_MUTED, linewidth=0.6)
            fig2.tight_layout()
            canvas2 = FigureCanvasTkAgg(fig2, master=lr_card)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------ Cross-Validation --
    def _build_cv_tab(self, frame):
        data = self.cv_results
        if not data:
            ttk.Label(frame, text=f"cv_results.json not found at:\n{CV_RESULTS_JSON}").pack(pady=20)
            return

        cv_df = pd.DataFrame(data)
        cv_df["params_str"] = cv_df["params"].apply(lambda d: ", ".join(f"{k}={v}" for k, v in d.items()))

        box_card = Card(frame, title="Cross-validation F1 spread across hyperparameter grid, per model")
        box_card.pack(fill="both", expand=True, padx=4, pady=(4, 10))
        fig = Figure(figsize=(8.5, 3.4), dpi=100)
        ax = fig.add_subplot(111)
        model_order = self.model_names or cv_df["model"].unique().tolist()
        box_data = [cv_df[cv_df["model"] == m]["avg_f1"].values for m in model_order]
        bp = ax.boxplot(box_data, tick_labels=model_order, patch_artist=True)
        for patch, color in zip(bp["boxes"], BAR_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel("CV avg F1")
        ax.tick_params(axis="x", rotation=12)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=box_card)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        detail_card = Card(frame, title="Every hyperparameter run")
        detail_card.pack(fill="both", expand=True, padx=4)
        cols = ["model", "params_str", "avg_f1"]
        tree = ttk.Treeview(detail_card, columns=cols, show="headings", height=len(cv_df) + 1)
        headers = {"model": "Model", "params_str": "Parameters", "avg_f1": "Avg F1"}
        for c in cols:
            tree.heading(c, text=headers[c])
            tree.column(c, width=260 if c == "params_str" else 150, anchor="center")
        for _, row in cv_df.sort_values(["model", "avg_f1"], ascending=[True, False]).iterrows():
            tree.insert("", "end", values=[row["model"], row["params_str"], f"{row['avg_f1']:.4f}"])
        tree.pack(fill="both", expand=True)

    # ------------------------------------------------------ Predict Route --
    def _build_predict_placeholder(self, frame):
        self.predict_body = frame
        self.predict_placeholder = ttk.Label(
            frame, text="Open this tab to start Spark and load your trained model...",
            style="Muted.TLabel", padding=20,
        )
        self.predict_placeholder.pack(anchor="w")

    def _start_spark_backend(self):
        self._spark_starting = True
        self.predict_placeholder.config(text="Starting Spark and loading model -- this can take a while...")
        self.update_idletasks()

        def worker():
            try:
                backend = SparkBackend(status_callback=lambda m: self.after(0, self._set_status, m))
            except Exception as e:
                self.after(0, self._on_spark_failed, e)
                return
            self.after(0, self._on_spark_ready, backend)

        threading.Thread(target=worker, daemon=True).start()

    def _on_spark_failed(self, error):
        self._spark_starting = False
        self.predict_placeholder.config(text=f"Could not start Spark / load model:\n\n{error}")
        messagebox.showerror("Startup failed", str(error))

    def _on_spark_ready(self, backend):
        self.backend = backend
        self._spark_starting = False
        self.predict_placeholder.destroy()
        self._set_status("Ready.")

        self.numeric_defaults = {c: backend.get_numeric_median(c) for c in backend.numeric_features}
        self.categorical_options = {c: backend.get_categorical_options(c) for c in backend.categorical_features}
        self.routes_df = backend.get_route_list()

        sub_notebook = ttk.Notebook(self.predict_body)
        sub_notebook.pack(fill="both", expand=True)
        lookup_tab = ttk.Frame(sub_notebook, style="TFrame")
        whatif_tab = ttk.Frame(sub_notebook, style="TFrame")
        sub_notebook.add(lookup_tab, text="\U0001F50E  Lookup an existing route")
        sub_notebook.add(whatif_tab, text="\U0001F9EA  What-if scenario")

        self._build_lookup_tab(lookup_tab)
        self._build_whatif_tab(whatif_tab)

    def _build_lookup_tab(self, frame):
        control_card = Card(frame, title="Select a route")
        control_card.pack(fill="x", padx=4, pady=(4, 12))
        row = ttk.Frame(control_card, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text="Route (line_ref):", style="Card.TLabel").pack(side="left")
        route_values = self.routes_df["line_ref"].astype(str).tolist() if not self.routes_df.empty else []
        self.route_combo = ttk.Combobox(row, values=route_values, state="readonly", width=30)
        if route_values:
            self.route_combo.current(0)
        self.route_combo.pack(side="left", padx=12)
        self.lookup_predict_btn = ttk.Button(row, text="Predict", command=self._predict_lookup)
        self.lookup_predict_btn.pack(side="left")

        self.lookup_result_frame = ttk.Frame(frame, style="TFrame")
        self.lookup_result_frame.pack(fill="both", expand=True, padx=4)

    def _predict_lookup(self):
        if self.routes_df.empty:
            messagebox.showwarning("No data", "No routes found in the feature dataset.")
            return
        line_ref = self.route_combo.get()
        self.lookup_predict_btn.config(state="disabled")
        self._set_status(f"Predicting for route {line_ref}... (this can take a few seconds)")
        try:
            pred_class, probs, feature_values = self.backend.predict_for_route(line_ref)
        finally:
            self.lookup_predict_btn.config(state="normal")
        self._set_status("Ready.")
        if pred_class is None:
            messagebox.showerror("Not found", f"No feature row found for line_ref={line_ref}")
            return
        self._render_result(self.lookup_result_frame, pred_class, probs, details=feature_values)

    def _build_whatif_tab(self, frame):
        form_row = ttk.Frame(frame, style="TFrame")
        form_row.pack(fill="x", padx=4, pady=(4, 12))

        self.numeric_vars, self.categorical_vars = {}, {}

        numeric_card = Card(form_row, title="Numeric features")
        numeric_card.pack(side="left", fill="both", expand=True, padx=(0, 6))
        for c in self.backend.numeric_features:
            row = ttk.Frame(numeric_card, style="Card.TFrame")
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=c, width=22, style="Card.TLabel").pack(side="left")
            var = tk.StringVar(value=str(self.numeric_defaults.get(c, 0)))
            ttk.Entry(row, textvariable=var, width=12).pack(side="left")
            self.numeric_vars[c] = var

        categorical_card = Card(form_row, title="Categorical features")
        categorical_card.pack(side="left", fill="both", expand=True, padx=(6, 0))
        for c in self.backend.categorical_features:
            row = ttk.Frame(categorical_card, style="Card.TFrame")
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=c, width=22, style="Card.TLabel").pack(side="left")
            options = self.categorical_options.get(c, [])
            default = options[0] if options else ""
            var = tk.StringVar(value=str(default))
            combo = ttk.Combobox(row, textvariable=var, values=options, state="readonly", width=18)
            combo.pack(side="left")
            self.categorical_vars[c] = var

        button_row = ttk.Frame(frame, style="TFrame")
        button_row.pack(fill="x", padx=4)
        self.whatif_predict_btn = ttk.Button(button_row, text="Predict", command=self._predict_whatif)
        self.whatif_predict_btn.pack(pady=6)

        self.whatif_result_frame = ttk.Frame(frame, style="TFrame")
        self.whatif_result_frame.pack(fill="both", expand=True, padx=4)

    def _predict_whatif(self):
        try:
            feature_values = {c: v.get() for c, v in self.numeric_vars.items()}
            feature_values.update({c: v.get() for c, v in self.categorical_vars.items()})
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        self.whatif_predict_btn.config(state="disabled")
        self._set_status("Predicting... (this can take a few seconds)")
        try:
            pred_class, probs = self.backend.predict_whatif(feature_values)
        except Exception as e:
            self._set_status("Ready.")
            messagebox.showerror("Prediction failed", str(e))
            return
        finally:
            self.whatif_predict_btn.config(state="normal")
        self._set_status("Ready.")
        self._render_result(self.whatif_result_frame, pred_class, probs)

    def _render_result(self, container, pred_class, probs, details=None):
        for widget in container.winfo_children():
            widget.destroy()

        result_card = Card(container, title="Prediction result")
        result_card.pack(fill="both", expand=True)

        ttk.Label(result_card, text=f"Predicted popularity tier:  {pred_class}",
                  style="Metric.TLabel").pack(anchor="w", pady=(0, 12))

        body = ttk.Frame(result_card, style="Card.TFrame")
        body.pack(fill="both", expand=True)

        fig = Figure(figsize=(4.6, 3.3), dpi=100)
        ax = fig.add_subplot(111)
        classes = list(probs.keys())
        values = list(probs.values())
        ax.bar(classes, values, color=BAR_COLORS[: len(classes)])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Predicted class probabilities")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=8)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=body)
        canvas.draw()
        canvas.get_tk_widget().pack(side="left", padx=(0, 12))

        if details:
            details_frame = ttk.Frame(body, style="Card.TFrame")
            details_frame.pack(side="left", fill="both", expand=True)
            ttk.Label(details_frame, text="Route feature values", style="CardHeading.TLabel").pack(anchor="w", pady=(0, 6))
            text = tk.Text(details_frame, height=14, width=42, bg=BG_INPUT, fg=FG_TEXT,
                            relief="flat", font=("Consolas", 9))
            text.pack(fill="both", expand=True)
            for k, v in details.items():
                text.insert("end", f"{k}: {v}\n")
            text.config(state="disabled")


if __name__ == "__main__":
    app = DashboardApp()
    app.mainloop()
