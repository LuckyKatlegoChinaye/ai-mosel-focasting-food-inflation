from __future__ import annotations

import shlex
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import END, StringVar, Tk, filedialog, ttk
from tkinter.scrolledtext import ScrolledText

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"


class ForecastingGui(Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Food Inflation Forecasting")
        self.geometry("960x640")
        self.minsize(860, 520)

        self._busy = False
        self.status_var = StringVar(value="Ready")
        self.progress_var = StringVar(value="Idle")
        self.selected_file_var = StringVar(value="No file selected")
        self.driver_var = StringVar(value="all")
        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=16)
        main.pack(fill="both", expand=True)

        title = ttk.Label(main, text="Food Inflation Forecasting Dashboard", font=("Segoe UI", 14, "bold"))
        title.pack(anchor="w", pady=(0, 12))

        row = ttk.Frame(main)
        row.pack(fill="x", pady=(0, 10))

        ttk.Button(row, text="Upload CSV", command=self._upload_file).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="Run Preprocessing", command=lambda: self._run_step("Preprocessing", "preprocessing.py")).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="Train Models", command=lambda: self._run_step("Models", "models.py")).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="Evaluate Results", command=lambda: self._run_step("Evaluation", "evaluation.py")).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="Run All", command=self._run_all).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="Clear Log", command=self._clear_log).pack(side="left")

        status_row = ttk.Frame(main)
        status_row.pack(fill="x", pady=(0, 8))
        ttk.Label(status_row, text="Status:").pack(side="left")
        ttk.Label(status_row, textvariable=self.status_var).pack(side="left", padx=(6, 0))

        progress_row = ttk.Frame(main)
        progress_row.pack(fill="x", pady=(0, 8))
        ttk.Label(progress_row, text="Progress:").pack(side="left")
        ttk.Label(progress_row, textvariable=self.progress_var).pack(side="left", padx=(6, 0))

        driver_frame = ttk.Frame(main)
        driver_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(driver_frame, text="Training focus:").pack(side="left")
        driver_combo = ttk.Combobox(
            driver_frame,
            textvariable=self.driver_var,
            values=["all", "bdi", "brent"],
            state="readonly",
            width=16,
        )
        driver_combo.pack(side="left", padx=(6, 0))
        ttk.Label(main, textvariable=self.selected_file_var).pack(anchor="w", pady=(0, 10))

        self.output_box = ScrolledText(main, wrap="word", height=28)
        self.output_box.pack(fill="both", expand=True)
        self.output_box.insert(END, "Food inflation forecasting workflow\n")
        self.output_box.configure(state="disabled")

    def _upload_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if file_path:
            self.selected_file_var.set(f"Selected file: {file_path}")
            self._log(f"\n[upload] Selected file: {file_path}\n")

    def _log(self, message: str) -> None:
        self.output_box.configure(state="normal")
        self.output_box.insert(END, message)
        self.output_box.see(END)
        self.output_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self.output_box.configure(state="normal")
        self.output_box.delete("1.0", END)
        self.output_box.configure(state="disabled")

    def _run_step(self, label: str, script_name: str) -> None:
        if self._busy:
            self._log("\nA workflow step is already running. Please wait for it to finish.\n")
            return

        self._busy = True
        self.status_var.set(f"Running {label}...")
        self.progress_var.set(f"Executing {script_name}")
        self._log(f"\n[{label.lower()}] Starting {script_name} with focus '{self.driver_var.get()}'...\n")

        def worker() -> None:
            command = [sys.executable, str(SRC_DIR / script_name)]
            if script_name == "preprocessing.py" and self.selected_file_var.get() != "No file selected":
                file_path = self.selected_file_var.get().replace("Selected file: ", "")
                command.extend(["--input", file_path])
            if script_name in {"models.py", "evaluation.py"}:
                command.extend(["--driver", self.driver_var.get()])

            result = subprocess.run(
                command,
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
            )
            self.after(0, lambda: self._finish_step(label, result))

        threading.Thread(target=worker, daemon=True).start()

    def _run_all(self) -> None:
        if self._busy:
            self._log("\nA workflow step is already running. Please wait for it to finish.\n")
            return

        self._busy = True
        self.status_var.set("Running full workflow...")
        self.progress_var.set("preprocessing → models → evaluation")
        self._log("\n[workflow] Starting preprocessing → models → evaluation...\n")

        def worker() -> None:
            selected_file = self.selected_file_var.get()
            selected_path = None
            if selected_file and selected_file != "No file selected":
                selected_path = selected_file.replace("Selected file: ", "")

            preprocessing_cmd = [sys.executable, str(SRC_DIR / "preprocessing.py")]
            if selected_path:
                preprocessing_cmd.extend(["--input", selected_path])

            models_cmd = [sys.executable, str(SRC_DIR / "models.py"), "--driver", self.driver_var.get()]
            evaluation_cmd = [sys.executable, str(SRC_DIR / "evaluation.py"), "--driver", self.driver_var.get()]

            command = " && ".join(
                [
                    " ".join(map(shlex.quote, preprocessing_cmd)),
                    " ".join(map(shlex.quote, models_cmd)),
                    " ".join(map(shlex.quote, evaluation_cmd)),
                ]
            )
            result = subprocess.run(command, cwd=ROOT_DIR, shell=True, capture_output=True, text=True)
            self.after(0, lambda: self._finish_step("Workflow", result))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_step(self, label: str, result: subprocess.CompletedProcess[str]) -> None:
        self._busy = False
        self.progress_var.set("Idle")

        if result.stdout:
            self._log(result.stdout)
        if result.stderr:
            self._log(result.stderr)

        if result.returncode == 0:
            self.status_var.set(f"{label} completed successfully")
            self._log(f"\n[{label.lower()}] Finished with exit code 0\n")
        else:
            self.status_var.set(f"{label} failed")
            self._log(f"\n[{label.lower()}] Failed with exit code {result.returncode}\n")


if __name__ == "__main__":
    app = ForecastingGui()
    app.mainloop()
