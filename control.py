import io
import base64
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # important for headless plotting
import matplotlib.pyplot as plt


def controlSensitizingGraph(controlDataBytes, outputCol):
    # Read CSV bytes
    df = pd.read_csv(io.BytesIO(controlDataBytes))

    if outputCol not in df.columns:
        return {
            "ok": False,
            "error": f"Column '{outputCol}' not found. Available columns: {list(df.columns)}"
        }

    data = df[outputCol].dropna().reset_index(drop=True)

    if len(data) < 2:
        return {"ok": False, "error": "Need at least 2 non-missing values to compute std dev."}

    mean = data.mean()
    std = data.std(ddof=1)
    if std == 0 or pd.isna(std):
        return {"ok": False, "error": "Standard deviation is 0/NaN; z-scores are undefined."}

    zScores = []
    problemPoints = set()
    messages = []

    for i, x in enumerate(data):
        z = (x - mean) / std
        zScores.append(float(z))

        # Rule 1: One point outside +/-3
        if abs(z) >= 3:
            messages.append(f"[Rule 1] Point {i} outside control limits: z={z:.2f}")
            problemPoints.add(i)

        # Rule 2: Two of three consecutive points outside +/-2 on same side
        if i >= 2:
            window = zScores[i - 2 : i + 1]
            pos = sum(v >= 2 for v in window)
            neg = sum(v <= -2 for v in window)
            if pos >= 2 or neg >= 2:
                messages.append(f"[Rule 2] 2 of 3 outside +/-2 between {i-2} and {i}")
                problemPoints.update(range(i - 2, i + 1))

        # Rule 3: Four of five beyond +/-1 on same side
        if i >= 4:
            window = zScores[i - 4 : i + 1]
            pos = sum(v >= 1 for v in window)
            neg = sum(v <= -1 for v in window)
            if pos >= 4 or neg >= 4:
                messages.append(f"[Rule 3] 4 of 5 beyond +/-1 between {i-4} and {i}")
                problemPoints.update(range(i - 4, i + 1))

        # Rule 4: Eight consecutive on same side of center line
        if i >= 7:
            window = zScores[i - 7 : i + 1]
            all_pos = all(v > 0 for v in window)
            all_neg = all(v < 0 for v in window)
            if all_pos or all_neg:
                messages.append(f"[Rule 4] 8 points on one side between {i-7} and {i}")
                problemPoints.update(range(i - 7, i + 1))

        # Rule 5: Six steadily increasing or decreasing (uses raw data)
        if i >= 5:
            w = list(data.iloc[i - 5 : i + 1])
            inc = all(w[j] < w[j + 1] for j in range(5))
            dec = all(w[j] > w[j + 1] for j in range(5))
            if inc or dec:
                messages.append(f"[Rule 5] 6 points steadily {'increasing' if inc else 'decreasing'} between {i-5} and {i}")
                problemPoints.update(range(i - 5, i + 1))

        # Rule 6: Fifteen within +/-1
        if i >= 14:
            window = zScores[i - 14 : i + 1]
            if all(abs(v) < 1 for v in window):
                messages.append(f"[Rule 6] 15 points within +/-1 between {i-14} and {i}")
                problemPoints.update(range(i - 14, i + 1))

        # Rule 7: Fourteen alternating up/down (raw data)
        if i >= 13:
            w = list(data.iloc[i - 13 : i + 1])
            diffs = [w[j + 1] - w[j] for j in range(13)]
            if all(d != 0 for d in diffs):
                alternating = all(diffs[j] * diffs[j + 1] < 0 for j in range(12))
                if alternating:
                    messages.append(f"[Rule 7] 14 points alternating up/down between {i - 13} and {i}")
                    problemPoints.update(range(i - 13, i + 1))

        # Rule 8: Eight outside +/-1 on both sides
        if i >= 7:
            window = zScores[i - 7 : i + 1]
            if all(abs(v) >= 1 for v in window) and any(v > 0 for v in window) and any(v < 0 for v in window):
                messages.append(f"[Rule 8] 8 points outside +/-1 on both sides between {i-7} and {i}")
                problemPoints.update(range(i - 7, i + 1))

    # Plot
    x = list(range(len(zScores)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, zScores, marker="o")

    prob = sorted(problemPoints)
    if prob:
        ax.scatter(prob, [zScores[i] for i in prob], color="red", s=60, zorder=10)

    for level in range(-3, 4):
        linestyle = "--" if level != 0 else "-"
        alpha = 0.7 if level != 0 else 1
        ax.axhline(
            level,
            color=("red" if abs(level) >= 2 else "black"),
            linestyle=linestyle,
            linewidth=1,
            alpha=alpha,
        )

    ax.set_title(f"Std Deviations from Mean for '{outputCol}'")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Number of Standard Deviations")
    ax.grid(True, linestyle=":", linewidth=0.5)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)

    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "ok": True,
        "n": int(len(zScores)),
        "mean": float(mean),
        "std": float(std),
        "problem_points": prob,
        "messages": messages,
        "plot_png_base64": img_b64,
    }
