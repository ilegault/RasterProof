import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection


def plot_heatmap(dose, x_edges, y_edges, aperture_rect, metrics_dict=None):
    """
    Plot dose map as a heatmap with aperture outline and optional metric annotations.
    aperture_rect = (xL, xR, yB, yT).
    Returns matplotlib Figure.
    """
    xL, xR, yB, yT = aperture_rect
    fig, ax = plt.subplots(figsize=(7, 6))

    pcm = ax.pcolormesh(x_edges, y_edges, dose.T, cmap="hot", shading="auto")
    fig.colorbar(pcm, ax=ax, label="Dose (a.u.)")

    rect = patches.Rectangle(
        (xL, yB), xR - xL, yT - yB,
        linewidth=2, edgecolor="white", facecolor="none", linestyle="--",
    )
    ax.add_patch(rect)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    title = "Dose Map"
    if metrics_dict:
        flat = metrics_dict.get("flatness_pct", float("nan"))
        title = f"Dose Map — Flatness: {flat:.1f}%"

        pinch = metrics_dict.get("pinch_pct", float("nan"))
        ss = metrics_dict.get("steady_state", None)
        ss_str = "✓ STEADY" if ss else "✗ TRANSIENT"

        annotation = f"Pinch: {pinch:.1f}%\n{ss_str}"
        ax.text(
            0.02, 0.98, annotation,
            transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
        )

    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_dose_3d(dose, x_edges, y_edges, aperture_rect, metrics_dict=None):
    """
    3D surface plot of dose map — inspired by Yan et al. (2005) Fig. 4.
    Dramatically reveals cusping and non-uniformity that the 2D heatmap hides.
    aperture_rect = (xL, xR, yB, yT).
    Returns matplotlib Figure.
    """
    xL, xR, yB, yT = aperture_rect

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Subsample for 3D performance — 64x64 max for interactive rendering
    stride_x = max(1, len(x_centers) // 64)
    stride_y = max(1, len(y_centers) // 64)
    xc = x_centers[::stride_x]
    yc = y_centers[::stride_y]
    Z = dose[::stride_x, ::stride_y]

    X, Y = np.meshgrid(xc, yc, indexing="ij")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Color by height (like Yan's rainbow colormap)
    norm = plt.Normalize(Z.min(), Z.max())
    facecolors = plt.cm.jet(norm(Z))

    ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)

    # Draw aperture outline projected onto the floor
    z_floor = Z.min()
    box_x = [xL, xR, xR, xL, xL]
    box_y = [yB, yB, yT, yT, yB]
    box_z = [z_floor] * 5
    ax.plot(box_x, box_y, box_z, "w--", linewidth=1.5, alpha=0.7, label="Aperture")

    ax.set_xlabel("X (mm)", fontsize=9, labelpad=6)
    ax.set_ylabel("Y (mm)", fontsize=9, labelpad=6)
    ax.set_zlabel("Dose (a.u.)", fontsize=9, labelpad=6)

    title = "Dose surface (3D)"
    if metrics_dict:
        flat = metrics_dict.get("flatness_pct", float("nan"))
        title = f"Dose surface — Flatness: {flat:.1f}%"

    ax.set_title(title, fontsize=10, pad=10)
    ax.view_init(elev=30, azim=-60)

    # Add colorbar mapped to Z range
    sm = plt.cm.ScalarMappable(cmap="jet", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1, label="Dose (a.u.)")

    fig.tight_layout()
    return fig


def plot_velocity_profile(params, t_arr, x_arr, y_arr):
    """
    Probe velocity vs time — inspired by Teo et al. (2018) Fig. 7.
    Shows why sinusoidal/Lissajous patterns have non-uniform dose deposition:
    velocity -> 0 at turnarounds means the beam lingers at edges (cusping).

    Returns matplotlib Figure with two panels:
      top: velocity magnitude over time
      bottom: velocity histogram (equivalent to inverse dwell distribution)
    """
    if len(t_arr) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough samples", ha="center", va="center")
        return fig

    dt = t_arr[1] - t_arr[0]

    # Instantaneous velocity components via finite differences
    vx = np.gradient(x_arr, dt)
    vy = np.gradient(y_arr, dt)
    speed = np.sqrt(vx**2 + vy**2)   # mm/s

    # Subsample for plotting — 5000 points max for legibility
    n = len(t_arr)
    stride = max(1, n // 5000)
    t_plot = t_arr[::stride] * 1000   # convert to ms
    s_plot = speed[::stride]

    pattern = params.get("pattern", "classic")
    fx = params.get("fx_hz", 0)
    fy = params.get("fy_hz", 0)

    # Theoretical max velocity for annotation
    ax_mm = params.get("ax_mm", 1.0)
    ay_mm = params.get("ay_mm", 1.0)
    if pattern == "classic":
        v_theory = 4 * ax_mm * fx
    elif pattern in ("sinusoidal", "wobble"):
        v_theory = np.pi * ax_mm * fx
    elif pattern == "lissajous":
        v_theory = np.pi * np.sqrt((ax_mm * fx)**2 + (ay_mm * fy)**2)
    else:
        v_theory = speed.max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        f"Probe velocity profile — pattern: {pattern}  "
        f"(fx={fx:.0f} Hz, fy={fy:.0f} Hz)",
        fontsize=10,
    )

    # --- Left: velocity vs time ---
    ax1.plot(t_plot, s_plot, lw=0.7, color="#2176ae", alpha=0.85)
    ax1.axhline(speed.mean(), color="orange", lw=1.5, ls="--",
                label=f"Mean = {speed.mean():.0f} mm/s")
    ax1.axhline(v_theory, color="red", lw=1.0, ls=":",
                label=f"v_max theory = {v_theory:.0f} mm/s")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Speed (mm/s)")
    ax1.set_title("Speed vs time")
    ax1.legend(fontsize=8)
    ax1.set_xlim(t_plot[0], min(t_plot[-1], t_plot[0] + 20))  # show first 20 ms

    # --- Right: speed histogram ---
    # This is the inverse of dose: low-speed regions get high dose
    ax2.hist(speed, bins=80, color="#2176ae", edgecolor="none", alpha=0.8,
             density=True)
    ax2.axvline(speed.mean(), color="orange", lw=1.5, ls="--",
                label=f"Mean = {speed.mean():.0f} mm/s")
    ax2.set_xlabel("Speed (mm/s)")
    ax2.set_ylabel("Probability density")
    ax2.set_title("Speed distribution\n(more time at low speed → higher dose)")
    ax2.legend(fontsize=8)

    # Annotation: if sinusoidal, explain the arcsin distribution
    if pattern in ("sinusoidal", "lissajous"):
        ax2.text(
            0.98, 0.97,
            "Sinusoidal → arcsine\ndistribution:\nmore time at v≈0\n(edge cusping)",
            transform=ax2.transAxes, fontsize=7,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="#ccaa00", alpha=0.9),
        )
    elif pattern == "classic":
        ax2.text(
            0.98, 0.97,
            "Triangle wave →\nnearly constant speed\n(good uniformity)\nSmall peak at v≈0\n(vertex dwell)",
            transform=ax2.transAxes, fontsize=7,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f4e8",
                      edgecolor="#448844", alpha=0.9),
        )

    fig.tight_layout()
    return fig


def animate_trajectory(x_traj, y_traj, t_arr, aperture_rect=None, save_path=None):
    """
    Animate beam trajectory. Subsamples to <= 500 frames.
    Returns FuncAnimation object.
    """
    stride = max(1, len(t_arr) // 500)
    x_s = x_traj[::stride]
    y_s = y_traj[::stride]
    t_s = t_arr[::stride]
    n_frames = len(x_s)
    tail_len = min(50, n_frames)

    fig, ax = plt.subplots(figsize=(6, 6))

    if aperture_rect is not None:
        xL, xR, yB, yT = aperture_rect
        rect = patches.Rectangle(
            (xL, yB), xR - xL, yT - yB,
            linewidth=2, edgecolor="blue", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)

    scatter = ax.scatter([], [], s=20, c=[], cmap="plasma", vmin=t_s[0], vmax=t_s[-1])
    dot, = ax.plot([], [], "ro", markersize=8)

    margin = max(np.ptp(x_s), np.ptp(y_s)) * 0.05 + 1
    ax.set_xlim(x_s.min() - margin, x_s.max() + margin)
    ax.set_ylim(y_s.min() - margin, y_s.max() + margin)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Beam Trajectory")

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        dot.set_data([], [])
        return scatter, dot

    def update(frame):
        start = max(0, frame - tail_len)
        tail_x = x_s[start : frame + 1]
        tail_y = y_s[start : frame + 1]
        tail_t = t_s[start : frame + 1]
        offsets = np.column_stack([tail_x, tail_y])
        scatter.set_offsets(offsets)
        scatter.set_array(tail_t)
        dot.set_data([x_s[frame]], [y_s[frame]])
        return scatter, dot

    ani = FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        blit=True, interval=40,
    )

    if save_path is not None:
        ani.save(save_path, writer="pillow", fps=25)

    return ani


def plot_dwell_hist(rho, aperture_mask):
    """
    Histogram of dwell-time values inside aperture.
    Fixed: uses auto-binning (Freedman-Diaconis) and drops near-zero pixels
    to avoid isolated bars from sparse sampling.
    Returns matplotlib Figure.
    """
    vals = rho[aperture_mask > 0]

    # Drop exact zeros — they are unvisited pixels, not beam dwell
    vals = vals[vals > 0]

    fig, ax = plt.subplots(figsize=(6, 4))

    if len(vals) > 0:
        # Use Freedman-Diaconis rule for bin width (better for skewed data)
        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
        if iqr > 0:
            bin_width = 2.0 * iqr / len(vals) ** (1 / 3)
            n_bins = max(20, min(200, int((vals.max() - vals.min()) / bin_width)))
        else:
            n_bins = 50

        ax.hist(vals, bins=n_bins, color="steelblue", edgecolor="none", alpha=0.8)
        mu = vals.mean()
        ax.axvline(mu, color="orange", linestyle="--", linewidth=2,
                   label=f"Mean = {mu:.4g} s")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No dwell data inside aperture",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Dwell Time (s/bin)")
    ax.set_ylabel("Pixel Count")
    ax.set_title("Dwell-Time Distribution (aperture pixels only)")
    fig.tight_layout()
    return fig