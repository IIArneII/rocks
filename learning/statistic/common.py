"""
--------------------------------------------------------------------
Common utilities for rock-size analysis.

Shared functions and classes for both GUI tools.
--------------------------------------------------------------------
"""
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator


class ScaleSelector:
    """OpenCV helper to pick two points and get real scale (mm / px)."""

    def __init__(self, window_name='Select scale (2 clicks) - ESC to quit'):
        self.window = window_name
        self.points = []
        self.finished = False

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 2:
            self.points.append((x, y))

    def get_scale(self, img):
        """Show image, get 2 clicks, ask user for real length; return mm_per_px."""
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback(self.window, self._on_mouse)

        drawn = img.copy()
        while True:
            if len(self.points) == 2:
                cv2.line(drawn, self.points[0], self.points[1], (0, 255, 0), 2)
            cv2.imshow(self.window, drawn)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                cv2.destroyWindow(self.window)
                sys.exit('User cancelled.')
            if len(self.points) == 2:
                cv2.imshow(self.window, drawn)
                cv2.waitKey(10)
                break
        cv2.destroyWindow(self.window)

        while True:
            try:
                true_len = float(input('Enter length of the drawn line in millimetres: '))
                break
            except ValueError:
                print('Not a number, try again.')
        px_len = np.linalg.norm(np.subtract(*self.points))
        mm_per_px = true_len / px_len
        print(f"Scale: {mm_per_px:.4f} mm / px  ( {px_len:.2f} px = {true_len:.2f} mm )")
        return mm_per_px, self.points


def rosin_rammler_fit(df):
    """
    Fit Rosin–Rammler (Weibull) distribution to the fragment sizes,
    weighted by mass (approximated by area).
    Returns: Xc (characteristic size), n (shape exponent), and function F(x)
    such that F(x)=fraction of mass passing.
    """
    # 1) convert area to mass weight (assume density const → mass ∝ area_mm2)
    df = df.sort_values('equiv_diam_mm')
    total_mass = df['area_mm2'].sum()
    if total_mass == 0:
        return np.nan, np.nan, lambda x: np.zeros_like(x, dtype=float)

    cum_mass = np.cumsum(df['area_mm2'].values) / total_mass

    # 2) Rosin–Rammler fit on mass-weighted P
    # Exclude the point where cumulative mass is 1.0 to avoid log(0)
    valid_mask = cum_mass < 1.0
    sizes_fit = df['equiv_diam_mm'].values[valid_mask]
    cum_mass_fit = cum_mass[valid_mask]

    if len(sizes_fit) < 2:
        return np.nan, np.nan, lambda x: np.zeros_like(x, dtype=float)

    ln_x  = np.log(sizes_fit)
    ln_ln = np.log(-np.log(1 - cum_mass_fit))

    # Linear regression:  ln(-ln(1-P))  =  n*ln(x) - n*ln(Xc)
    n, c = np.polyfit(ln_x, ln_ln, 1)
    if abs(n) < 1e-9:
        Xc = np.nan
    else:
        Xc = np.exp(-c / n)

    def F(x):                                    # cumulative model
        with np.errstate(divide='ignore', invalid='ignore'):
            return 1 - np.exp(-(np.asanyarray(x, dtype=float) / Xc)**n)

    return Xc, n, F


def plot_size_distribution(df, outfile_png):
    """
    Draw histogram + cumulative passing curve on logarithmic X axis.
    Adds:
      • D10, D50, D80 markers
      • Coefficient of determination R²
      • Maximum deviation Δ_max (between D10 and D90)
    Saves figure to `outfile_png`.
    """
    sizes = df['equiv_diam_mm'].values
    if sizes.size < 2:
        print("Not enough fragments for a histogram.")
        return

    # ---------- binning ---------------------------------------------------
    n_bins = 15
    log_min, log_max = np.log10(sizes.min()), np.log10(sizes.max())
    bins = np.logspace(log_min, log_max, n_bins)
    counts, edges = np.histogram(sizes, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])            # geometric mean

    # ---------- empirical cumulative (mass-weighted) ----------------------
    df_sorted = df.sort_values('equiv_diam_mm')
    sizes_sorted = df_sorted['equiv_diam_mm'].values
    cum_mass = np.cumsum(df_sorted['area_mm2'].values) / df_sorted['area_mm2'].sum()
    P_emp = cum_mass                                     # fraction finer 0–1
    pct_emp = P_emp * 100                                # % finer (by mass)

    # ---------- characteristic sizes (mass-weighted) ----------------------
    percentiles_to_find = [10, 30, 50, 70, 80, 90]
    d_vals = np.interp(np.array(percentiles_to_find) / 100.0, P_emp, sizes_sorted)
    d10, d30, d50, d70, d80, d90 = d_vals

    # ---------- Rosin–Rammler fit -----------------------------------------
    Xc, n, F_rr = rosin_rammler_fit(df)
    B = 1.0 / n if n != 0 else np.inf
    P_pred = F_rr(sizes_sorted)           # fraction finer predicted

    # ---------- goodness-of-fit metrics -----------------------------------
    # R² (comparing mass-based empirical vs. predicted)
    ss_res = np.sum((P_emp - P_pred) ** 2)
    ss_tot = np.sum((P_emp - P_emp.mean()) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # Δ_max between D10 and D90
    mask_d10_90 = (sizes_sorted >= d10) & (sizes_sorted <= d90)
    if np.any(mask_d10_90):
        delta_max = np.max(np.abs(P_emp[mask_d10_90] - P_pred[mask_d10_90])) * 100  # in %
    else:
        delta_max = 0.0

    # ---------- plotting ---------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Histogram
    ax1.bar(centers, counts, width=np.diff(edges), color='crimson',
            align='center', alpha=0.8, label='Frequency (by count)')
    ax1.set_xlabel('Size (mm)')
    ax1.set_ylabel('Count', color='crimson')
    ax1.set_xscale('log')
    
    # Set fixed X-axis range and logarithmic ticks
    ax1.set_xlim(1, 10000)
    ax1.set_xticks([1, 10, 100, 1000, 10000])
    ax1.set_xticklabels(['1', '10', '100', '1000', '10000'])
    
    # Set minimum count axis limit to 100
    current_ylim = ax1.get_ylim()
    ax1.set_ylim(0, max(40, current_ylim[1]))
    
    ax1.tick_params(axis='y', labelcolor='crimson')

    # Empirical cumulative (mass-based)
    ax2 = ax1.twinx()
    ax2.plot(sizes_sorted, pct_emp, color='royalblue', lw=2,
             label='% passing by mass (empirical)')
    ax2.set_ylabel('% Passing by mass (finer)', color='royalblue')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelcolor='royalblue')

    # Rosin–Rammler curve
    x_smooth = np.logspace(log_min, log_max, 300)
    ax2.plot(x_smooth, F_rr(x_smooth) * 100, color='royalblue',
             ls='--', lw=1, label='Rosin–Rammler (mass fit)')

    # D10, D50, D80 markers
    for p, val in zip([10, 50, 80], [d10, d50, d80]):
        ax2.axvline(val, color='k', lw=0.8, ls=':')
        ax2.axhline(p,  color='k', lw=0.8, ls=':')
        ax2.plot(val, p, 'ko')
        ax2.text(val * 1.05, p + 2, f'D{p} = {val:.1f} mm',
                 fontsize=8, va='bottom', ha='left')

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    # ---------- stats textbox ---------------------------------------------
    r2_colour = 'red' if R2 < 0.92 else 'black'
    stats_txt = (f'D10 = {d10:,.1f} mm\n'
                 f'D50 = {d50:,.1f} mm\n'
                 f'D80 = {d80:,.1f} mm\n'
                 f'Xc  = {Xc:,.1f} mm\n'
                 f'XMA = {sizes.max():,.0f} mm\n'
                 f'n   = {n:,.2f}\n'
                 f'B   = {B:,.2f}\n'
                 f'R²  = {R2:,.3f}\n'
                 f'Δ_max = {delta_max:,.2f} %')
    bbox_props = dict(boxstyle='round', fc='white', alpha=0.75)
    txt_obj = ax1.text(0.03, 0.97, stats_txt, transform=ax1.transAxes,
                       va='top', ha='left', bbox=bbox_props)
    # colour the R² line if necessary
    if R2 < 0.92:
        txt_lines = stats_txt.splitlines()
        txt_lines[-2] = f'R²  = {R2:,.3f}  ✗'   # mark fail
        txt_obj.set_text('\n'.join(txt_lines))
        txt_obj._text.set_color(r2_colour)      # type: ignore

    # ----------------------------------------------------------------------
    ax1.grid(True, which='both', ls=':', lw=0.5)
    fig.tight_layout()
    fig.savefig(outfile_png, dpi=150)
    plt.close(fig)
    print('Cumulative plot saved to', outfile_png)


def overlay_results(img, coloured_mask, scale_pts, mm_per_px, out_file, window_title='Segmentation result'):
    """Blend mask on image, draw reference line, save & show."""
    overlay = cv2.addWeighted(img, 0.6, coloured_mask, 0.4, 0)
    p1, p2 = scale_pts
    cv2.line(overlay, p1, p2, (0, 255, 0), 2)
    dist_mm = np.linalg.norm(np.subtract(p1, p2)) * mm_per_px
    cv2.putText(overlay, f"{dist_mm:.1f} mm", p2, cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(out_file, overlay)
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow(window_title, overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
