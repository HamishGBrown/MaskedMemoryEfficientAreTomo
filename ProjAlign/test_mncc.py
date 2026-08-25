#!/usr/bin/env python3
"""
Regression test for the masked normalised cross-correlation (MNCC)
used in ProjAlign/CCentralXcf.cpp :: mCorrelateMasked / mFindPeakMncc.

Tests the reference Python implementation of the MNCC algorithm against
known shifts.  All algorithmic choices mirror the CUDA implementation:

  num(t)  = IFFT( conj(FFT(ref))  x FFT(img*G) )
  d1(t)   = IFFT( conj(FFT(ref^2)) x FFT(G)    )
  d2      = sum( img^2 * G )                        [scalar]
  MNCC(t) = num(t) / sqrt( |d1(t)| * d2 + eps )

Peak location uses parabolic sub-pixel refinement with wrap-around
neighbours (matching the fix in mFindPeakMncc).

Run:
    python3 ProjAlign/test_mncc.py

Exit code 0 = all tests passed, 1 = failures.
"""

import sys
import numpy as np


# ---------------------------------------------------------------------------
# Reference MNCC implementation (mirrors CCentralXcf::mCorrelateMasked)
# ---------------------------------------------------------------------------

def mncc_map(ref, img, mask, eps_scale=1e-6):
    """
    Compute the MNCC map for ref/img with a single mask G applied to img.

    Zero-shift is at pixel (0, 0) in the returned array, matching the
    uncentred FFT convention used by CUFFT in the CUDA code.

    Parameters
    ----------
    ref, img, mask : 2-D float arrays of the same shape.
    eps_scale      : regularisation relative to d2.

    Returns
    -------
    mncc : 2-D float array, same shape as inputs.
    """
    img_masked = img * mask
    d2 = float(np.sum(img ** 2 * mask))

    F_ref  = np.fft.rfft2(ref)
    F_img  = np.fft.rfft2(img_masked)
    F_ref2 = np.fft.rfft2(ref ** 2)
    F_mask = np.fft.rfft2(mask)

    num = np.fft.irfft2(np.conj(F_ref)  * F_img,  s=ref.shape)
    d1  = np.fft.irfft2(np.conj(F_ref2) * F_mask, s=ref.shape)

    eps = eps_scale * d2 if d2 > 0.0 else 1e-10
    return num / np.sqrt(np.abs(d1) * d2 + eps)


# ---------------------------------------------------------------------------
# Sub-pixel peak finder (mirrors mFindPeakMncc after the bug-fix)
# ---------------------------------------------------------------------------

def find_peak_subpixel(mncc):
    """
    Locate the peak of the MNCC map with sub-pixel accuracy via
    parabolic interpolation, using wrap-around neighbours.

    Returns (shift_x, shift_y) where negative values mean the image
    is shifted left / up relative to the reference.
    """
    ny, nx = mncc.shape
    flat_idx = np.argmax(mncc)
    py, px = divmod(flat_idx, nx)

    # Wrap-around neighbours (FFT output is cyclic)
    xp = (px + 1) % nx
    xm = (px - 1) % nx
    yp = (py + 1) % ny
    ym = (py - 1) % ny

    ic = mncc[py, px]
    ax = (mncc[py, xp] + mncc[py, xm]) * 0.5 - ic
    bx = (mncc[py, xp] - mncc[py, xm]) * 0.5
    ay = (mncc[yp, px] + mncc[ym, px]) * 0.5 - ic
    by = (mncc[yp, px] - mncc[ym, px]) * 0.5

    sub_x = -bx / (2.0 * ax) if abs(ax) > 1e-30 else 0.0
    sub_y = -by / (2.0 * ay) if abs(ay) > 1e-30 else 0.0

    # Clamp: sub-pixel correction should be < 1 pixel for a genuine peak
    if abs(sub_x) > 1.0:
        sub_x = 0.0
    if abs(sub_y) > 1.0:
        sub_y = 0.0

    peak_x = px + sub_x
    peak_y = py + sub_y

    # Convert FFT pixel position to shift (wrap-around)
    shift_x = peak_x - nx if peak_x > nx * 0.5 else peak_x
    shift_y = peak_y - ny if peak_y > ny * 0.5 else peak_y
    return float(shift_x), float(shift_y)


def find_peak_integer_only(mncc):
    """Reproduce the pre-fix integer-only behaviour of mFindPeakMncc."""
    ny, nx = mncc.shape
    flat_idx = np.argmax(mncc)
    py, px = divmod(flat_idx, nx)
    shift_x = px - nx if px > nx // 2 else px
    shift_y = py - ny if py > ny // 2 else py
    return float(shift_x), float(shift_y)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def shift_image_fft(img, shift_x, shift_y):
    """Phase-shift an image by (shift_x, shift_y) pixels (sub-pixel exact)."""
    ny, nx = img.shape
    F = np.fft.fft2(img)
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    KX, KY = np.meshgrid(kx, ky)
    F_shifted = F * np.exp(-2j * np.pi * (KX * shift_x + KY * shift_y))
    return np.real(np.fft.ifft2(F_shifted))


def make_test_image(size=128, sigma=18, noise=0.05, seed=0):
    """Gaussian blob plus noise, suitable as a synthetic projection."""
    rng = np.random.default_rng(seed)
    x = np.arange(size)
    X, Y = np.meshgrid(x, x)
    cx = cy = size / 2.0
    img = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    img += rng.standard_normal(img.shape) * noise
    return img


def make_circular_mask(size=128, radius_frac=0.38):
    """Binary circular mask (1 inside, 0 outside)."""
    x = np.arange(size)
    X, Y = np.meshgrid(x, x)
    cx = cy = size / 2.0
    return ((X - cx) ** 2 + (Y - cy) ** 2 <= (radius_frac * size) ** 2).astype(float)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def run_tests():
    N = 128
    ref_base = make_test_image(N)
    mask = make_circular_mask(N)
    ref = ref_base * mask

    # (applied_x, applied_y, tolerance_pixels, description)
    cases = [
        ( 0.00,  0.00, 0.10, "zero shift"),
        ( 1.00,  0.00, 0.10, "integer +1 x"),
        (-1.00,  0.00, 0.10, "integer -1 x"),
        ( 0.00,  2.00, 0.10, "integer +2 y"),
        ( 3.50,  0.00, 0.15, "half-pixel 3.5 x"),
        ( 0.00, -2.50, 0.15, "half-pixel -2.5 y"),
        ( 2.30,  1.70, 0.15, "sub-pixel diagonal"),
        (-1.80,  3.20, 0.15, "sub-pixel diagonal neg"),
        ( 0.30,  0.30, 0.15, "small sub-pixel"),
        ( 5.00,  4.00, 0.15, "larger shift"),
    ]

    n_pass = 0
    n_fail = 0
    failures = []

    hdr = f"  {'Description':<28} {'Applied(x,y)':>14}  {'Recovered(x,y)':>16}  {'Error(x,y)':>14}  Result"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for sx, sy, tol, desc in cases:
        img = shift_image_fft(ref, sx, sy)
        mncc = mncc_map(ref, img, mask)
        rx, ry = find_peak_subpixel(mncc)
        ex, ey = abs(rx - sx), abs(ry - sy)
        ok = ex < tol and ey < tol
        result = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        else:
            n_fail += 1
            failures.append((desc, sx, sy, rx, ry, ex, ey, tol))
        print(f"  {desc:<28} ({sx:+.2f},{sy:+.2f})        ({rx:+.3f},{ry:+.3f})        ({ex:.3f},{ey:.3f})   {result}")

    return n_pass, n_fail, failures


def run_integer_only_demo():
    """
    Show that integer-only peak finding (pre-fix behaviour) gives coarse
    results on sub-pixel shifts, illustrating the reported bug.
    """
    N = 128
    ref = make_test_image(N) * make_circular_mask(N)
    mask = make_circular_mask(N)

    sub_pixel_cases = [(2.3, 1.7), (0.3, 0.3), (-1.8, 3.2)]
    print("\n  Integer-only vs sub-pixel peak finding (bug illustration):")
    print(f"  {'Applied':>12}  {'Int-only':>12}  {'Sub-pixel':>12}")
    for sx, sy in sub_pixel_cases:
        img = shift_image_fft(ref, sx, sy)
        mncc = mncc_map(ref, img, mask)
        ix, iy = find_peak_integer_only(mncc)
        rx, ry = find_peak_subpixel(mncc)
        print(f"  ({sx:+.1f},{sy:+.1f})    ({ix:+.1f},{iy:+.1f})    ({rx:+.3f},{ry:+.3f})")


def run_mask_tests():
    """Verify that masking different regions gives correct shifts."""
    N = 128
    ref_full = make_test_image(N)
    sx, sy = 3.7, -2.1

    masks = {
        "full mask (ones)": np.ones((N, N)),
        "circular 38%":     make_circular_mask(N, 0.38),
        "circular 20%":     make_circular_mask(N, 0.20),
    }

    print("\n  Effect of mask size on accuracy (shift = +3.7, -2.1):")
    print(f"  {'Mask':>20}  {'Recovered(x,y)':>16}  {'Error(x,y)':>14}  Result")
    all_ok = True
    for name, mask in masks.items():
        ref = ref_full * mask
        img = shift_image_fft(ref, sx, sy)
        mncc = mncc_map(ref, img, mask)
        rx, ry = find_peak_subpixel(mncc)
        ex, ey = abs(rx - sx), abs(ry - sy)
        ok = ex < 0.20 and ey < 0.20
        all_ok = all_ok and ok
        result = "PASS" if ok else "FAIL"
        print(f"  {name:>20}  ({rx:+.3f},{ry:+.3f})        ({ex:.3f},{ey:.3f})   {result}")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("MNCC sub-pixel accuracy tests")
    print("=" * 70)
    print()

    print("Sub-pixel shift recovery:")
    n_pass, n_fail, failures = run_tests()

    run_integer_only_demo()

    mask_ok = run_mask_tests()

    print()
    print("=" * 70)
    total = n_pass + n_fail + (0 if mask_ok else 1)
    print(f"Result: {n_pass} passed, {n_fail + (0 if mask_ok else 1)} failed")
    if failures or not mask_ok:
        print("\nFailed cases:")
        for f in failures:
            desc, sx, sy, rx, ry, ex, ey, tol = f
            print(f"  {desc}: applied ({sx:.2f},{sy:.2f}), got ({rx:.3f},{ry:.3f}), "
                  f"error ({ex:.3f},{ey:.3f}), tol {tol:.2f}")
        if not mask_ok:
            print("  mask size tests failed")
        sys.exit(1)
    else:
        print("All tests passed.")
        sys.exit(0)
