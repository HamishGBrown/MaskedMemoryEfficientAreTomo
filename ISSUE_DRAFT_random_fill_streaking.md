Title: Random-fill for out-of-bounds pixels in GShiftRotate2D.cu produces a periodic sawtooth pattern instead of noise

## Summary

`mGRandom()` in `Util/GShiftRotate2D.cu` is meant to fill pixels that fall
outside the source image bounds after rotation/shift (used when
`bRandomFill = true`, which is the default in most call sites) with
locally-plausible pseudo-random texture, to avoid the sharp edge that a
constant/zero fill would create. In practice it produces a clean, periodic
sawtooth/ramp pattern instead of noise, visible as high-frequency
"sinusoidal" streaking in the border regions of rotated images and, after
reconstruction, in the tomogram.

## Root cause

```cpp
unsigned int next = y * giInSize[0] + x;
for(int k=0; k<iWinPixels; k++)
{	next = (next * 7) % iWinPixels;
	int ix = (next % iWin) - iWin / 2 + x;
	if(ix < 0 || ix >= iInImgX) continue;
	int iy = (next / iWin) - iWin / 2 + y;
	if(iy < 0 || iy >= giInSize[1]) continue;
	return gfInImg[iy * giInSize[0] + ix];
}
```

Two compounding issues:

1. The loop returns as soon as the first in-bounds candidate is found. Since
   the 31x31 search window is almost always fully inside the source image
   (except right at the true image edge), the loop exits at `k == 0` in the
   overwhelming majority of cases — the "retry" logic is effectively dead
   code.
2. That means the value actually used is just the first step of
   `next = (seed * 7) % 961`, where `seed = y * width + x`. This is a linear
   congruential map with a small, non-prime modulus (31^2) and a small
   multiplier. For a horizontal run of consecutive out-of-bounds pixels
   (`x, x+1, x+2, ...`, the common case along a rotated image's border),
   `seed` increases by 1 each step, so the sampled offset increases by a
   constant `+7 (mod 961)` each step — a deterministic ramp, not noise.

Simulated offsets for 15 consecutive boundary pixels (`y` fixed, `x`
incrementing) using the current code:

```
x= 0 -> offset=( +4, +9)
x= 1 -> offset=(+11, +9)
x= 2 -> offset=(-13,+10)
x= 3 -> offset=( -6,+10)
x= 4 -> offset=( +1,+10)
x= 5 -> offset=( +8,+10)
x= 6 -> offset=(+15,+10)
x= 7 -> offset=( -9,+11)
...
```

`dx` steps by a constant `+7 mod 31` every pixel (wrapping every 31 pixels),
`dy` creeps up by `+1` roughly every 4-5 pixels — a regular sawtooth, which
when used to sample a smooth image produces a repeating, sinusoid-like
texture rather than speckle.

## Suggested fix

Replace the chained LCG state update with a single-pass, well-mixed integer
hash of `seed + k`, so the sampled offset for each retry — and critically,
for adjacent pixels — is uncorrelated instead of linearly related:

```cpp
static __device__ unsigned int mGHash(unsigned int x)
{
	x ^= x >> 17;  x *= 0xed5ad4bbu;
	x ^= x >> 11;  x *= 0xac4c1b51u;
	x ^= x >> 15;  x *= 0x31848babu;
	x ^= x >> 14;
	return x;
}

static __device__ float mGRandom(int x, int y, int iInImgX, float* gfInImg)
{
	if(x < 0) x = -x;
	else if(x >= iInImgX) x = 2 * iInImgX - x;
	if(y < 0) y = -y;
	else if(y >= giInSize[1]) y = 2 * giInSize[1] - y;
	//------------------------------------------------
	int iWin = 31;
	int iWinPixels = iWin * iWin;
	unsigned int seed = y * giInSize[0] + x;
	for(int k=0; k<iWinPixels; k++)
	{	unsigned int next = mGHash(seed + k) % iWinPixels;
		int ix = (next % iWin) - iWin / 2 + x;
		if(ix < 0 || ix >= iInImgX) continue;
		int iy = (next / iWin) - iWin / 2 + y;
		if(iy < 0 || iy >= giInSize[1]) continue;
		return gfInImg[iy * giInSize[0] + ix];
	}
	return gfInImg[y * giInSize[0] + x];
}
```

Same interface, same window/mirroring/fallback logic, only the "how do we
pick a pseudo-random nearby pixel" step changes. Verified in isolation
(Python simulation) that this hash removes the linear correlation between
neighboring pixels' sampled offsets.

## Where this matters

`bRandomFill` defaults to `true` in the call sites that generate the aligned
tilt series / drive reconstruction, so this affects the default pipeline
whenever an image needs to be rotated (e.g. to bring the tilt axis to
vertical) and the rotated grid extends beyond the original image bounds.

## Affected files

- `AreTomo2`: `Util/GShiftRotate2D.cu`
- Likely also `AreTomo3`: `AreTomo/Util/GShiftRotate2D.cu` (same routine,
  same LCG, same call pattern via `CCorrTomoStack.cpp`)
