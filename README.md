# AreTomo2

## This fork: masked, memory-efficient alignment and reconstruction

This is a fork of [czimaginginstitute/AreTomo2](https://github.com/czimaginginstitute/AreTomo2)
(branched from upstream commit `01e7219`), adding masked projection alignment and
memory-efficient reconstruction. It was developed to make montage tomography more
workable since:
- 1 TB RAM jobs never run on my cluster  
- Contaminants, FIB curtaining, lamella edges etc. are often less avoidable in montage tomography
    and require masking 

The code might be more useful to a general tomography audience with a penchant for dirty grids and 
less budget to ride out the current RAM inflation crisis :D.

Key changes relative to upstream:

- **Masked normalized cross-correlation (MNCC) projection alignment** (`-MaskFile`) —
  `ProjAlign/CCentralXcf.cpp`, `ProjAlign/GMnccXcf.cu`. Projection-matching alignment
  can be restricted to a user-supplied mask (Padfield 2012 masked NCC), so the
  cross-correlation isn't biased by contaminants, fib-curtaining, or other features you don't
  want driving alignment.
- **Streaming WBP reconstruction** — `Recon/CDoWbpRecon.cpp`, `MrcUtil/CTomoStack.cpp`.
  The output tomogram is written slice-by-slice directly to disk via `pwrite()` instead
  of being accumulated entirely in RAM, removing the full output-volume allocation from
  peak memory use for large tilt series. See
  [docs/streaming_wbp_reconstruction.md](docs/streaming_wbp_reconstruction.md).
- **`-OutputROI` sub-region reconstruction** — `CInput.cpp`, `Recon/CDoWbpRecon.cpp`,
  `Recon/CDoSartRecon.cpp`. Reconstruct and write only a specified XY sub-region of the
  tomogram; compatible with streamed reconstruction, unlike `-CropVol`.
- **Fixed periodic-noise bug in random-fill border padding** — `Util/GShiftRotate2D.cu`.
  The pseudo-random fill used for out-of-bounds pixels after rotation/shift produced a
  deterministic sawtooth pattern instead of noise; replaced the linear-congruential
  update with a proper integer hash. See
  [ISSUE_DRAFT_random_fill_streaking.md](ISSUE_DRAFT_random_fill_streaking.md).
- **ROI editor tool** (`tools/roi_editor.py`) — interactive matplotlib GUI for creating
  and editing the patch-centre ROI file used by `-RoiFile`.

---

AreTomo2, a multi-GPU accelerated software package that fully automates motion-corrected marker-free tomographic alignment and reconstruction, now includes robust GPU-accelerated CTF (Contrast Transfer Function) estimation in a single package. AreTomo2  is part of our endeavor to build a fully-automated high-throughput processing pipeline that enables real-time reconstruction of tomograms in parallel with tomographic data collection. It strives to be fast and accurate, as well as provides for easy integration into subtomogram processing workflows by generating IMod compatible files containing alignment and CTF parameters needed to bootstrap subtomogram averaging. AreTomo2 can also be used for on-the-fly reconstruction of tomograms and CTF estimation in parallel with tilt series collection, enabling real-time assessment of sample quality and adjustment of collection parameters.

![ReadmeImg](https://github.com/czimaginginstitute/AreTomo2/blob/main/docs/ReadmeImg.png)

An example of AreTomo2 reconstructed tomogram. For more details, please refer to “AreTomo: An integrated software package for automated marker-free, motion-corrected cryo-electron tomographic alignment and reconstruction”, J. Struct Biology:  X Vol 6, 2022

## Installation
AreTomo2 is developed on Linux platform equipped with at least one Nvidia GPU card. To compile from the source, follow the steps below:

1.	git clone https://github.com/czimaginginstitute/AreTomo2.git
2.	cd AreTomo2 
3.	make exe -f makefile11 [CUDAHOME=path/cuda-xx.x]

If the compute capability of GPUs is 5.x, use makefile instead. If CUDAHOME is not provided, the default installation path of CUDA given in makefile or makefile11 will be used.

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## Reporting Security Issues

If you believe you have found a security issue, please responsibly disclose by contacting us at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com).
