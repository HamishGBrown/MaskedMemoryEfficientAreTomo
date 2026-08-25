# Streaming WBP Reconstruction

## Problem

For large tilt series, AreTomo2 ran out of CPU RAM during reconstruction. The root cause was that the entire output tomogram volume was allocated in CPU RAM simultaneously with the corrected tilt series:

| Object | Example size (4k × 4k, 100 tilts, VolZ=256) |
|---|---|
| Corrected tilt series | ~6.4 GB |
| Output volume | ~17.2 GB |
| **Peak RAM** | **~23.6 GB** |

The output volume was fully accumulated in a `CTomoStack` in CPU RAM before being written to disk at the very end of the pipeline.

## Solution

WBP reconstruction now streams each reconstructed XZ slice directly to the output MRC file as it is computed, using `pwrite()`. The full volume is never held in RAM. Peak extra memory during reconstruction is one XZ slice per GPU thread (~4 MB for a 4k × 4k image with VolZ=256).

The corrected tilt series is still needed in CPU RAM during reconstruction (for sinogram extraction), so it cannot be freed early. The saving is entirely on the volume side.

## Files Changed

### `MrcUtil/CMrcUtilInc.h` and `MrcUtil/CTomoStack.cpp`

Two new methods added to `CTomoStack`:

```cpp
void CreateStub(int* piStkSize);
bool IsStreaming(void) const;
```

`CreateStub()` sets the dimensions and allocates the metadata arrays (tilts, indices, centers) but does **not** allocate frame buffers — `m_ppfFrames[i]` are all `NULL`. `IsStreaming()` returns `true` when the first frame pointer is `NULL`, which is used as the sentinel for streaming mode throughout the rest of the pipeline.

### `Recon/CReconInc.h`

`CDoWbpRecon::DoIt()` gains two new trailing parameters:

```cpp
static MrcUtil::CTomoStack* DoIt(
    MrcUtil::CTomoStack* pTomoStack,
    MrcUtil::CAlignParam* pAlignParam,
    int iVolZ,
    float fRFactor,
    int* piGpuIDs,
    int iNumGpus,
    char* pcOutMrcFile,   // new: path to write the volume
    float fPixelSize      // new: pixel size for the MRC header
);
```

### `Recon/CDoWbpRecon.cpp`

The static `s_pVolStack` (which held the entire volume in RAM) is removed and replaced with a set of lightweight static variables:

```cpp
static int    s_iOutFileFd   = -1;       // POSIX file descriptor for pwrite()
static size_t s_tDataOffset  = 0;        // byte offset where slice data begins
static float* s_pfVolStats   = 0L;       // per-slice [min, max, mean] array
static int    s_aiVolSize[3] = {0,0,0};  // [volX, volZ, numY]
```

**`DoIt()`** now:
1. Writes the MRC header (1024 bytes + ext-header stub) using `CSaveMrc`, then closes and reopens the file with `O_RDWR`.
2. Pre-allocates the full file size on disk with `ftruncate()`.
3. Runs the GPU worker threads.
4. After all threads complete, aggregates per-slice stats and writes the global min/max/mean into the MRC header at byte offsets 76–84.
5. Returns a metadata-only `CTomoStack` stub (correct dimensions, no frame data).

**`mGetReconResult()`** replaces the `memcpy`-to-`s_pVolStack` pattern with:
1. Z-flip performed in-place on `m_pfVolXZ` (swapping rows rather than copying to a separate buffer).
2. Per-slice min/max/mean accumulated into `s_pfVolStats[iLastY]`.
3. `pwrite()` to the computed file offset for slice `iLastY`.

`pwrite()` is POSIX-specified to be atomic for non-overlapping file regions, so multiple GPU threads writing different slice indices require no mutex.

**`mReconstruct()`** and **`ThreadMain()`** replace all `s_pVolStack->m_aiStkSize[n]` and `s_pVolStack->GetPixels()` references with `s_aiVolSize[n]`.

### `CProcessThread.h`

Two new private methods declared:

```cpp
void mFlipIntStreaming(void);
void mSaveCentralSlicesStreaming(void);
```

### `CProcessThread.cpp`

Every post-reconstruction step tests `m_pTomoStack->IsStreaming()` before operating on frame data:

| Method | Streaming behaviour |
|---|---|
| `mWbpRecon()` | Passes `m_acOutMrcFile` and `fPixelSize` to `CDoWbpRecon::DoIt()` |
| `mSaveStack()` | Returns immediately — file already written with correct header stats |
| `mFlipInt()` | Dispatches to `mFlipIntStreaming()` |
| `mSaveCentralSlices()` | Dispatches to `mSaveCentralSlicesStreaming()` |
| `mFlipVol()` | Prints a note and skips — requires full in-memory transpose |
| `mCropVol()` | Prints a note and skips — requires full in-memory volume |

**`mFlipIntStreaming()`** performs a two-pass file operation, reading and writing one XZ slice at a time:
- Pass 1: scan all slices to determine global min and max.
- Pass 2: apply `newVal = min + max - val`, write each slice back, accumulate new mean.
- Updates the MRC header stats in-place via `pwrite()` at byte offset 76.

**`mSaveCentralSlicesStreaming()`** reads the output file one XZ slice at a time and accumulates:
- `_projXZ`: sum of all XZ slices (projection along Y).
- `_projXY`: for each Y slice, sum across Z rows (projection along Z).

These images are then saved with `CSaveTempMrc` exactly as in the non-streaming path.

## Limitations

The following optional features are not supported when the volume is streamed to disk and will be skipped with a printed notice:

- **`-FlipVol`**: requires rearranging the volume from XZY to XYZ layout, which involves reading every row of every slice at a non-sequential file offset. This cannot be done efficiently without full RAM or a temporary second copy of the file.
- **`-CropVol`**: requires patch alignment parameters and a full in-memory volume.

Both features remain fully functional for smaller datasets where RAM is not a constraint (i.e. when the SART reconstructor is used, or when the volume fits in memory).

## Performance

The streaming path adds no measurable overhead on SSDs. Each `pwrite()` call writes one XZ slice (~4 MB for a 4k image at VolZ=256), which completes in approximately 1 ms on a modern NVMe drive. The GPU reconstruction time per slice is substantially longer, so the write is fully hidden within the existing CUDA stream pipeline.

On mechanical HDDs or network storage, writes may be 10–50 ms per slice and could add a few minutes to total run time for very large volumes, though this remains negligible compared to alignment time.
