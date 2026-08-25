#include "CReconInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <memory.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

using namespace Recon;

static MrcUtil::CTomoStack* s_pTomoStack = 0L;
static MrcUtil::CAlignParam* s_pAlignParam = 0L;
static int s_iStartTilt = 0;
static int s_iNumTilts = 0;
static int s_iNumIters = 1;
static int s_iNumSubsets = 1;

static int    s_iOutFileFd  = -1;
static size_t s_tDataOffset = 0;
static float* s_pfVolStats  = 0L;      // per-slice stats: [min | max | mean] × numY
static int    s_aiVolSize[3] = {0,0,0}; // [outVolX, volZ, outNumY] – output/on-disk dims
static int    s_iFullVolX = 0;         // full (uncropped) reconstruction width
static int    s_iOutX0 = 0;            // column offset of the ROI within the full width
static int    s_iOutY0 = 0;            // row offset of the ROI within the full stack

MrcUtil::CTomoStack* CDoSartRecon::DoIt
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam,
	int iStartTilt,
	int iNumTilts,
	int iVolZ,
	int iIterations,
	int iNumSubsets,
	int* piGpuIDs,
	int iNumGpus,
	char* pcOutMrcFile,
	float fPixelSize,
	int* piOutRoi
)
{	s_pTomoStack = pTomoStack;
	s_pAlignParam = pAlignParam;
	s_iStartTilt = iStartTilt;
	s_iNumTilts = iNumTilts;
	s_iNumIters = iIterations;
	s_iNumSubsets = iNumSubsets;
	//--------------------------
	// The volume is always reconstructed at full width/height internally;
	// an ROI (piOutRoi = xmin, xmax, ymin, ymax) only restricts what gets
	// written to disk. This is essential for SART: its forward-projection
	// step (GForProj) needs the full-width volume estimate to compute a
	// correct residual against the full-width measured projections —
	// narrowing the reconstruction itself would corrupt the ROI with
	// truncation artifacts. Y rows outside the ROI are skipped entirely
	// (each row is an independent 2D reconstruction), while X columns
	// outside the ROI are simply not copied into the output buffer in
	// mGetReconResult().
	//--------------------------
	s_iFullVolX = pTomoStack->m_aiStkSize[0] / 2 * 2;
	int iFullNumY = pTomoStack->m_aiStkSize[1];
	//--------------------------
	int iX0 = 0, iX1 = s_iFullVolX;
	int iY0 = 0, iY1 = iFullNumY;
	if(piOutRoi != 0L)
	{	if(piOutRoi[1] > piOutRoi[0] && piOutRoi[0] >= 0)
		{	iX0 = piOutRoi[0];
			iX1 = piOutRoi[1];
			if(iX1 > s_iFullVolX) iX1 = s_iFullVolX;
		}
		if(piOutRoi[3] > piOutRoi[2] && piOutRoi[2] >= 0)
		{	iY0 = piOutRoi[2];
			iY1 = piOutRoi[3];
			if(iY1 > iFullNumY) iY1 = iFullNumY;
		}
	}
	s_iOutX0 = iX0;
	s_iOutY0 = iY0;
	s_aiVolSize[0] = iX1 - iX0;   // cropped output width
	s_aiVolSize[1] = iVolZ;
	s_aiVolSize[2] = iY1 - iY0;   // cropped output numY
	//--------------------------
	// Write MRC header and pre-allocate the output file on disk.
	// Mirrors CDoWbpRecon::DoIt — no full volume is held in RAM.
	//--------------------------
	{	Mrc::CSaveMrc aSaveMrc;
		aSaveMrc.OpenFile(pcOutMrcFile);
		aSaveMrc.SetMode(Mrc::eMrcFloat);
		aSaveMrc.SetImgSize(s_aiVolSize, s_aiVolSize[2], 1, fPixelSize);
		aSaveMrc.SetExtHeader(0, 32, 0);
		aSaveMrc.SaveMinMaxMean(0.0f, 0.0f, 0.0f);
		aSaveMrc.m_pSaveMain->DoIt();
	}
	//--------------------------
	s_tDataOffset = 1024 + (size_t)32 * sizeof(float) * s_aiVolSize[2];
	size_t tTotalBytes = s_tDataOffset
		+ (size_t)s_aiVolSize[0] * s_aiVolSize[1] * sizeof(float)
		  * s_aiVolSize[2];
	//--------------------------
	s_iOutFileFd = open(pcOutMrcFile, O_RDWR);
	ftruncate(s_iOutFileFd, (off_t)tTotalBytes);
	//--------------------------
	int iNumY = s_aiVolSize[2];
	s_pfVolStats = new float[iNumY * 3];
	for(int i=0; i<iNumY; i++)
	{	s_pfVolStats[i]         =  1e30f;  // min
		s_pfVolStats[iNumY+i]   = -1e30f;  // max
		s_pfVolStats[iNumY*2+i] =  0.0f;   // mean
	}
	//--------------------------
	Util::CNextItem nextItem;
	nextItem.Create(iNumY);
	CDoSartRecon* pThreads = new CDoSartRecon[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].Run(&nextItem, piGpuIDs[i]);
	}
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
	}
	delete[] pThreads;
	printf("SART reconstruction completed.\n\n");
	//--------------------------
	// Aggregate per-slice stats and update the MRC header in-place.
	float fMin = 1e30f, fMax = -1e30f;
	double dMean = 0.0;
	for(int i=0; i<iNumY; i++)
	{	if(s_pfVolStats[i]       < fMin) fMin = s_pfVolStats[i];
		if(s_pfVolStats[iNumY+i] > fMax) fMax = s_pfVolStats[iNumY+i];
		dMean += s_pfVolStats[iNumY*2+i] / iNumY;
	}
	delete[] s_pfVolStats;
	s_pfVolStats = 0L;
	//--------------------------
	// MRC main-header layout: amin at byte 76, amax at 80, amean at 84.
	float afStats[3] = {fMin, fMax, (float)dMean};
	pwrite(s_iOutFileFd, afStats, sizeof(float)*3, 76);
	close(s_iOutFileFd);
	s_iOutFileFd = -1;
	//--------------------------
	MrcUtil::CTomoStack* pVolStack = new MrcUtil::CTomoStack;
	pVolStack->CreateStub(s_aiVolSize);
	return pVolStack;
}

CDoSartRecon::CDoSartRecon(void)
{
}

CDoSartRecon::~CDoSartRecon(void)
{
	this->Clean();
}

void CDoSartRecon::Clean(void)
{
	CDoBaseRecon::Clean();
	m_aTomoSart.Clean();
}

void CDoSartRecon::ThreadMain(void)
{
	cudaSetDevice(m_iGpuID);
	//----------------------
	int iPadX = (s_iFullVolX / 2 + 1) * 2;
	size_t tBytes = sizeof(float) * iPadX * s_pTomoStack->m_aiStkSize[2];
	cudaMalloc(&m_gfPadSinogram, tBytes);
	cudaMallocHost(&m_pfPadSinogram, tBytes);
	//---------------------------------------
	tBytes = (size_t)s_iFullVolX * s_aiVolSize[1] * sizeof(float);
	cudaMalloc(&m_gfVolXZ, tBytes);
	cudaMemset(m_gfVolXZ, 0, tBytes);
	cudaMallocHost(&m_pfVolXZ, tBytes);
	//---------------------------------
	tBytes = (size_t)s_aiVolSize[0] * s_aiVolSize[1] * sizeof(float);
	cudaMallocHost(&m_pfOutXZ, tBytes);
	//---------------------------------
	m_aTomoSart.Setup
	( s_iFullVolX, s_aiVolSize[1],
	  s_iNumSubsets, s_iNumIters, s_pTomoStack, s_pAlignParam,
	  s_iStartTilt, s_iNumTilts
	);
	//-------------------------
	cudaStreamCreate(&m_stream);
	cudaEventCreate(&m_eventSino);
	//----------------------------
	int iLastY = -1;
	while(true)
	{	int iYLocal = m_pNextItem->GetNext();
		if(iYLocal < 0) break;
		int iY = iYLocal + s_iOutY0;
		//---------------
		if(iYLocal % 101 == 0)
		{	int iLeft = s_aiVolSize[2] - 1 - iYLocal;
			printf("...... reconstruct slice %4d, "
			   "%4d slices left\n", iYLocal+1, iLeft);
		}
		//-------------------------------------------
		mExtractSinogram(iY);
		mGetReconResult(iLastY);
		mReconstruct(iY);
		iLastY = iYLocal;
	}
	cudaStreamSynchronize(m_stream);
	mGetReconResult(iLastY);
	//----------------------
	cudaStreamDestroy(m_stream);
	cudaEventDestroy(m_eventSino);
}

void CDoSartRecon::mExtractSinogram(int iY)
{
	int iProjX = s_pTomoStack->m_aiStkSize[0];
	int iPadX = (iProjX / 2 + 1) * 2;
	size_t tBytes = sizeof(float) * iProjX;
	for(int i=0; i<s_pTomoStack->m_aiStkSize[2]; i++)
	{	float* pfProj = s_pTomoStack->GetFrame(i);
		float* pfSrc = pfProj + iY * iProjX;
		float* pfDst = m_pfPadSinogram + i * iPadX;
		memcpy(pfDst, pfSrc, tBytes);
	}
	//-----------------------------------
	cudaStreamWaitEvent(m_stream, m_eventSino, 0);
	tBytes = sizeof(float) * iPadX * s_pTomoStack->m_aiStkSize[2];
	cudaMemcpyAsync(m_gfPadSinogram, m_pfPadSinogram, tBytes,
		cudaMemcpyDefault, m_stream);
	cudaEventRecord(m_eventSino, m_stream);
}

//-----------------------------------------------------------------------------
// Mirrors CDoWbpRecon::mGetReconResult:
// 1. Flip z in-place to match IMOD handedness.
// 2. Accumulate per-slice statistics.
// 3. Write the slice directly to the output file with pwrite().
//    pwrite() is thread-safe for non-overlapping offsets — no mutex needed.
//-----------------------------------------------------------------------------
void CDoSartRecon::mGetReconResult(int iLastY)
{
	if(iLastY < 0) return;
	cudaStreamSynchronize(m_stream);
	//--------------------------------
	int iSliceX = s_iFullVolX;
	int iSliceZ = s_aiVolSize[1];
	//--------------------------------
	// In-place z-flip to match IMOD convention.
	for(int z=0; z < iSliceZ/2; z++)
	{	float* pfTop = m_pfVolXZ + z * iSliceX;
		float* pfBot = m_pfVolXZ + (iSliceZ - 1 - z) * iSliceX;
		for(int x=0; x<iSliceX; x++)
		{	float fTmp = pfTop[x];
			pfTop[x]   = pfBot[x];
			pfBot[x]   = fTmp;
		}
	}
	//--------------------------------
	// Copy just the ROI columns of each z-row into the compact output
	// buffer that gets written to disk; the reconstruction itself always
	// covers the full width (see DoIt() for why, re: SART correctness).
	int iOutX = s_aiVolSize[0];
	for(int z=0; z<iSliceZ; z++)
	{	memcpy(m_pfOutXZ + z * iOutX,
		   m_pfVolXZ + z * iSliceX + s_iOutX0, iOutX * sizeof(float));
	}
	int iOutPixels = iOutX * iSliceZ;
	//--------------------------------
	float fMin = 1e30f, fMax = -1e30f;
	double dSum = 0.0;
	for(int i=0; i<iOutPixels; i++)
	{	float v = m_pfOutXZ[i];
		if(v < fMin) fMin = v;
		if(v > fMax) fMax = v;
		dSum += v;
	}
	int iNumY = s_aiVolSize[2];
	s_pfVolStats[iLastY]         = fMin;
	s_pfVolStats[iNumY + iLastY] = fMax;
	s_pfVolStats[iNumY*2+iLastY] = (float)(dSum / iOutPixels);
	//--------------------------------
	size_t tSliceBytes = (size_t)iOutPixels * sizeof(float);
	off_t  tOffset     = (off_t)s_tDataOffset + (off_t)iLastY * tSliceBytes;
	pwrite(s_iOutFileFd, m_pfOutXZ, tSliceBytes, tOffset);
}

void CDoSartRecon::mReconstruct(int iY)
{
	size_t tBytes = (size_t)s_iFullVolX * s_aiVolSize[1] * sizeof(float);
	cudaMemsetAsync(m_gfVolXZ, 0, tBytes, m_stream);
	m_aTomoSart.DoIt(m_gfPadSinogram, m_gfVolXZ, m_stream);
	//--------------------------------------------------
	cudaMemcpyAsync(m_pfVolXZ, m_gfVolXZ, tBytes,
		cudaMemcpyDefault, m_stream);
}
