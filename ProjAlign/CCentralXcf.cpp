#include "CProjAlignInc.h"
#include "../Util/CUtilInc.h"
#include <memory.h>
#include <stdio.h>

using namespace ProjAlign;

CCentralXcf::CCentralXcf(void)
{
	m_gfPadRef  = 0L;
	m_gfPadImg  = 0L;
	m_gfPadBuf  = 0L;
	m_gfPadMask = 0L;
	m_gfCmpMask = 0L;
	m_gfPadRef2 = 0L;
	m_gfPadD1   = 0L;
	m_gfPadC    = 0L;
	m_pfMnccImg = 0L;
	m_iVolZ     = 0;
	m_fTilt     = 0.0f;
	m_fPower    = 0.5f;
	m_fBFactor  = 300.0f;
	m_bHasMask  = false;
	m_fSumG     = 0.0f;
}

CCentralXcf::~CCentralXcf(void)
{
	this->Clean();
}

void CCentralXcf::Clean(void)
{
	if(m_gfPadRef  != 0L) cudaFree(m_gfPadRef);
	if(m_gfPadImg  != 0L) cudaFree(m_gfPadImg);
	if(m_gfPadBuf  != 0L) cudaFree(m_gfPadBuf);
	if(m_gfPadMask != 0L) cudaFree(m_gfPadMask);
	if(m_gfCmpMask != 0L) cudaFree(m_gfCmpMask);
	if(m_gfPadRef2 != 0L) cudaFree(m_gfPadRef2);
	if(m_gfPadD1   != 0L) cudaFree(m_gfPadD1);
	if(m_gfPadC    != 0L) cudaFree(m_gfPadC);
	if(m_pfMnccImg != 0L) cudaFreeHost(m_pfMnccImg);
	m_gfPadRef  = 0L;
	m_gfPadImg  = 0L;
	m_gfPadBuf  = 0L;
	m_gfPadMask = 0L;
	m_gfCmpMask = 0L;
	m_gfPadRef2 = 0L;
	m_gfPadD1   = 0L;
	m_gfPadC    = 0L;
	m_pfMnccImg = 0L;
	m_bHasMask  = false;
	m_fSumG     = 0.0f;
	//--------------
	m_fft2D.DestroyPlan();
	m_fft2DInv.DestroyPlan();
	m_projXcf.Clean();
	m_aMncc.Clean();
}

void CCentralXcf::SetupXcf(float fPower, float fBFactor)
{
	m_fPower = fPower;
	m_fBFactor = fBFactor;
}

void CCentralXcf::Setup(int* piImgSize, int iVolZ)
{
	this->Clean();
	m_aiImgSize[0] = piImgSize[0];
	m_aiImgSize[1] = piImgSize[1];
	m_iVolZ = iVolZ;
	//--------------
	m_aiCentSize[0] = m_aiImgSize[0];
        m_aiCentSize[1] = m_aiImgSize[1];
	m_aiPadSize[0] = (m_aiCentSize[0] / 2 + 1) * 2;
	m_aiPadSize[1] = m_aiCentSize[1];
	//-------------------------------
        size_t tBytes = m_aiPadSize[0] * m_aiPadSize[1] * sizeof(float);
        cudaMalloc(&m_gfPadRef,  tBytes);
	cudaMalloc(&m_gfPadImg,  tBytes);
	cudaMalloc(&m_gfPadBuf,  tBytes);
	cudaMalloc(&m_gfPadMask, tBytes);
	cudaMalloc(&m_gfPadRef2, tBytes);
	cudaMalloc(&m_gfPadD1,   tBytes);
	cudaMalloc(&m_gfPadC,    tBytes);
	size_t tCmpBytes = sizeof(cufftComplex)
	   * (m_aiPadSize[0]/2) * m_aiPadSize[1];
	cudaMalloc(&m_gfCmpMask, tCmpBytes);
	cudaMallocHost(&m_pfMnccImg,
	   sizeof(float) * m_aiCentSize[0] * m_aiCentSize[1]);
	//------------------------------
	bool bForward = true;
	int aiCmpSize[] = {m_aiPadSize[0]/2, m_aiPadSize[1]};
	m_projXcf.Setup(aiCmpSize);
	m_fft2D.CreatePlan(m_aiCentSize, bForward);
	m_fft2DInv.CreatePlan(m_aiCentSize, !bForward);
	m_aMncc.Setup(m_aiPadSize[0] * m_aiPadSize[1]);
}

void CCentralXcf::DoIt
(	float* pfRef,
	float* pfImg,
	float fTilt
)
{	m_fTilt = fTilt;
	//--------------
	mGetCentral(pfRef, m_gfPadRef);
	mGetCentral(pfImg, m_gfPadImg);
	//-----------------------------
	mNormalize(m_gfPadRef);
	mNormalize(m_gfPadImg);
	//---------------------
	if(m_bHasMask) mCorrelateMasked();
	else           mCorrelate();
}

void CCentralXcf::GetShift(float* pfShift)
{
	pfShift[0] = m_afShift[0];
	pfShift[1] = m_afShift[1];
}

void CCentralXcf::mGetCentral(float* pfImg, float* gfPadImg)
{
	size_t tBytes = sizeof(float) * m_aiCentSize[0];
	int iX = (m_aiImgSize[0] - m_aiCentSize[0]) / 2;
	int iY = (m_aiImgSize[1] - m_aiCentSize[1]) / 2;
	int iOffset = iY * m_aiImgSize[0] + iX;
	//-------------------------------------
	for(int y=0; y<m_aiCentSize[1]; y++)
	{	float* pfSrc = pfImg + y * m_aiImgSize[0] + iOffset;
		float* gfDst = gfPadImg + y * m_aiPadSize[0];
		cudaMemcpy(gfDst, pfSrc, tBytes, cudaMemcpyDefault);
	}
}

void CCentralXcf::mNormalize(float* gfPadImg)
{
	bool bPadded = true;
	Util::GCalcMoment2D aGCalcMoment2D;
	aGCalcMoment2D.SetSize(m_aiPadSize, bPadded);
	float fMean = aGCalcMoment2D.DoIt(gfPadImg, 1, true);
	//---------------------------------------------------
	Util::GNormalize2D aGNorm2D;
	aGNorm2D.DoIt(gfPadImg, m_aiPadSize, bPadded, fMean, 1.0f);
	//---------------------------------------------------------
	CParam* pParam = CParam::GetInstance();
	float afCent[] = {0.0f, 0.0f};
	float afMaskSize[] = {0.0f, 0.0f}; 
	afCent[0] = m_aiCentSize[0] * 0.5f;
	afCent[1] = m_aiCentSize[1] * 0.5f;
	afMaskSize[0] = m_aiCentSize[0] * pParam->m_afMaskSize[0];
	afMaskSize[1] = m_aiCentSize[1] * pParam->m_afMaskSize[1];
	//--------------------------------------------------------
	Util::GRoundEdge aGRoundEdge;
	float fPower = 4.0f;
	aGRoundEdge.DoIt(gfPadImg, m_aiPadSize, bPadded, fPower);	
}

void CCentralXcf::mCorrelate(void)
{
	bool bNorm = true;
	m_fft2D.Forward(m_gfPadRef, !bNorm);
	m_fft2D.Forward(m_gfPadImg, !bNorm);
	//----------------------------------
	cufftComplex* gRefCmp = (cufftComplex*)m_gfPadRef;
	cufftComplex* gImgCmp = (cufftComplex*)m_gfPadImg;
	m_projXcf.DoIt(gRefCmp, gImgCmp, m_fBFactor, m_fPower);
	//-----------------------------------------------------
	bool bClean = true;
	m_projXcf.SearchPeak();
	m_projXcf.GetShift(m_afShift, 1.0f);
}

//-------------------------------------------------------------------
// Load a binned mask into GPU memory and precompute FFT(G).
// pfBinnedMask is a CPU float array of size m_aiCentSize[0]*[1].
//-------------------------------------------------------------------
void CCentralXcf::SetMask(float* pfBinnedMask)
{
	m_bHasMask = false;
	if(!pfBinnedMask) return;
	//------------------------
	// sum(G): needed for the local-mean-correction terms in
	// mCorrelateMasked (Padfield eq. 21). The mask is fixed once
	// loaded, so this is computed here rather than per-projection.
	double dSumG = 0.0;
	for(int i=0; i<m_aiCentSize[0]*m_aiCentSize[1]; i++) dSumG += pfBinnedMask[i];
	m_fSumG = (float)dSumG;
	//------------------------
	// Copy mask to GPU with same padded row layout as mGetCentral.
	size_t tRowBytes = sizeof(float) * m_aiCentSize[0];
	size_t tPadBytes = sizeof(float) * m_aiPadSize[0] * m_aiPadSize[1];
	cudaMemset(m_gfPadMask, 0, tPadBytes);
	for(int y=0; y<m_aiCentSize[1]; y++)
	{	float* pfSrc = pfBinnedMask + y * m_aiCentSize[0];
		float* gfDst = m_gfPadMask  + y * m_aiPadSize[0];
		cudaMemcpy(gfDst, pfSrc, tRowBytes, cudaMemcpyDefault);
	}
	//-------------------------------------------------------
	// Precompute FFT(G): forward FFT via m_gfPadBuf as temp
	// so m_gfPadMask is preserved for ApplyMask / MaskedSumSq.
	cudaMemcpy(m_gfPadBuf, m_gfPadMask, tPadBytes, cudaMemcpyDeviceToDevice);
	m_fft2D.Forward(m_gfPadBuf, false);
	size_t tCmpBytes = sizeof(cufftComplex)
	   * (m_aiPadSize[0]/2) * m_aiPadSize[1];
	cudaMemcpy(m_gfCmpMask, m_gfPadBuf, tCmpBytes, cudaMemcpyDeviceToDevice);
	m_bHasMask = true;
}

//-------------------------------------------------------------------
// Masked Normalised Cross-Correlation (Padfield 2012, eq. 21),
// single-mask case: ref is fully valid (m1==1); G is the mask on
// img (set by SetMask).
//
//   c(t)    = IFFT( conj(FFT(ref))   x FFT(G)     )
//   num(t)  = IFFT( conj(FFT(ref))   x FFT(img*G) ) - c(t) * Simg/Sg
//   d1(t)   = IFFT( conj(FFT(ref²))  x FFT(G)     ) - c(t)² / Sg
//   d2      = sum(img² * G) - Simg² / Sg
//   MNCC(t) = num(t) / sqrt( |d1(t)| * d2 + eps )
//
// where Sg = sum(G) (precomputed once in SetMask) and Simg = sum(img*G).
// The subtracted terms remove the influence of the local mean of ref
// and img under the (shifted) mask footprint. Without them, the score
// is only a raw energy correlation, biased whenever ref/img have a
// non-zero local mean under the mask -- the common case for real
// projections, and precisely when masking is needed most.
//-------------------------------------------------------------------
void CCentralXcf::mCorrelateMasked(void)
{
	int iPadPix = m_aiPadSize[0] * m_aiPadSize[1];
	int iCmpPix = (m_aiPadSize[0]/2) * m_aiPadSize[1];
	bool bHaveSumG = (m_fSumG > 1e-6f);
	//--------------------------------------------------
	// Simg = sum(img*G), d2 = sum(img²*G), before masking img.
	float fSimgG = m_aMncc.MaskedSum(m_gfPadImg, m_gfPadMask, iPadPix);
	float fD2    = m_aMncc.MaskedSumSq(m_gfPadImg, m_gfPadMask, iPadPix);
	if(bHaveSumG) fD2 -= (fSimgG * fSimgG) / m_fSumG;
	if(fD2 < 0.0f) fD2 = 0.0f;
	//----------------------------------------------
	// ref² → m_gfPadRef2
	m_aMncc.Square(m_gfPadRef, m_gfPadRef2, iPadPix);
	//------------------------------------------------
	// img → img * G
	m_aMncc.ApplyMask(m_gfPadImg, m_gfPadMask, iPadPix);
	//----------------------------------------------
	// Forward FFT of ref, img*G, ref²
	m_fft2D.Forward(m_gfPadRef,  false);
	m_fft2D.Forward(m_gfPadImg,  false);
	m_fft2D.Forward(m_gfPadRef2, false);
	//--------------------------------------------------
	// num spectrum: conj(FFT(ref)) x FFT(img*G) → m_gfPadBuf
	m_aMncc.CrossMultiply
	(	(cufftComplex*)m_gfPadRef,
		(cufftComplex*)m_gfPadImg,
		(cufftComplex*)m_gfPadBuf,
		iCmpPix
	);
	// Explicit sync + error check after each cross-multiply: without
	// this, mFindPeakMncc intermittently segfaults on a corrupted
	// device pointer (see .claude/sessions/2026-08-26-0f860083.md) --
	// root cause not yet identified, this is a confirmed workaround.
	cudaDeviceSynchronize();
	cudaError_t eErr1 = cudaGetLastError();
	if(eErr1 != cudaSuccess) fprintf(stderr,
	   "CrossMultiply (num) error: %s\n", cudaGetErrorString(eErr1));
	// d1 spectrum: conj(FFT(ref²)) x FFT(G) → m_gfPadD1
	m_aMncc.CrossMultiply
	(	(cufftComplex*)m_gfPadRef2,
		m_gfCmpMask,
		(cufftComplex*)m_gfPadD1,
		iCmpPix
	);
	cudaDeviceSynchronize();
	cudaError_t eErr2 = cudaGetLastError();
	if(eErr2 != cudaSuccess) fprintf(stderr,
	   "CrossMultiply (d1) error: %s\n", cudaGetErrorString(eErr2));
	// c spectrum: conj(FFT(ref)) x FFT(G) → m_gfPadC
	m_aMncc.CrossMultiply
	(	(cufftComplex*)m_gfPadRef,
		m_gfCmpMask,
		(cufftComplex*)m_gfPadC,
		iCmpPix
	);
	cudaDeviceSynchronize();
	cudaError_t eErr3 = cudaGetLastError();
	if(eErr3 != cudaSuccess) fprintf(stderr,
	   "CrossMultiply (c) error: %s\n", cudaGetErrorString(eErr3));
	//----------------------------------------------
	// Inverse FFT: spectra → real maps
	m_fft2DInv.Inverse((cufftComplex*)m_gfPadBuf);
	m_fft2DInv.Inverse((cufftComplex*)m_gfPadD1);
	m_fft2DInv.Inverse((cufftComplex*)m_gfPadC);
	//----------------------------------------------
	// Subtract the local-mean cross terms using c(t).
	if(bHaveSumG)
	{	float fNumFactor = fSimgG / m_fSumG;
		// c(t) is a raw (unnormalized) IFFT output carrying one factor
		// of N = m_aiCentSize[0]*m_aiCentSize[1] (cufft round-trips
		// forward+inverse without dividing by N anywhere -- see
		// GFFT2D::Forward/Inverse). Squaring c(t) below doubles that to
		// N^2, while gfD1[i] itself only carries a single N, so the
		// correction needs an extra 1/N to match scale.
		float fInvSumG   = 1.0f /
		   (m_fSumG * (float)(m_aiCentSize[0] * m_aiCentSize[1]));
		m_aMncc.CorrectMaps
		(	m_gfPadBuf, m_gfPadD1, m_gfPadC,
			fNumFactor, fInvSumG, iPadPix
		);
	}
	//----------------------------------------------
	// Pointwise normalisation: MNCC = num / sqrt(|d1|*d2 + eps)
	float fEps = (fD2 > 0.0f) ? (1e-6f * fD2) : 1e-10f;
	m_aMncc.MnccFinalize(m_gfPadBuf, m_gfPadD1, fD2, fEps, iPadPix);
	//---------------------------------------------------
	mFindPeakMncc();
}

//-------------------------------------------------------------------
// Copy MNCC result (de-padded) from GPU to CPU and locate the peak.
// The peak pixel position is converted to a shift with FFT wrap-around.
// Sub-pixel accuracy is obtained via parabolic interpolation with
// wrap-around neighbours (the FFT output is cyclic).
//-------------------------------------------------------------------
void CCentralXcf::mFindPeakMncc(void)
{
	int iCentX = m_aiCentSize[0];
	int iCentY = m_aiCentSize[1];
	int iPadX  = m_aiPadSize[0];
	// Strided 2D copy: strip the 2 padding floats at end of each row.
	cudaMemcpy2D
	(	m_pfMnccImg, iCentX * sizeof(float),
		m_gfPadBuf,  iPadX  * sizeof(float),
		iCentX * sizeof(float), iCentY,
		cudaMemcpyDefault
	);
	// Integer peak search
	float fPeak = (float)-1e20;
	int iPeakX = 0, iPeakY = 0;
	for(int y=0; y<iCentY; y++)
	{	float* pfRow = m_pfMnccImg + y * iCentX;
		for(int x=0; x<iCentX; x++)
		{	if(pfRow[x] > fPeak)
			{	fPeak  = pfRow[x];
				iPeakX = x;
				iPeakY = y;
			}
		}
	}
	// Sub-pixel parabolic refinement using wrap-around neighbours
	// (the MNCC map is cyclic: shift N-1 is adjacent to shift 0).
	int ic = iPeakY * iCentX + iPeakX;
	int xp = (iPeakX < iCentX-1) ? ic + 1 : iPeakY * iCentX;
	int xm = (iPeakX > 0)        ? ic - 1 : iPeakY * iCentX + iCentX - 1;
	int yp = (iPeakY < iCentY-1) ? ic + iCentX : iPeakX;
	int ym = (iPeakY > 0)        ? ic - iCentX : (iCentY-1) * iCentX + iPeakX;
	double a  = (m_pfMnccImg[xp] + m_pfMnccImg[xm]) * 0.5 - m_pfMnccImg[ic];
	double b  = (m_pfMnccImg[xp] - m_pfMnccImg[xm]) * 0.5;
	double c  = (m_pfMnccImg[yp] + m_pfMnccImg[ym]) * 0.5 - m_pfMnccImg[ic];
	double d  = (m_pfMnccImg[yp] - m_pfMnccImg[ym]) * 0.5;
	double dSubX = (fabs(a) > 1e-30) ? -b / (2.0 * a) : 0.0;
	double dSubY = (fabs(c) > 1e-30) ? -d / (2.0 * c) : 0.0;
	if(fabs(dSubX) > 1.0) dSubX = 0.0;
	if(fabs(dSubY) > 1.0) dSubY = 0.0;
	double dPeakX = iPeakX + dSubX;
	double dPeakY = iPeakY + dSubY;
	// Convert to shift with FFT circular wrap-around
	m_afShift[0] = (float)((dPeakX > iCentX * 0.5) ? (dPeakX - iCentX) : dPeakX);
	m_afShift[1] = (float)((dPeakY > iCentY * 0.5) ? (dPeakY - iCentY) : dPeakY);
}
