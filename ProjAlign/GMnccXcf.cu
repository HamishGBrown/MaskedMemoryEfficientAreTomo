#include "CProjAlignInc.h"
#include <stdio.h>

using namespace ProjAlign;

//------------------------------------------------------
// Apply mask in-place: gfImg[i] *= gfMask[i]
//------------------------------------------------------
static __global__ void mGApplyMask(
	float* gfImg, const float* gfMask, int iPixels)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < iPixels) gfImg[i] *= gfMask[i];
}

//------------------------------------------------------
// Square: gfOut[i] = gfIn[i]^2
//------------------------------------------------------
static __global__ void mGSquare(
	const float* gfIn, float* gfOut, int iPixels)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < iPixels) gfOut[i] = gfIn[i] * gfIn[i];
}

//------------------------------------------------------
// Complex cross-multiply: gcOut[i] = conj(gcA[i]) * gcB[i]
// Computes the cross-correlation spectrum: XCF = IFFT(conj(FFT(A)) * FFT(B))
//------------------------------------------------------
static __global__ void mGCrossMultiply(
	const cufftComplex* gcA,
	const cufftComplex* gcB,
	cufftComplex* gcOut,
	int iCmpPixels)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iCmpPixels) return;
	float aR =  gcA[i].x;
	float aI = -gcA[i].y;   // conjugate
	float bR =  gcB[i].x;
	float bI =  gcB[i].y;
	gcOut[i].x = aR * bR - aI * bI;
	gcOut[i].y = aR * bI + aI * bR;
}

//------------------------------------------------------
// Finalize MNCC pointwise:
//   gfNum[i] = gfNum[i] / sqrt(|gfD1[i]| * fD2 + fEps)
//------------------------------------------------------
static __global__ void mGMnccFinalize(
	float* gfNum,
	const float* gfD1,
	float fD2, float fEps,
	int iPixels)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iPixels) return;
	float fDenom = sqrtf(fabsf(gfD1[i]) * fD2 + fEps);
	gfNum[i] = (fDenom > 0.0f) ? (gfNum[i] / fDenom) : 0.0f;
}

//------------------------------------------------------
// Parallel reduction: sum of gfImg[i]^2 * gfMask[i]
// Called before ApplyMask so gfImg is still unmasked.
// Each block reduces blockDim.x elements into one partial sum.
//------------------------------------------------------
static __global__ void mGMaskedSumSq(
	const float* gfImg,
	const float* gfMask,
	float* gfPartial,
	int iPixels)
{
	extern __shared__ float s_data[];
	int tid = threadIdx.x;
	int i   = blockIdx.x * blockDim.x + tid;
	float v = (i < iPixels) ? (gfImg[i] * gfImg[i] * gfMask[i]) : 0.0f;
	s_data[tid] = v;
	__syncthreads();
	for(int s = blockDim.x >> 1; s > 0; s >>= 1)
	{	if(tid < s) s_data[tid] += s_data[tid + s];
		__syncthreads();
	}
	if(tid == 0) gfPartial[blockIdx.x] = s_data[0];
}

//------------------------------------------------------
// Parallel reduction: sum of gfImg[i] * gfMask[i] (unsquared).
// Called before ApplyMask so gfImg is still unmasked.
//------------------------------------------------------
static __global__ void mGMaskedSum(
	const float* gfImg,
	const float* gfMask,
	float* gfPartial,
	int iPixels)
{
	extern __shared__ float s_data[];
	int tid = threadIdx.x;
	int i   = blockIdx.x * blockDim.x + tid;
	float v = (i < iPixels) ? (gfImg[i] * gfMask[i]) : 0.0f;
	s_data[tid] = v;
	__syncthreads();
	for(int s = blockDim.x >> 1; s > 0; s >>= 1)
	{	if(tid < s) s_data[tid] += s_data[tid + s];
		__syncthreads();
	}
	if(tid == 0) gfPartial[blockIdx.x] = s_data[0];
}

//------------------------------------------------------
// Subtract the local-mean cross terms (Padfield masked NCC, eq. 21),
// using the precomputed correlate(ref, mask) map gfC:
//   gfNum[i] -= gfC[i] * fNumFactor      (fNumFactor = sum(img*G)/sum(G))
//   gfD1[i]  -= gfC[i]^2 * fInvSumG      (fInvSumG   = 1/sum(G))
//------------------------------------------------------
static __global__ void mGCorrectMaps(
	float* gfNum,
	float* gfD1,
	const float* gfC,
	float fNumFactor, float fInvSumG,
	int iPixels)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iPixels) return;
	float c = gfC[i];
	gfNum[i] -= c * fNumFactor;
	gfD1[i]  -= c * c * fInvSumG;
}

//----------------------------------------------------------------------

GMnccXcf::GMnccXcf(void)
{
	m_gfPartial  = 0L;
	m_pfPartial  = 0L;
	m_iNumBlocks = 0;
}

GMnccXcf::~GMnccXcf(void)
{
	this->Clean();
}

void GMnccXcf::Clean(void)
{
	if(m_gfPartial != 0L) cudaFree(m_gfPartial);
	if(m_pfPartial != 0L) cudaFreeHost(m_pfPartial);
	m_gfPartial  = 0L;
	m_pfPartial  = 0L;
	m_iNumBlocks = 0;
}

void GMnccXcf::Setup(int iPixels)
{
	this->Clean();
	m_iNumBlocks = (iPixels + 255) / 256;
	cudaMalloc(&m_gfPartial, m_iNumBlocks * sizeof(float));
	cudaMallocHost(&m_pfPartial, m_iNumBlocks * sizeof(float));
}

void GMnccXcf::ApplyMask(float* gfImg, const float* gfMask, int iPixels)
{
	dim3 blk(256), grd((iPixels + 255) / 256);
	mGApplyMask<<<grd, blk>>>(gfImg, gfMask, iPixels);
}

void GMnccXcf::Square(const float* gfIn, float* gfOut, int iPixels)
{
	dim3 blk(256), grd((iPixels + 255) / 256);
	mGSquare<<<grd, blk>>>(gfIn, gfOut, iPixels);
}

void GMnccXcf::CrossMultiply(
	const cufftComplex* gcA,
	const cufftComplex* gcB,
	cufftComplex* gcOut,
	int iCmpPixels)
{
	dim3 blk(256), grd((iCmpPixels + 255) / 256);
	mGCrossMultiply<<<grd, blk>>>(gcA, gcB, gcOut, iCmpPixels);
}

void GMnccXcf::MnccFinalize(
	float* gfNum, const float* gfD1,
	float fD2, float fEps, int iPixels)
{
	dim3 blk(256), grd((iPixels + 255) / 256);
	mGMnccFinalize<<<grd, blk>>>(gfNum, gfD1, fD2, fEps, iPixels);
}

float GMnccXcf::MaskedSumSq(const float* gfImg, const float* gfMask, int iPixels)
{
	int iBlockSize = 256;
	int iNumBlocks = (iPixels + iBlockSize - 1) / iBlockSize;
	if(iNumBlocks > m_iNumBlocks) iNumBlocks = m_iNumBlocks;
	size_t tShared = iBlockSize * sizeof(float);
	mGMaskedSumSq<<<iNumBlocks, iBlockSize, tShared>>>(
		gfImg, gfMask, m_gfPartial, iPixels);
	cudaMemcpy(m_pfPartial, m_gfPartial,
		iNumBlocks * sizeof(float), cudaMemcpyDefault);
	//----------------------------------
	double dSum = 0.0;
	for(int i=0; i<iNumBlocks; i++) dSum += m_pfPartial[i];
	return (float)dSum;
}

float GMnccXcf::MaskedSum(const float* gfImg, const float* gfMask, int iPixels)
{
	int iBlockSize = 256;
	int iNumBlocks = (iPixels + iBlockSize - 1) / iBlockSize;
	if(iNumBlocks > m_iNumBlocks) iNumBlocks = m_iNumBlocks;
	size_t tShared = iBlockSize * sizeof(float);
	mGMaskedSum<<<iNumBlocks, iBlockSize, tShared>>>(
		gfImg, gfMask, m_gfPartial, iPixels);
	cudaMemcpy(m_pfPartial, m_gfPartial,
		iNumBlocks * sizeof(float), cudaMemcpyDefault);
	//----------------------------------
	double dSum = 0.0;
	for(int i=0; i<iNumBlocks; i++) dSum += m_pfPartial[i];
	return (float)dSum;
}

void GMnccXcf::CorrectMaps(
	float* gfNum, float* gfD1, const float* gfC,
	float fNumFactor, float fInvSumG, int iPixels)
{
	dim3 blk(256), grd((iPixels + 255) / 256);
	mGCorrectMaps<<<grd, blk>>>(gfNum, gfD1, gfC, fNumFactor, fInvSumG, iPixels);
}
