#include "CProjAlignInc.h"
#include "../CInput.h"
#include <Mrcfile/CMrcFileInc.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Util/Util_Time.h>

using namespace ProjAlign;

CProjAlignMain::CProjAlignMain(void)
{
	m_pfReproj = 0L;
	m_pfTiltAngles = 0L;
	m_pbSkipProjs = 0L;
	m_pCorrTomoStack = 0L;
}

CProjAlignMain::~CProjAlignMain(void)
{
	this->Clean();
}

void CProjAlignMain::Clean(void)
{
	if(m_pfReproj != 0L) cudaFreeHost(m_pfReproj);
	if(m_pfTiltAngles != 0L) delete[] m_pfTiltAngles;
	if(m_pbSkipProjs != 0L) delete[] m_pbSkipProjs;
	if(m_pCorrTomoStack != 0L) delete m_pCorrTomoStack;
	m_pfReproj = 0L;
	m_pfTiltAngles = 0L;
	m_pbSkipProjs = 0L;
	m_pCorrTomoStack = 0L;
	//--------------------
	m_centralXcf.Clean();
	m_corrProj.Clean();
}

void CProjAlignMain::Setup
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam,
	float fBFactor,
	int iNthGpu
)
{	this->Clean();
	//------------
	m_fBFactor = fBFactor;
	m_iNthGpu = iNthGpu;
	CInput* pInput = CInput::GetInstance();
	cudaSetDevice(pInput->m_piGpuIDs[m_iNthGpu]);
	//-------------------------------------------
	CParam* pParam = CParam::GetInstance();
	int iBinX = (int)(pTomoStack->m_aiStkSize[0] 
	   / pParam->m_fXcfSize + 0.5f);
	int iBinY = (int)(pTomoStack->m_aiStkSize[1] 
	   / pParam->m_fXcfSize + 0.5f);
	m_iBin = (iBinX > iBinY) ? iBinX : iBinY;
	if(m_iBin < 1) m_iBin = 1;
	//------------------------
	bool bShiftOnly = true, bRandomFill = true;
	bool bFourierCrop = true, bRWeight = true;
	float fTiltAxis = pAlignParam->GetTiltAxis(
	   pAlignParam->m_iNumFrames / 2);
	m_pCorrTomoStack = new Correct::CCorrTomoStack;
	m_pCorrTomoStack->Set0(pInput->m_piGpuIDs[m_iNthGpu]);
	m_pCorrTomoStack->Set1(pTomoStack->m_aiStkSize, 0, fTiltAxis);
	m_pCorrTomoStack->Set2((float)m_iBin, !bFourierCrop, bRandomFill);
	m_pCorrTomoStack->Set3(!bShiftOnly, false, bRWeight);
	m_pBinStack = m_pCorrTomoStack->GetCorrectedStack(false);
	//-------------------------------------------------------
	m_iVolZ = pParam->m_iVolZ / m_iBin / 2 * 2;
	m_iNumProjs = pTomoStack->m_aiStkSize[2];
	m_pbSkipProjs = new bool[m_iNumProjs];
	//------------------------------------
	int iPixels = m_pBinStack->GetPixels();
	cudaMallocHost(&m_pfReproj, sizeof(float) * iPixels);
	//---------------------------------------------------
	m_centralXcf.Setup(m_pBinStack->m_aiStkSize, m_iVolZ);
	mLoadMask();
	//----------------------------------------------------
	bool bPadded = true;
	m_corrProj.Setup(pTomoStack->m_aiStkSize, !bPadded,
	   bRandomFill, !bFourierCrop, fTiltAxis, (float)m_iBin, iNthGpu);
	//----------------------------------------------------------------
	m_aCalcReproj.Setup(m_pBinStack->m_aiStkSize, m_iVolZ, iNthGpu);
}

float CProjAlignMain::DoIt
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam
)
{	m_pTomoStack = pTomoStack;
	m_pAlignParam = pAlignParam;
	m_iNumProjs = m_pTomoStack->m_aiStkSize[2];
	m_iZeroTilt = m_pAlignParam->GetFrameIdxFromTilt(0.0f);
	//-----------------------------------------------------
	CInput* pInput = CInput::GetInstance();
	cudaSetDevice(pInput->m_piGpuIDs[m_iNthGpu]);
	//-------------------------------------------
	bool bCopy = true;
	m_pfTiltAngles = m_pAlignParam->GetTilts(bCopy);
	//----------------------------------------------
	mBinStack();
	//----------
	float fError = mDoAll();
	printf("\n");
	return fError;
}

float CProjAlignMain::mDoAll(void)
{
	m_pAlignParam->FitRotCenterZ();
	m_pAlignParam->RemoveOffsetZ(-1.0f);
	//----------------------------------
	for(int i=0; i<m_iNumProjs; i++)
	{	m_pbSkipProjs[i] = true;
	}
	m_pbSkipProjs[m_iZeroTilt] = false;
	//--------------------------------
	printf("# Projection matching measurements\n");
	printf("# tilt angle   x shift   y shift\n");
	float fMaxErr = (float)-1e20;
	for(int i=1; i<m_iNumProjs; i++)
	{	int iProj = m_iZeroTilt + i;
		if(iProj < m_iNumProjs && iProj >= 0)
		{	float fErr = mAlignProj(iProj);
			m_pbSkipProjs[iProj] = false;
			if(fErr > fMaxErr) fMaxErr = fErr;
		}
		//----------------------------------------
		iProj = m_iZeroTilt - i;
		if(iProj >= 0 && iProj < m_iNumProjs)
		{	float fErr = mAlignProj(iProj);
			m_pbSkipProjs[iProj] = false;
			if(fErr > fMaxErr) fMaxErr = fErr;
		}
	}
	printf("\n");
	return fMaxErr;
}

float CProjAlignMain::mAlignProj(int iProj)
{
	float* pfRawProj = m_pTomoStack->GetFrame(iProj);
	m_corrProj.SetProj(pfRawProj);
	mCalcReproj(iProj);
	//-----------------
	float fShift = mMeasure(iProj);
	mCorrectProj(iProj);
	return fShift;
}

float CProjAlignMain::mMeasure(int iProj)
{
	float fTilt = m_pAlignParam->GetTilt(iProj);
	//------------------------------------------
	float afShift[2] = {0.0f};
	float* pfProj = m_pBinStack->GetFrame(iProj);
	m_centralXcf.SetupXcf(0.5f, m_fBFactor);
	m_centralXcf.DoIt(m_pfReproj, pfProj, fTilt);
        m_centralXcf.GetShift(afShift);
        afShift[0] *= m_afBinning[0];
        afShift[1] *= m_afBinning[1];
	//---------------------------
	printf("  %6.2f  %8.2f  %8.2f\n", fTilt, afShift[0], afShift[1]);
	//---------------------------------------------------------------
	float fShift = afShift[0] * afShift[0] + afShift[1] * afShift[1]; 
	fShift = (float)sqrt(fShift);
	//---------------------------
	float fTiltAxis = m_pAlignParam->GetTiltAxis(iProj);
	MrcUtil::CAlignParam::RotShift(afShift, fTiltAxis, afShift);
	float afInducedS[2] = {0.0f};
	m_pAlignParam->CalcZInducedShift(iProj, afInducedS);
	afShift[0] += afInducedS[0];
	afShift[1] += afInducedS[1];
	m_pAlignParam->AddShift(iProj, afShift);
	//--------------------------------------
	return fShift;
}

void CProjAlignMain::mCalcBinning(void)
{	
	int* piProjSize = m_pTomoStack->m_aiStkSize;
	CParam* pParam = CParam::GetInstance();
	int iBinX = (int)(piProjSize[0] / pParam->m_fXcfSize + 0.5f);
	int iBinY = (int)(piProjSize[1] / pParam->m_fXcfSize + 0.5f);
	m_iBin = (iBinX > iBinY) ? iBinX : iBinY;
	if(m_iBin < 1) m_iBin = 1;
}

void CProjAlignMain::mBinStack(void)
{
	bool bShiftOnly = true;
	bool bRandomFill = true;
	bool bFourierCrop = true;
	CInput* pInput = CInput::GetInstance();
	m_pCorrTomoStack->DoIt(m_pTomoStack, m_pAlignParam, 0L);
	m_pCorrTomoStack->GetBinning(m_afBinning);
}
	
void CProjAlignMain::mRemoveSpikes(MrcUtil::CTomoStack* pTomoStack)
{
	CInput* pInput = CInput::GetInstance();
	CRemoveSpikes::DoIt
	( pTomoStack, pInput->m_piGpuIDs,
	  pInput->m_iNumGpus
	);
}

void CProjAlignMain::mCalcReproj(int iProj)
{
	m_pbSkipProjs[iProj] = true;
	m_aCalcReproj.DoIt(m_pBinStack->m_ppfFrames, m_pfTiltAngles, 
	  m_pbSkipProjs, iProj, m_pfReproj);
	m_pbSkipProjs[iProj] = false;
}

void CProjAlignMain::mCorrectProj(int iProj)
{
	float afShift[2] = {0.0f};
	m_pAlignParam->GetShift(iProj, afShift);
	float fTiltAxis = m_pAlignParam->GetTiltAxis(iProj);
	m_corrProj.DoIt(afShift, fTiltAxis);
	//----------------------------------
	bool bPadded = true;
	float* pfBinProj = m_pBinStack->GetFrame(iProj);
	m_corrProj.GetProj(pfBinProj, m_pBinStack->m_aiStkSize, !bPadded);
}

//-------------------------------------------------------------------
// Load the optional -MaskFile MRC and bin it to match the binned
// projection stack size, then pass to CCentralXcf::SetMask().
// Supports MRC modes: 0 (int8), 1 (int16), 2 (float32), 6 (uint16).
//-------------------------------------------------------------------
void CProjAlignMain::mLoadMask(void)
{
	CInput* pInput = CInput::GetInstance();
	if(strlen(pInput->m_acMaskFile) == 0) return;
	//----------------------------------------------
	Mrc::CLoadMrc aLoadMrc;
	if(!aLoadMrc.OpenFile(pInput->m_acMaskFile))
	{	fprintf(stderr, "Error: cannot open mask file %s\n",
		   pInput->m_acMaskFile);
		exit(1);
	}
	int iMode  = aLoadMrc.m_pLoadMain->GetMode();
	int iFullX = aLoadMrc.m_pLoadMain->GetSizeX();
	int iFullY = aLoadMrc.m_pLoadMain->GetSizeY();
	int iFullPix = iFullX * iFullY;
	//-------------------------------------------
	int iBytesPerPix;
	switch(iMode)
	{	case 0:  iBytesPerPix = 1; break;  // int8
		case 1:  iBytesPerPix = 2; break;  // int16
		case 2:  iBytesPerPix = 4; break;  // float32
		case 6:  iBytesPerPix = 2; break;  // uint16
		default: iBytesPerPix = 4; break;
	}
	void* pvRaw = new char[iFullPix * iBytesPerPix];
	aLoadMrc.m_pLoadImg->DoIt(0, pvRaw);
	aLoadMrc.CloseFile();
	//-------------------
	// Convert raw pixels to float.
	float* pfFull = new float[iFullPix];
	switch(iMode)
	{	case 0:
		{	signed char* p = (signed char*)pvRaw;
			for(int i=0; i<iFullPix; i++) pfFull[i] = (float)p[i];
			break;
		}
		case 1:
		{	short* p = (short*)pvRaw;
			for(int i=0; i<iFullPix; i++) pfFull[i] = (float)p[i];
			break;
		}
		case 2:
			memcpy(pfFull, pvRaw, iFullPix * sizeof(float));
			break;
		case 6:
		{	unsigned short* p = (unsigned short*)pvRaw;
			for(int i=0; i<iFullPix; i++) pfFull[i] = (float)p[i];
			break;
		}
		default:
			memcpy(pfFull, pvRaw, iFullPix * sizeof(float));
	}
	delete[] (char*)pvRaw;
	//------------------------
	// Box-filter bin to match m_pBinStack size.
	int iOutX  = m_pBinStack->m_aiStkSize[0];
	int iOutY  = m_pBinStack->m_aiStkSize[1];
	int iBinX  = iFullX / iOutX;  if(iBinX < 1) iBinX = 1;
	int iBinY  = iFullY / iOutY;  if(iBinY < 1) iBinY = 1;
	float fInvBin = 1.0f / (iBinX * iBinY);
	//--------------------------------------
	float* pfBinned = new float[iOutX * iOutY];
	memset(pfBinned, 0, iOutX * iOutY * sizeof(float));
	for(int oy=0; oy<iOutY; oy++)
	{	for(int dy=0; dy<iBinY; dy++)
		{	int sy = oy * iBinY + dy;
			if(sy >= iFullY) continue;
			float* pfRow = pfFull + sy * iFullX;
			float* pfOut = pfBinned + oy * iOutX;
			for(int ox=0; ox<iOutX; ox++)
			{	float fSum = 0.0f;
				for(int dx=0; dx<iBinX; dx++)
				{	int sx = ox * iBinX + dx;
					if(sx < iFullX) fSum += pfRow[sx];
				}
				pfOut[ox] += fSum * fInvBin;
			}
		}
	}
	delete[] pfFull;
	//---
	m_centralXcf.SetMask(pfBinned);
	printf("Mask loaded: %s, binned %dx%d → %dx%d.\n",
	   pInput->m_acMaskFile, iFullX, iFullY, iOutX, iOutY);
	delete[] pfBinned;
}
