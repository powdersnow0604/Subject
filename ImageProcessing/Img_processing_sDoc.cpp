
// Img_processing_sDoc.cpp: CImgprocessingsDoc 클래스의 구현
//
#include "pch.h"
#include "framework.h"
// SHARED_HANDLERS는 미리 보기, 축소판 그림 및 검색 필터 처리기를 구현하는 ATL 프로젝트에서 정의할 수 있으며
// 해당 프로젝트와 문서 코드를 공유하도록 해 줍니다.
#ifndef SHARED_HANDLERS
#include "Img_processing_s.h"
#endif

#include "Img_processing_sDoc.h"

#pragma warning(disable: 4703)
#include "CDownSamplingDlg.h"
#include "CUpSampleDlg.h"
#include "CQuantizationDlg.h"
#include "CConstantDlg.h"
#include "CStressTransformDlg.h"
#include "CConstantNoLimitDlg.h"
#define _USE_MATH_DEFINES
#include <math.h>

#include <propkey.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CImgprocessingsDoc

IMPLEMENT_DYNCREATE(CImgprocessingsDoc, CDocument)

BEGIN_MESSAGE_MAP(CImgprocessingsDoc, CDocument)
END_MESSAGE_MAP()


// CImgprocessingsDoc 생성/소멸

CImgprocessingsDoc::CImgprocessingsDoc() noexcept
{
	// TODO: 여기에 일회성 생성 코드를 추가합니다.

}

CImgprocessingsDoc::~CImgprocessingsDoc()
{
}

BOOL CImgprocessingsDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: 여기에 재초기화 코드를 추가합니다.
	// SDI 문서는 이 문서를 다시 사용합니다.

	return TRUE;
}




// CImgprocessingsDoc serialization

void CImgprocessingsDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: 여기에 저장 코드를 추가합니다.
	}
	else
	{
		// TODO: 여기에 로딩 코드를 추가합니다.
	}
}

#ifdef SHARED_HANDLERS

// 축소판 그림을 지원합니다.
void CImgprocessingsDoc::OnDrawThumbnail(CDC& dc, LPRECT lprcBounds)
{
	// 문서의 데이터를 그리려면 이 코드를 수정하십시오.
	dc.FillSolidRect(lprcBounds, RGB(255, 255, 255));

	CString strText = _T("TODO: implement thumbnail drawing here");
	LOGFONT lf;

	CFont* pDefaultGUIFont = CFont::FromHandle((HFONT) GetStockObject(DEFAULT_GUI_FONT));
	pDefaultGUIFont->GetLogFont(&lf);
	lf.lfHeight = 36;

	CFont fontDraw;
	fontDraw.CreateFontIndirect(&lf);

	CFont* pOldFont = dc.SelectObject(&fontDraw);
	dc.DrawText(strText, lprcBounds, DT_CENTER | DT_WORDBREAK);
	dc.SelectObject(pOldFont);
}

// 검색 처리기를 지원합니다.
void CImgprocessingsDoc::InitializeSearchContent()
{
	CString strSearchContent;
	// 문서의 데이터에서 검색 콘텐츠를 설정합니다.
	// 콘텐츠 부분은 ";"로 구분되어야 합니다.

	// 예: strSearchContent = _T("point;rectangle;circle;ole object;");
	SetSearchContent(strSearchContent);
}

void CImgprocessingsDoc::SetSearchContent(const CString& value)
{
	if (value.IsEmpty())
	{
		RemoveChunk(PKEY_Search_Contents.fmtid, PKEY_Search_Contents.pid);
	}
	else
	{
		CMFCFilterChunkValueImpl *pChunk = nullptr;
		ATLTRY(pChunk = new CMFCFilterChunkValueImpl);
		if (pChunk != nullptr)
		{
			pChunk->SetTextValue(PKEY_Search_Contents, value, CHUNK_TEXT);
			SetChunkValue(pChunk);
		}
	}
}

#endif // SHARED_HANDLERS

// CImgprocessingsDoc 진단

#ifdef _DEBUG
void CImgprocessingsDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CImgprocessingsDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CImgprocessingsDoc 명령


BOOL CImgprocessingsDoc::OnOpenDocument(LPCTSTR lpszPathName)
{
	if (!CDocument::OnOpenDocument(lpszPathName))
		return FALSE;

	CFile File;

	File.Open(lpszPathName, CFile::modeRead | CFile::typeBinary);

	if (File.GetLength() == (unsigned long long)(512 * 512)) {
		m_height = 512;
		m_width = 512;
	}
	else if (File.GetLength() == (unsigned long long)(256 * 256)) {
		m_height = 256;
		m_width = 256;
	}
	else if (File.GetLength() == (unsigned long long)(128 * 128)) {
		m_height = 128;
		m_width = 128;
	}
	else if (File.GetLength() == (unsigned long long)(64 * 64)) {
		m_height = 64;
		m_width = 64;
	}
	else {
		AfxMessageBox((LPCTSTR)L"Not Support Image Size");
		return 0;
	}

	m_size = m_width * m_height;

	m_InputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_size);

	for (int i = 0; i < m_size; i++) {
		m_InputImage[i] = 255;
	}

	File.Read(m_InputImage, m_size);
	File.Close();

	return TRUE;
}


BOOL CImgprocessingsDoc::OnSaveDocument(LPCTSTR lpszPathName)
{
	FILE* file;
	//CFile File;
	CFileDialog SaveDlg(FALSE,(LPCTSTR)L"raw",NULL,OFN_HIDEREADONLY);
	
	if (SaveDlg.DoModal() == IDOK) {
		/*
		File.Open(SaveDlg.GetPathName(), CFile::modeCreate | CFile::modeWrite);
		File.Write(m_OutputImage, m_Re_size);
		File.Close();*/

		_wfopen_s(&file,SaveDlg.GetPathName(), L"wb+");
	}

	if (file)
	{
		fwrite(m_OutputImage, sizeof(unsigned char), m_Re_size, file);
		fclose(file);
		return TRUE;
	}
	else
		return FALSE;
	
	//return CDocument::OnSaveDocument(lpszPathName);
}


void CImgprocessingsDoc::OnDownSampling()
{
	int i, j;

	CDownSamplingDlg dlg;

	if (dlg.DoModal() == IDOK)
	{
		m_Re_height = m_height / dlg.m_DownSampleRate;
		m_Re_width = m_width / dlg.m_DownSampleRate;
		m_Re_size = m_Re_height * m_Re_width;

		m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				m_OutputImage[i * m_Re_width + j] = m_InputImage[i * dlg.m_DownSampleRate * m_width + j * dlg.m_DownSampleRate];
			}
		}
	}
}


void CImgprocessingsDoc::OnUpSampling()
{
	int i, j;

	CUpSampleDlg dlg;

	if (dlg.DoModal() == IDOK)
	{
		m_Re_height = m_height * dlg.m_UpSampleRate;
		m_Re_width = m_width * dlg.m_UpSampleRate;
		m_Re_size = m_Re_height * m_Re_width;

		m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);
		
		for (i = 0; i < m_Re_size; i++) {
			m_OutputImage[i] = 0;
		}

		for (i = 0; i < m_height; i++) {
			for (j = 0; j < m_width; j++) {
				m_OutputImage[i * dlg.m_UpSampleRate * m_Re_width + j * dlg.m_UpSampleRate] = m_InputImage[i * m_width + j];
			}
		}
	}
}


void CImgprocessingsDoc::OnReverse()
{
	int i, j;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_Re_width + j] = 255 - m_InputImage[i * m_width + j];
		}
	}
}


void CImgprocessingsDoc::OnQuantization()
{
	CQuantizationDlg dlg;

	if (dlg.DoModal() == IDOK)
	{
		int i, j, value, LEVEL;
		double HIGH, * TEMP;

		m_Re_height = m_height;
		m_Re_width = m_width;
		m_Re_size = m_Re_height * m_Re_width;

		m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);
		
		TEMP = (double*)malloc(sizeof(double) * m_Re_size);

		LEVEL = 256;
		HIGH = 256.;

		value = (int)pow(2, dlg.m_QuantBit);

		for (i = 0; i < m_size; i++) {
			for (j = 0; j < value; j++) {
				if (m_InputImage[i] >= (LEVEL / value) * j && m_InputImage[i] < (LEVEL / value) * (j + 1)) {
					TEMP[i] = (double)(HIGH / value) * j;
				}
			}
		}

		for (i = 0; i < m_size; i++) {
			m_OutputImage[i] = (unsigned char)TEMP[i];
		}
	}
}


void CImgprocessingsDoc::OnSumConstant(BOOL InFunc, double **Target, int height, int width, double alpha)
{
	int i, j;

	if (!InFunc) {
		CConstantDlg dlg;

		m_Re_height = m_height;
		m_Re_width = m_width;
		m_Re_size = m_Re_height * m_Re_width;

		m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

		if (dlg.DoModal() == IDOK)
		{
			for (i = 0; i < m_size; i++) {
				if (m_InputImage[i] + dlg.m_Constant >= 255)
					m_OutputImage[i] = 255;
				else
					m_OutputImage[i] = (unsigned char)(m_InputImage[i] + dlg.m_Constant);
			}
		}
	}
	else {
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				Target[i][j] = 255. < Target[i][j] + alpha ? 255. : Target[i][j] + alpha;
			}
		}
	}
}


void CImgprocessingsDoc::OnSubConstant()
{
	CConstantDlg dlg;

	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (dlg.DoModal() == IDOK)
	{
		for (i = 0; i < m_size; i++) {
			if (m_InputImage[i] - dlg.m_Constant <= 0)
				m_OutputImage[i] = 0;
			else
				m_OutputImage[i] = (unsigned char)(m_InputImage[i] - dlg.m_Constant);
		}
	}
}


void CImgprocessingsDoc::OnMulConstant()
{
	CConstantDlg dlg;

	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (dlg.DoModal() == IDOK)
	{
		for (i = 0; i < m_size; i++) {
			if (m_InputImage[i] * dlg.m_Constant >= 255)
				m_OutputImage[i] = 255;
			else if (m_InputImage[i] * dlg.m_Constant <= 0)
				m_OutputImage[i] = 0;
			else
				m_OutputImage[i] = (unsigned char)(m_InputImage[i] * dlg.m_Constant);
		}
	}
}


void CImgprocessingsDoc::OnDivConstant()
{
	CConstantDlg dlg;

	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (dlg.DoModal() == IDOK)
	{
		for (i = 0; i < m_size; i++) {
			if (m_InputImage[i] / dlg.m_Constant >= 255)
				m_OutputImage[i] = 255;
			else if (m_InputImage[i] / dlg.m_Constant <= 0)
				m_OutputImage[i] = 0;
			else
				m_OutputImage[i] = (unsigned char)(m_InputImage[i] / dlg.m_Constant);
		}
	}
}


void CImgprocessingsDoc::OnAndOperate()
{
	CConstantDlg dlg;

	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (dlg.DoModal() == IDOK)
	{
		for (i = 0; i < m_size; i++) {
			if ((m_InputImage[i] & (unsigned char)dlg.m_Constant) >= 255)
				m_OutputImage[i] = 255;
			else if ((m_InputImage[i] & (unsigned char)dlg.m_Constant) < 0)
				m_OutputImage[i] = 0;
			else
				m_OutputImage[i] = (m_InputImage[i] & (unsigned char)dlg.m_Constant);
		}
	}
}


void CImgprocessingsDoc::OnOrOperate()
{
	CConstantDlg dlg;

	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (dlg.DoModal() == IDOK)
	{
		for (i = 0; i < m_size; i++) {
			if ((m_InputImage[i] | (unsigned char)dlg.m_Constant) >= 255)
				m_OutputImage[i] = 255;
			else if ((m_InputImage[i] | (unsigned char)dlg.m_Constant) < 0)
				m_OutputImage[i] = 0;
			else
				m_OutputImage[i] = (m_InputImage[i] | (unsigned char)dlg.m_Constant);
		}
	}
}


void CImgprocessingsDoc::OnXorOperate()
{
	CConstantDlg dlg;

	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (dlg.DoModal() == IDOK)
	{
		for (i = 0; i < m_size; i++) {
			if ((m_InputImage[i] ^ (unsigned char)dlg.m_Constant) >= 255)
				m_OutputImage[i] = 255;
			else if ((m_InputImage[i] ^ (unsigned char)dlg.m_Constant) < 0)
				m_OutputImage[i] = 0;
			else
				m_OutputImage[i] = (m_InputImage[i] ^ (unsigned char)dlg.m_Constant);
		}
	}
}


void CImgprocessingsDoc::OnNotOperate()
{
	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	for (i = 0; i < m_size; i++) {	
		m_OutputImage[i] = ~m_InputImage[i];
	}
	
}


void CImgprocessingsDoc::OnGammaCorrection()
{
	CConstantDlg dlg;

	int i;
	double temp;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (dlg.DoModal() == IDOK)
	{
		for (i = 0; i < m_size; i++) {
			temp = pow(m_InputImage[i], 1 / dlg.m_Constant);
			if (temp <= 0)
				m_OutputImage[i] = 0;
			else if (temp >= 255)
				m_OutputImage[i] = 255;
			else
				m_OutputImage[i] = (unsigned char)temp;
		}
	}
}


void CImgprocessingsDoc::OnBinarization()
{
	CConstantDlg dlg;

	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (dlg.DoModal() == IDOK)
	{
		for (i = 0; i < m_size; i++) {
			if (m_InputImage[i] >= dlg.m_Constant)
				m_OutputImage[i] = 255;
			else
				m_OutputImage[i] = 0;
		}
	}
}


void CImgprocessingsDoc::OnStressTransform()
{
	CStressTransformDlg dlg;

	int i;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (dlg.DoModal() == IDOK)
	{
		for (i = 0; i < m_size; i++) {
			if (m_InputImage[i] >= dlg.m_StartPoint && m_InputImage[i] <= dlg.m_EndPoint)
				m_OutputImage[i] = 255;
			else
				m_OutputImage[i] = m_InputImage[i];
		}
	}
}


void CImgprocessingsDoc::OnIcCompress()
{
	CConstantDlg dlg;
	int i;
	unsigned char min = m_InputImage[0], max = m_InputImage[0];

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	for (i = 0; i < m_size; i++)
	{
		if (m_InputImage[i] > max)
			max = m_InputImage[i];
		else if (m_InputImage[i] < min)
			min = m_InputImage[i];
	}

	if (dlg.DoModal() == IDOK)
	{
		for (i = 0; i < m_size; i++) {
			m_OutputImage[i] = (unsigned char)((max - min - 2 * dlg.m_Constant) / (max - min) * (m_InputImage[i] - min) + min + dlg.m_Constant);
		}
	}
}


void CImgprocessingsDoc::OnIcStretching()
{
	int i;
	unsigned char min = m_InputImage[0], max = m_InputImage[0];

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	for (i = 0; i < m_size; i++)
	{
		if (m_InputImage[i] > max)
			max = m_InputImage[i];
		else if (m_InputImage[i] < min)
			min = m_InputImage[i];
	}

	for (i = 0; i < m_size; i++) {
		m_OutputImage[i] = (unsigned char)lround(((m_InputImage[i] - min) * 255. / (max - min)));
	}
}


void CImgprocessingsDoc::OnEndInSearch()
{

	CConstantDlg dlg;

	int i;
	unsigned char min = m_InputImage[0], max = m_InputImage[0];

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	for (i = 0; i < m_size; i++)
	{
		if (m_InputImage[i] > max)
			max = m_InputImage[i];
		else if (m_InputImage[i] < min)
			min = m_InputImage[i];
	}

	if (dlg.DoModal() == IDOK)
	{
		max -= (unsigned char)dlg.m_Constant;
		min += (unsigned char)dlg.m_Constant;

		for (i = 0; i < m_size; i++) {
			if (m_InputImage[i] >= max)
				m_OutputImage[i] = 255;
			else if (m_InputImage[i] <= min)
				m_OutputImage[i] = 0;
			else
				m_OutputImage[i] = (unsigned char)lround(((m_InputImage[i] - min) * 255. / (max - min)));
		}
	}
}


void CImgprocessingsDoc::OnHistogram()
{
	int i, j, value;
	unsigned char LOW, HIGH;
	double MAX, MIN, DIF;

	m_Re_height = 256;
	m_Re_width = 256;
	m_Re_size = m_Re_height * m_Re_width;

	LOW = 0;
	HIGH = 255;

	for (i = 0; i < 256; i++) {
		m_HIST[i] = LOW;
	}

	for (i = 0; i < m_size; i++) {
		value = (int)m_InputImage[i];
		m_HIST[value]++;
	}

	MAX = m_HIST[0];
	MIN = m_HIST[0];

	for (i = 1; i < 256; i++) {
		if (m_HIST[i] > MAX) {
			MAX = m_HIST[i];
		}
		else if (m_HIST[i] < MIN) {
			MIN = m_HIST[i];
		}
	}

	DIF = MAX - MIN;

	for (i = 0; i < 256; i++) {
		m_Scale_HIST[i] = (unsigned char)lround((m_HIST[i] - MIN) * HIGH / DIF);
	}

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * (m_Re_size + 256 *20));

	for (i = 0; i < m_Re_size; i++) {
		m_OutputImage[i] = 255;
	}

	for (i = 0; i < 256; i++) {
		for (j = 0; j < m_Scale_HIST[i]; j++) {
			m_OutputImage[m_Re_width * (m_Re_height - j - 1) + i] = 0;
		}
	}

	for (i = m_Re_height; i < m_Re_height + 5; i++) {
		for (j = 0; j < 256; j++) {
			m_OutputImage[m_Re_width * i + j] = 255;
		}
	}

	for (i = m_Re_height+5; i < m_Re_height + 20; i++) {
		for (j = 0; j < 256; j++) {
			m_OutputImage[m_Re_width * i + j] = j;
		}
	}

	m_Re_height += 20;
	m_Re_size = m_Re_height * m_Re_width;
}


void CImgprocessingsDoc::OnHistoEqual()
{
	int i, value;
	unsigned char LOW, HIGH, TEMP;
	double sum = 0.;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	LOW = 0;
	HIGH = 255;

	for (i = 0; i < 256; i++) {
		m_HIST[i] = LOW;
	}

	for (i = 0; i < m_size; i++) {
		value = (int)m_InputImage[i];
		m_HIST[value]++;
	}

	for (i = 0; i < 256; i++) {
		sum += m_HIST[i];
		m_Sum_Of_HIST[i] = sum;
	}

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	for (i = 0; i < m_size; i++) {
		TEMP = m_InputImage[i];
		m_OutputImage[i] = (unsigned char)lround(m_Sum_Of_HIST[TEMP] * HIGH / m_size);
	}
}


void CImgprocessingsDoc::OnHistoSpec()
{
	int i, value, Dvalue, top, bottom, DADD;
	unsigned char* m_DTEMP, m_Sum_Of_ScHIST[256], m_TABLE[256];
	unsigned char LOW, HIGH, Temp, * m_Org_Temp;
	double m_DHIST[256], m_Sum_Of_DHIST[256], SUM = 0., DSUM = 0.;
	double DMIN, DMAX;

	top = 255;
	bottom = top - 1;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);
	m_Org_Temp = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	CFile File;
	CFileDialog openDlg(TRUE);

	if (openDlg.DoModal() == IDOK) {
		File.Open(openDlg.GetPathName(), CFile::modeRead);

		if (File.GetLength() == (unsigned)m_size) {
			m_DTEMP = (unsigned char*)malloc(sizeof(unsigned char) * m_size);
			File.Read(m_DTEMP, m_size);
			File.Close();
		}
		else{
			AfxMessageBox(L"Image size not matched");
			return;
		}
	}

	LOW = 0;
	HIGH = 255;

	for (i = 0; i < 256; i++) {
		m_HIST[i] = LOW;
		m_DHIST[i] = LOW;
		m_TABLE[i] = LOW;
	}
	
	for (i = 0; i < m_size; i++) {
		value = (int)m_InputImage[i];
		m_HIST[value]++;
		Dvalue = (int)m_DTEMP[i];
		m_DHIST[Dvalue]++;
	}

	for (i = 0; i < 256; i++) {
		SUM += m_HIST[i];
		m_Sum_Of_HIST[i] = SUM;
		DSUM += m_DHIST[i];
		m_Sum_Of_DHIST[i] = DSUM;
	}

	for (i = 0; i < m_size; i++) {
		Temp = m_InputImage[i];
		m_Org_Temp[i] = (unsigned char)lround(m_Sum_Of_HIST[Temp] * HIGH / m_size);
	}

	DMIN = m_Sum_Of_DHIST[0];
	DMAX = m_Sum_Of_DHIST[255];

	/*
	for (i = 0; i < 256; i++) {
		m_Sum_Of_ScHIST[i] = (unsigned char)lround((m_Sum_Of_DHIST[i]-DMIN) * HIGH / (DMAX-DMIN));
	}*/
	
	for (i = 0; i < 256; i++) {
		m_Sum_Of_ScHIST[i] = (unsigned char)lround(m_Sum_Of_DHIST[i] * HIGH / m_size);
	}

	while (1) {
		for (i = m_Sum_Of_ScHIST[bottom]; i <= m_Sum_Of_ScHIST[top]; i++) {
			m_TABLE[i] = top;
		}

		top = bottom;
		bottom--;

		if (bottom < -1)
			break;
	}

	for (i = 0; i < m_size; i++) {
		DADD = (int)m_Org_Temp[i];
		m_OutputImage[i] = m_TABLE[DADD];
	}
}


void CImgprocessingsDoc::OnInoutChange()
{
	unsigned char* temp = m_OutputImage;
	m_OutputImage = m_InputImage;
	m_InputImage = temp;

	m_height ^= m_Re_height; m_Re_height ^= m_height; m_height ^= m_Re_height;
	m_width ^= m_Re_width; m_Re_width ^= m_width; m_width ^= m_Re_width;

	m_size = m_height * m_width;
	m_Re_size = m_Re_height * m_Re_width;
}


#define MASK_PROCESS_S
#define EMBO_VERSION_1
void CImgprocessingsDoc::OnEmbossing()
{
	CConstantDlg dlg;

	int i, j;
	//double EmboMask[3][3] = { {-1,0,0},{0,0,0},{0,0,1} };
	//double EmboMask[3][3] = { {-1,-1,0},{-1,0,1},{0,1,1} };
	//double EmboMask[3][3] = { {0,0,0},{0,1,0},{0,0,0} };
	//double EmboMask[3][3] = { {1,1,1},{1,-8,1},{1,1,1} };

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (dlg.DoModal() == IDOK)
	{
		if (dlg.m_Constant - (int)dlg.m_Constant > 0 || (int)dlg.m_Constant % 2 == 0)
		{
			AfxMessageBox(L"constant should be odd integer greater than 3");
			return;
		}

		if ((int)dlg.m_Constant < 3)
			dlg.m_Constant = 3.;

		double** EmboMask = Image2DMem((int)dlg.m_Constant, (int)dlg.m_Constant);
		

#ifdef EMBO_VERSION_1
		for (i = 0; i <= (int)dlg.m_Constant / 2; i++) {
			for (j = 0; j <= (int)dlg.m_Constant/2 - i; j++) {
				EmboMask[i][j] = -1;
				EmboMask[(int)dlg.m_Constant -1 - i][(int)dlg.m_Constant - 1 - j] = 1;
			}
		}
#else
		EmboMask[0][0] = -1;
		EmboMask[(int)dlg.m_Constant - 1][(int)dlg.m_Constant - 1] = 1;
#endif



#ifdef MASK_PROCESS_S
		m_tempImage = OnMaskProcess_s(m_InputImage, EmboMask,(int)dlg.m_Constant);
#else
		m_tempImage = OnMaskProcess(m_InputImage, EmboMask, (int)dlg.m_Constant);
#endif



		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				m_tempImage[i][j] += 128;
				if (m_tempImage[i][j] > 255.)
					m_tempImage[i][j] = 255.;
				else if (m_tempImage[i][j] < 0.)
					m_tempImage[i][j] = 0.;
			}
		}

		m_tempImage = OnScale(m_tempImage, m_Re_height, m_Re_width);

		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				m_OutputImage[i * m_Re_width + j] = (unsigned char)lround(m_tempImage[i][j]);
			}
		}
	}
}


double** CImgprocessingsDoc::OnMaskProcess(unsigned char* Target, double** Mask, int MaskLen)
{
	int i, j, n, m;
	double** tempInputImage, ** tempOutputImage, S = 0.;

	tempInputImage = Image2DMem(m_height + MaskLen-1, m_width + MaskLen-1);
	tempOutputImage = Image2DMem(m_height, m_width);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			tempInputImage[i+MaskLen/2][j+MaskLen/2] = (double)Target[i * m_width + j];
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			for (n = 0; n < MaskLen; n++) {
				for (m = 0; m < MaskLen; m++) {
					S += Mask[n][m] * tempInputImage[i + n][j + m];
				}
			}
			tempOutputImage[i][j] = S;
			S = 0.;
		}
	}

	return tempOutputImage;
}


double** CImgprocessingsDoc::OnMaskProcess_s(unsigned char* Target, double **Mask, int MaskLen)
{
	int i, j, n, m, row, col;
	double** tempInputImage, ** tempOutputImage, S = 0.;

	tempInputImage = Image2DMem(m_height, m_width);
	tempOutputImage = Image2DMem(m_height, m_width);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			tempInputImage[i][j] = (double)Target[i * m_width + j];
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			for (n = 0; n < MaskLen; n++) {
				row = (i + n - MaskLen / 2) % m_height;
				if (row < 0)
					row += m_height;
				for (m = 0; m < MaskLen; m++) {
					col = (j + m - MaskLen/2) % m_width;
					if (col < 0)
						col += m_width;

					S += Mask[n][m] * tempInputImage[row][col];
				}
			}
			tempOutputImage[i][j] = S;
			S = 0.;
		}
	}

	return tempOutputImage;
}


double** CImgprocessingsDoc::OnMaskProcessArr(unsigned char* Target, double Mask[5][5])
{
	int i, j, n, m;
	double** tempInputImage, ** tempOutputImage, S = 0.;

	tempInputImage = Image2DMem(m_height + 4, m_width + 4);
	tempOutputImage = Image2DMem(m_height, m_width);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			tempInputImage[i + 2][j + 2] = (double)Target[i * m_width + j];
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			for (n = 0; n < 5; n++) {
				for (m = 0; m < 5; m++) {
					S += Mask[n][m] * tempInputImage[i + n][j + m];
				}
			}
			tempOutputImage[i][j] = S;
			S = 0.;
		}
	}

	return tempOutputImage;
}


double** CImgprocessingsDoc::OnScale(double** Target, int height, int width)
{
	int i, j;
	double min, max;

	min = max = Target[0][0];

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (Target[i][j] < min)
				min = Target[i][j];
			else if (Target[i][j] > max)
				max = Target[i][j];
		}
	}

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			Target[i][j] = (Target[i][j] - min) * 255. / (max - min);
		}
	}

	return Target;
}


double** CImgprocessingsDoc::Image2DMem(int height, int width)
{
	double** Temp;
	int i;
	Temp = (double**)malloc(sizeof(double*) * height);
	Temp[0] = (double*)calloc(width * height, sizeof(double));

	for (i = 1; i < height; i++) {
		Temp[i] = Temp[i - 1] + width;
	}

	return Temp;
	
}


void CImgprocessingsDoc::OnBlurr()
{
	CConstantDlg dlg;

	int i, j;
	/*double BlurrMask[3][3] = {{1. / 9, 1. / 9, 1. / 9},
								{1. / 9, 1. / 9, 1. / 9},
								{1. / 9, 1. / 9, 1. / 9} };*/

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (dlg.DoModal() == IDOK)
	{
		if (dlg.m_Constant - (int)dlg.m_Constant > 0 || (int)dlg.m_Constant % 2 == 0)
		{
			AfxMessageBox(L"constant should be odd integer greater than 3");
			return;
		}

		if ((int)dlg.m_Constant < 3)
			dlg.m_Constant = 3.;

		double** BlurrMask = Image2DMem((int)dlg.m_Constant, (int)dlg.m_Constant);

		for (i = 0; i < (int)dlg.m_Constant; i++) {
			for (j = 0; j < (int)dlg.m_Constant; j++) {
				BlurrMask[i][j] = 1. / (dlg.m_Constant * dlg.m_Constant);
			}
		}
#ifdef MASK_PROCESS_S
		m_tempImage = OnMaskProcess_s(m_InputImage, BlurrMask,(int)dlg.m_Constant);
#else
		m_tempImage = OnMaskProcess(m_InputImage, BlurrMask, (int)dlg.m_Constant);
#endif

		//m_tempImage = OnScale(m_tempImage, m_Re_height, m_Re_width);

		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				if (m_tempImage[i][j] > 255.)
					m_tempImage[i][j] = 255.;
				else if (m_tempImage[i][j] < 0.)
					m_tempImage[i][j] = 0.;
			}
		}

		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				m_OutputImage[i * m_Re_width + j] = (unsigned char)lround(m_tempImage[i][j]);
			}
		}
	}
}


void CImgprocessingsDoc::OnGaussianFilter()
{
	CConstantDlg dlg;

	int i, j, MaskLen, x, y;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (dlg.DoModal() == IDOK)
	{
		//MaskLen = (int)(dlg.m_Constant * 8);
		MaskLen = (int)(dlg.m_Constant * 6);

		if (MaskLen % 2 == 0)
			MaskLen += 1;

		if (MaskLen < 3)
			MaskLen = 3;

		double** GaussianMask = Image2DMem(MaskLen, MaskLen);

		for (i = 0; i <= MaskLen / 2; i++) {
			x = MaskLen / 2 - i;
			for (j = 0; j <= MaskLen/2; j++) {
				y = -(MaskLen / 2) + j;
				GaussianMask[i][j] = exp(-((double)(x * x + y * y) / (double)(2 * dlg.m_Constant * dlg.m_Constant))) / (2 * M_PI * dlg.m_Constant * dlg.m_Constant);
			}
		}

		for (i = 0; i < MaskLen / 2; i++) {
			for (j = 1; j <= MaskLen / 2; j++) {
				GaussianMask[i][MaskLen / 2 + j] = GaussianMask[i][MaskLen / 2 - j];
				GaussianMask[MaskLen / 2 + i + 1][j-1] = GaussianMask[MaskLen / 2 - i - 1][j-1];
				GaussianMask[MaskLen / 2 + i + 1][MaskLen / 2 + j] = GaussianMask[MaskLen / 2 - i - 1][MaskLen / 2 - j];
			}
			GaussianMask[MaskLen / 2][MaskLen/2 + i + 1] = GaussianMask[MaskLen / 2][MaskLen/2 - i - 1];
			GaussianMask[MaskLen/2 + i + 1][MaskLen / 2] = GaussianMask[MaskLen/2 - i - 1][MaskLen / 2];
		}


#ifdef MASK_PROCESS_S
		m_tempImage = OnMaskProcess_s(m_InputImage, GaussianMask, MaskLen);
#else
		m_tempImage = OnMaskProcess(m_InputImage, GaussianMask, MaskLen);
#endif
		//m_tempImage = OnScale(m_tempImage, m_Re_height, m_Re_width);

		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				if (m_tempImage[i][j] > 255.)
					m_tempImage[i][j] = 255.;
				else if (m_tempImage[i][j] < 0.)
					m_tempImage[i][j] = 0.;
			}
		}

		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				m_OutputImage[i * m_Re_width + j] = (unsigned char)lround(m_tempImage[i][j]);
			}
		}
	}
}


#define SHARPENING_VERSION_1
void CImgprocessingsDoc::OnSharpening()
{
	CConstantDlg dlg;

	int i, j;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (dlg.DoModal() == IDOK)
	{
		if (dlg.m_Constant - (int)dlg.m_Constant > 0 || (int)dlg.m_Constant % 2 == 0)
		{
			AfxMessageBox(L"constant should be odd integer greater than 3");
			return;
		}

		if ((int)dlg.m_Constant < 3)
			dlg.m_Constant = 3;

		double** SharpeningMask = Image2DMem((int)dlg.m_Constant, (int)dlg.m_Constant);

#ifdef SHARPENING_VERSION_1
		for (i = 0; i < (int)dlg.m_Constant; i++) {
			SharpeningMask[(int)dlg.m_Constant / 2][i] = -1;
		}
		for (i = 0; i < (int)dlg.m_Constant; i++) {
			SharpeningMask[i][(int)dlg.m_Constant / 2] = -1;
		}
		SharpeningMask[(int)dlg.m_Constant / 2][(int)dlg.m_Constant / 2] = 1 + 4 * ((int)dlg.m_Constant - 2);
#else
		for (i = 0; i < (int)dlg.m_Constant; i++) {
			for (j = 0; j < (int)dlg.m_Constant; j++) {
				SharpeningMask[i][j] = -1;
			}
		}
		SharpeningMask[(int)dlg.m_Constant / 2][(int)dlg.m_Constant / 2] = dlg.m_Constant * dlg.m_Constant;
#endif


#ifdef MASK_PROCESS_S
		m_tempImage = OnMaskProcess_s(m_InputImage, SharpeningMask, (int)dlg.m_Constant);
#else
		m_tempImage = OnMaskProcess(m_InputImage, SharpeningMask, (int)dlg.m_Constant);
#endif
		//m_tempImage = OnScale(m_tempImage, m_Re_height, m_Re_width);

		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				if (m_tempImage[i][j] > 255.)
					m_tempImage[i][j] = 255.;
				else if (m_tempImage[i][j] < 0.)
					m_tempImage[i][j] = 0.;
			}
		}

		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				m_OutputImage[i * m_Re_width + j] = (unsigned char)lround(m_tempImage[i][j]);
			}
		}
	}
}


void CImgprocessingsDoc::OnHpfSharp()
{
	
	CConstantDlg dlg;

	int i, j, MaskLen;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);
	
	if (dlg.DoModal() == IDOK)
	{
		if (dlg.m_Constant - (int)dlg.m_Constant > 0 || (int)dlg.m_Constant % 2 == 0)
		{
			AfxMessageBox(L"constant should be odd integer greater than 3");
			return;
		}

		if ((int)dlg.m_Constant < 3)
			dlg.m_Constant = 3;

		MaskLen = (int)dlg.m_Constant;

		double** HpfSharpMask = Image2DMem((int)dlg.m_Constant, (int)dlg.m_Constant);

		for (i = 0; i < (int)dlg.m_Constant; i++) {
			for (j = 0; j < (int)dlg.m_Constant; j++) {
				HpfSharpMask[i][j] = -1. / (double)(MaskLen * MaskLen);
			}
		}
		HpfSharpMask[(int)dlg.m_Constant / 2][(int)dlg.m_Constant / 2] = (double)(MaskLen * MaskLen - 1.) / (double)(MaskLen * MaskLen);


#ifdef MASK_PROCESS_S
		m_tempImage = OnMaskProcess_s(m_InputImage, HpfSharpMask, (int)dlg.m_Constant);
#else
		m_tempImage = OnMaskProcess(m_InputImage, HpfSharpMask, (int)dlg.m_Constant);
#endif
		//m_tempImage = OnScale(m_tempImage, m_Re_height, m_Re_width);

		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				if (m_tempImage[i][j] > 255.)
					m_tempImage[i][j] = 255.;
				else if (m_tempImage[i][j] < 0.)
					m_tempImage[i][j] = 0.;
			}
		}

		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				m_OutputImage[i * m_Re_width + j] = (unsigned char)lround(m_tempImage[i][j]);
			}
		}
	}
}


void CImgprocessingsDoc::OnInPlusOut()
{
	if (m_size != m_Re_size)
	{
		AfxMessageBox(L"size of Input Image and size of Output Image do not match");
		return;
	}

	int i, j;
	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_OutputImage[i * m_width + j] = m_InputImage[i * m_width + j] + m_OutputImage[i * m_width + j];
			if (m_OutputImage[i * m_width + j] > 255)
				m_OutputImage[i * m_width + j] = 255;
			else if (m_OutputImage[i * m_width + j] < 0)
				m_OutputImage[i * m_width + j] = 0;
		}
	}
}


void CImgprocessingsDoc::OnLpfSharp()
{
	CConstantDlg dlgMask, dlgAlpha;

	int i, j, MaskLen;
	double** LpfSharpMask, alpha;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (dlgMask.DoModal() == IDOK)
	{
		if (dlgMask.m_Constant - (int)dlgMask.m_Constant > 0 || (int)dlgMask.m_Constant % 2 == 0)
		{
			AfxMessageBox(L"constant should be odd integer greater than 3");
			return;
		}

		if ((int)dlgMask.m_Constant < 3)
			dlgMask.m_Constant = 3;

		MaskLen = (int)dlgMask.m_Constant;

		LpfSharpMask = Image2DMem(MaskLen, MaskLen);

		for (i = 0; i < MaskLen; i++) {
			for (j = 0; j < MaskLen; j++) {
				LpfSharpMask[i][j] = 1. / (MaskLen * MaskLen);
			}
		}
	}

	if (dlgAlpha.DoModal() == IDOK)
	{
		alpha = dlgAlpha.m_Constant;
	}

#ifdef MASK_PROCESS_S
	m_tempImage = OnMaskProcess_s(m_InputImage, LpfSharpMask, MaskLen);
#else
	m_tempImage = OnMaskProcess(m_InputImage, LpfSharpMask, MaskLen);
#endif

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_tempImage[i][j] = (alpha * m_InputImage[i * m_width + j] - (unsigned char)lround(m_tempImage[i][j]));
		}
	}
	
	//m_tempImage = OnScale(m_tempImage, m_Re_height, m_Re_width);

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			if (m_tempImage[i][j] > 255.)
				m_tempImage[i][j] = 255.;
			else if (m_tempImage[i][j] < 0.)
				m_tempImage[i][j] = 0.;
		}
	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_Re_width + j] = (unsigned char)lround(m_tempImage[i][j]);
		}
	}
}


#define USE_EDGE_BOUNDARYx
void CImgprocessingsDoc::OnDiffOperatorHor()
{
	int i, j;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	double** DiffHorMask = Image2DMem(3, 3);
	DiffHorMask[1][1] = 1.;
	DiffHorMask[0][1] = -1.;

#ifdef MASK_PROCESS_S
	m_tempImage = OnMaskProcess_s(m_InputImage, DiffHorMask, 3);
#else
	m_tempImage = OnMaskProcess(m_InputImage, DiffHorMask, 3);
#endif
	//m_tempImage = OnScale(m_tempImage, m_Re_height, m_Re_width);


#ifdef USE_EDGE_BOUNDARY
	m_tempImage = OnBoundaryEdge(m_tempImage, m_Re_height, m_Re_width);
#else
	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			if (m_tempImage[i][j] > 255.)
				m_tempImage[i][j] = 255.;
			else if (m_tempImage[i][j] < 0.)
				m_tempImage[i][j] = 0.;
		}
	}
#endif

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_Re_width + j] = (unsigned char)lround(m_tempImage[i][j]);
		}
	}
}


void CImgprocessingsDoc::OnHomogenOperator()
{
	int i, j, n ,m;
	double max, ** tempOutputImage;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	m_tempImage = Image2DMem(m_height + 2, m_width + 2);
	tempOutputImage = Image2DMem(m_Re_height, m_Re_width);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_tempImage[i + 1][j + 1] = (double)m_InputImage[i * m_width + j];
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			max = 0.;
			for (n = 0; n < 3; n++) {
				for (m = 0; m < 3; m++) {
					if (abs(m_tempImage[i + 1][j + 1] - m_tempImage[i + n][j + m]) >= max) {
						max = abs(m_tempImage[i + 1][j + 1] - m_tempImage[i + n][j + m]);
					}
				}
			}
			tempOutputImage[i][j] = max;
		}
	}


#ifdef USE_EDGE_BOUNDARY
	tempOutputImage = OnBoundaryEdge(tempOutputImage, m_Re_height, m_Re_width);
#else
	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			if (tempOutputImage[i][j] > 255.)
				tempOutputImage[i][j] = 255.;
			else if (tempOutputImage[i][j] < 0.)
				tempOutputImage[i][j] = 0.;
		}
	}
#endif


	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_Re_width + j] = (unsigned char)lround(tempOutputImage[i][j]);
		}
	}
}


double** CImgprocessingsDoc::OnBoundaryEdge(double** Target, int height, int width)
{
	CConstantDlg dlg;

	int i, j, BoundaryNum;
	double Boundary1;

	if (dlg.DoModal() == IDOK) {
		BoundaryNum = (int)dlg.m_Constant;
	}

	if (BoundaryNum != 1 && BoundaryNum != 2) {
		goto Default;
	}
	else if (BoundaryNum == 1)
	{
		if (dlg.DoModal() == IDOK) {
			Boundary1 = dlg.m_Constant;
		}

		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				if (Target[i][j] >= Boundary1)
					Target[i][j] = 255.;
				else 
					Target[i][j] = 0.;
			}
		}
		return Target;
	}
	else if (BoundaryNum == 2) {
		double Boundary2;

		if (dlg.DoModal() == IDOK) {
			Boundary1 = dlg.m_Constant;
		}

		if (dlg.DoModal() == IDOK) {
			Boundary2 = dlg.m_Constant;
		}

		if (Boundary1 >= Boundary2) {
			goto Default;
		}

		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				if (Target[i][j] <= Boundary1)
					Target[i][j] = 0.;
				else if (Target[i][j] >= Boundary2)
					Target[i][j] = 255.;
			}
		}

		return Target;
	}


Default:
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (Target[i][j] > 255.)
				Target[i][j] = 255.;
			else if (Target[i][j] < 0.)
				Target[i][j] = 0.;
		}
	}
	return Target;
}


#define EDGE_DIFF1_MASK3
void CImgprocessingsDoc::OnDiff1Edge()
{
	int i, j;
	double** RowDiffMask, **ColDiffMask, **RowDetec, **ColDetec;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);
	m_OutputImageLB = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);
	m_OutputImageRB = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	m_tempImage = Image2DMem(m_Re_height, m_Re_width);
	RowDiffMask = Image2DMem(3, 3);
	ColDiffMask = Image2DMem(3, 3);

#pragma region mask
#ifdef EDGE_DIFF1_MASK1         // robert
	RowDiffMask[0][1] = -2;
	RowDiffMask[2][1] = 2;

	ColDiffMask[1][0] = 2;
	ColDiffMask[1][2] = -2;
#elif defined(EDGE_DIFF1_MASK2) // prewitt
	for (i = 0; i < 3; i++) {
		RowDiffMask[0][i] = -1;
		RowDiffMask[2][i] = 1;

		ColDiffMask[i][0] = 1;
		ColDiffMask[i][2] = -1;
	}
#else                           // sobel
	for (i = 0; i < 3; i++) {
		RowDiffMask[0][i] = -1;
		RowDiffMask[2][i] = 1;

		ColDiffMask[i][0] = 1;
		ColDiffMask[i][2] = -1;
	}
	RowDiffMask[0][1] = -2;
	RowDiffMask[2][1] = 2;

	ColDiffMask[1][0] = 2;
	ColDiffMask[1][2] = -2;
	
#endif
#pragma endregion


#ifdef MASK_PROCESS_S
	ColDetec = OnMaskProcess_s(m_InputImage, ColDiffMask, 3);
	RowDetec = OnMaskProcess_s(m_InputImage, RowDiffMask, 3);
#else
	ColDetec = OnMaskProcess(m_InputImage, ColDiffMask, 3);
	RowDetec = OnMaskProcess(m_InputImage, RowDiffMask, 3); 
#endif


	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_tempImage[i][j] = hypot(RowDetec[i][j], ColDetec[i][j]); 
		}
	}
	//m_tempImage = OnScale(m_tempImage, m_Re_height, m_Re_width);


#ifdef USE_EDGE_BOUNDARY
	m_tempImage = OnBoundaryEdge(m_tempImage, m_Re_height, m_Re_width);
	ColDetec = OnBoundaryEdge(ColDetec, m_Re_height, m_Re_width);
	RowDetec = OnBoundaryEdge(RowDetec, m_Re_height, m_Re_width);
#else
	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			if (m_tempImage[i][j] > 255.)
				m_tempImage[i][j] = 255.;
			else if (m_tempImage[i][j] < 0.)
				m_tempImage[i][j] = 0.;

			if (ColDetec[i][j] > 255.)
				ColDetec[i][j] = 255.;
			else if (ColDetec[i][j] < 0.)
				ColDetec[i][j] = 0.;

			if (RowDetec[i][j] > 255.)
				RowDetec[i][j] = 255.;
			else if (RowDetec[i][j] < 0.)
				RowDetec[i][j] = 0.;
		}
	}
#endif


	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_Re_width + j] = (unsigned char)lround(m_tempImage[i][j]);
			m_OutputImageLB[i * m_Re_width + j] = (unsigned char)lround(ColDetec[i][j]);
			m_OutputImageRB[i * m_Re_width + j] = (unsigned char)lround(RowDetec[i][j]);
		}
	}

	m_printImageBool = TRUE;
}


#define THRESHOLD_METHOD_2
#define HYSTERISIS_EDGE_TRACKING_METHOD_1
void CImgprocessingsDoc::OnCannyEdgeDetection()
{
	//step 1: gaussian filtering
	OnGaussianFilter();


	//step 2: claculate gradient
	typedef struct {
		double magnitude;
		double phase;
	}gradient;

	int i, j, dr1, dc1, dr2, dc2;
	double** RowDiffMask, ** ColDiffMask, ** RowDetec, ** ColDetec, median = 0, max, min, temp;
	gradient** ImageGradient;

	m_tempImage = Image2DMem(m_Re_height+2, m_Re_width+2);
	m_OutputImageLB = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);
	m_OutputImageRB = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);
	RowDiffMask = Image2DMem(3, 3);
	ColDiffMask = Image2DMem(3, 3);

	ImageGradient = (gradient**)malloc((m_Re_height+2) * sizeof(gradient*));
	ImageGradient[0] = (gradient*)calloc(sizeof(gradient), (m_Re_height+2) * (m_Re_width+2));
	for (i = 1; i < m_Re_height+2; i++) {
		ImageGradient[i] = ImageGradient[i - 1] + (m_Re_width+2);
	}

	for (i = 0; i < 3; i++) {
		RowDiffMask[0][i] = -1.;
		RowDiffMask[2][i] = 1.;

		ColDiffMask[i][0] = 1.;
		ColDiffMask[i][2] = -1.;
	}
	RowDiffMask[0][1] = -2.;
	RowDiffMask[2][1] = 2.;

	ColDiffMask[1][0] = 2.;
	ColDiffMask[1][2] = -2.;


	ColDetec = OnMaskProcess(m_OutputImage, ColDiffMask, 3);
	RowDetec = OnMaskProcess(m_OutputImage, RowDiffMask, 3);


	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			ImageGradient[i + 1][j + 1].magnitude = hypot(RowDetec[i][j], ColDetec[i][j]);       
			ImageGradient[i+1][j+1].phase = atan(RowDetec[i][j]/ColDetec[i][j]);
		}
	}

	
	#pragma region Scale
	/*
	max = ImageGradient[1][1].magnitude;

	for (i = 1; i <= m_Re_height; i++) {
		for (j = 1; j <= m_Re_width; j++) {
			if (max < ImageGradient[i][j].magnitude)
				max = ImageGradient[i][j].magnitude;
		}
	}

	for (i = 1; i <= m_Re_height; i++) {
		for (j = 1; j <= m_Re_width; j++) {
			ImageGradient[i][j].magnitude = ImageGradient[i][j].magnitude * 255. / max;
		}
	}*/
	#pragma endregion
	

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			temp = ImageGradient[i + 1][j + 1].magnitude;
			if (temp > 255.)
				temp = 255.;
			m_OutputImageLB[i * m_Re_width + j] = (unsigned char)lround(temp);
		}
	}
	
	//step 3: Non - Maximum - Suppression
	for (i = 1; i <= m_Re_height; i++) {
		for (j = 1; j <= m_Re_width; j++) {
			if (-M_PI / 8 <= ImageGradient[i][j].phase && ImageGradient[i][j].phase < M_PI / 8){
				dr1 = 0; dc1 = 1; dr2 = 0; dc2 = -1;
			}
			else if (M_PI / 8 <= ImageGradient[i][j].phase && ImageGradient[i][j].phase < 3*M_PI / 8) {
				dr1 = 1; dc1 = -1; dr2 = -1; dc2 = 1;
			}
			else if (-3*M_PI / 8 <= ImageGradient[i][j].phase && ImageGradient[i][j].phase < M_PI / 8) {
				dr1 = 1; dc1 = 1; dr2 = -1; dc2 = -1;
			}
			else {
				dr1 = -1; dc1 = 0; dr2 = 1; dc2 = 0;
			}

			if (ImageGradient[i][j].magnitude > ImageGradient[i + dr1][j + dc1].magnitude &&
				ImageGradient[i][j].magnitude > ImageGradient[i + dr2][j + dc2].magnitude) {
				m_tempImage[i][j] = ImageGradient[i][j].magnitude;
			}
			else {
				m_tempImage[i][j] = 0.;
			}
		}
	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			temp = m_tempImage[i+1][j+1];
			if (temp > 255.)
				temp = 255.;
			m_OutputImageRB[i * m_Re_width + j] = (unsigned char)lround(temp);
		}
	}

	
	//step 4: Hysterisis Edge Tracking
	double Th, Tl;

	#pragma region Threshold
#ifdef THRESHOLD_METHOD_1
	double  sigma = 0.33;

	for (i = 0; i < 256; i++) {
		m_HIST[i] = 0.;
	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			temp = m_tempImage[i+1][j+1];
			if (temp > 255.)
				temp = 255.;
			m_HIST[(int)temp]++;
		}
	}

	temp = 0;
	for (i = 0; i < 256; i++) {
		temp += m_HIST[i];
		if (temp >= m_Re_size / 2.) {
			median = (double)i;
			break;
		}
	}

	Th = 255 < median * (1.+sigma) ? 255 : median * (1. + sigma);
	Tl = 0 > median * (1.-sigma) ? 0 : median * (1. - sigma);
	
#elif defined(THRESHOLD_METHOD_2)

	min = m_tempImage[1][1], max = m_tempImage[1][1];

	for (i = 1; i <= m_Re_height; i++) {
		for (j = 1; j <= m_Re_width; j++) {
			if (max < m_tempImage[i][j])
				max = m_tempImage[i][j];
		}
	}

	Th = max * 0.09;
	Tl = Th * 0.05;
#else

	min = m_tempImage[1][1], max = m_tempImage[1][1];

	for (i = 1; i <= m_Re_height; i++) {
		for (j = 1; j <= m_Re_width; j++) {
			if (max < m_tempImage[i][j])
				max = m_tempImage[i][j];
			else if (min > m_tempImage[i][j])
				min = m_tempImage[i][j];
		}
	}

	Th = min + (max - min) * 0.15;
	Tl = min + (max - min) * 0.03;
#endif
	#pragma endregion 


	#pragma region HYSTERISIS_EDGE_TRACKING_METHOD
#ifdef HYSTERISIS_EDGE_TRACKING_METHOD_1
	double** visited = Image2DMem(m_Re_height, m_Re_width);
	double** edge = Image2DMem(m_Re_height, m_Re_width);
	BOOL tbool;
	do {
		tbool = 0;
		for (i = 1; i <= m_Re_height; i++) {
			for (j = 1; j <= m_Re_width; j++) {
				if (m_tempImage[i][j] > Th) {
					edge[i - 1][j - 1] = 255.;
				}
				else if (m_tempImage[i][j] >= Tl) {
					tbool |= OnFollowEdge(m_tempImage, visited, edge, i - 1, j - 1, Tl, Th);
				}
			}
		}
	} while (tbool);
#elif defined(HYSTERISIS_EDGE_TRACKING_METHOD_2)
	double** visited = Image2DMem(m_Re_height, m_Re_width);
	double** edge = Image2DMem(m_Re_height, m_Re_width);
	
	for (i = 1; i <= m_Re_height; i++) {
		for (j = 1; j <= m_Re_width; j++) {
			if (m_tempImage[i][j] > Th) {
				edge[i - 1][j - 1] = 255.;
			}
			else if (m_tempImage[i][j] >= Tl) {
				(void)OnFollowEdge(m_tempImage, visited, edge, i - 1, j - 1, Tl, Th);
			}
		}
	}
	
#else
	unsigned char strong = 255, weak = 80;

	for (i = 1; i <= m_Re_height; i++) {
		for (j = 1; j <= m_Re_width; j++) {
			if (m_tempImage[i][j] < Tl) {
				m_tempImage[i][j] = 0;
			}
			else if (Tl <= m_tempImage[i][j] && m_tempImage[i][j] <= Th) {
				m_tempImage[i][j] = weak;
			}
			else if (m_tempImage[i][j] > Th) {
				m_tempImage[i][j] = strong;
			}

		}
	}
	
	for (i = 1; i <= m_Re_height; i++) {
		for (j = 1; j <= m_Re_width; j++) {
			if (m_tempImage[i][j] == weak) {
				if (m_tempImage[i - 1][j - 1] == strong || m_tempImage[i - 1][j] == strong || m_tempImage[i - 1][j + 1] == strong ||
					m_tempImage[i][j - 1] == strong || m_tempImage[i][j + 1] == strong ||
					m_tempImage[i + 1][j - 1] == strong || m_tempImage[i + 1][j] == strong || m_tempImage[i + 1][j + 1] == strong) {
					m_tempImage[i][j] = strong;
				}
				else {
					m_tempImage[i][j] = 0;
				}
				/*
				int connectedBool = 0;
				for (m = -1; m < 2; m++) {
					if (i + m < 0 || i + m > m_Re_height)
						continue;
					for (n = -1; n < 2; n++) {
						if (j + n < 0 || j + n > m_Re_width)
							continue;
						if (m_tempImage[i + m][j + n] > Th) {
							connectedBool = 1;
							goto OutOfFor;
						}
					}
				}
			OutOfFor:
				if (!connectedBool) {
					m_tempImage[i][j] = 0;
				}
				else {
					m_tempImage[i][j] = strong;
				}
				*/
			}
			
		}
	}
#endif
	#pragma endregion


	#pragma region LogicallyIneffective
	/*
	//m_tempImage = OnScale(m_tempImage, m_Re_height, m_Re_width);

	for (i = 1; i <= m_Re_height; i++) {
		for (j = 1; j <= m_Re_width; j++) {
			if (m_tempImage[i][j] > 255.)
				m_tempImage[i][j] = 255.;
			else if (m_tempImage[i][j] < 0.)
				m_tempImage[i][j] = 0.;
		}
	}*/  
	#pragma endregion

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
		#pragma region HYSTERISIS_EDGE_TRACKING_METHOD
		#if defined(HYSTERISIS_EDGE_TRACKING_METHOD_1) || defined(HYSTERISIS_EDGE_TRACKING_METHOD_2)
			m_OutputImage[i * m_Re_width + j] = (unsigned char)edge[i][j];
		#else
			m_OutputImage[i * m_Re_width + j] = (unsigned char)lround(m_tempImage[i+1][j+1]);
		#endif
		#pragma endregion
		}
	}

	m_printImageBool = TRUE;
}


BOOL CImgprocessingsDoc::OnFollowEdge(double** tempImage, double** visited, double** edge, int row, int col, double Tl, double Th)
{
	/*
	int i, j;

	visited[row][col] = 1;
	edge[row][col] = 255.;

	for (i = -1; i < 2; i++) {
		if (row + i < 0 || row + i >= m_Re_height)
			continue;
		for (j = -1; j < 2; j++) {
			if (col + j < 0 || col + j >= m_Re_width)
				continue;
			if (tempImage[row + 1 + i][col + 1 + j] >= Tl && !visited[row + i][col + j])
				OnFollowEdge(tempImage, visited, edge, row + i, col + j, Tl);
		}
	}
	*/

	int i, j;

	visited[row][col] = 1;

	for (i = -1; i < 2; i++) {
		if (row + i < 0 || row + i >= m_Re_height)
			continue;

		for (j = -1; j < 2; j++) {
			if (col + j < 0 || col + j >= m_Re_width)
				continue;

			if (tempImage[row + 1 + i][col + 1 + j] > Th) {
				tempImage[row + 1][col + 1] = 255.;
				edge[row][col] = 255.;
				return TRUE;
			}
			else if (tempImage[row + 1 + i][col + 1 + j] >= Tl && !visited[row + i][col + j]) {
				if (OnFollowEdge(tempImage, visited, edge, row + i, col + j, Tl, Th)) {
					tempImage[row + 1][col + 1] = 255.;
					edge[row][col] = 255.;
					return TRUE;
				}
			}
		}
	}

	return FALSE;
}


#define EDGE_LAPLACIAN_MASK_3
void CImgprocessingsDoc::OnLaplacian()
{
	int i, j;
	double **LaplacianMask;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	LaplacianMask = Image2DMem(3, 3);

#ifdef EDGE_LAPLACIAN_MASK_1
	for (i = -1; i < 3; i++) {
		LaplacianMask[1 + (i % 2)][1 + ((1 - i) % 2)] = 1;
	}
	LaplacianMask[1][1] = -4;
#elif defined(EDGE_LAPLACIAN_MASK_2)
	for (i = -1; i < 3; i++) {
		LaplacianMask[1 + (i % 2)][1 + ((1 - i) % 2)] = -1;
	}
	LaplacianMask[1][1] = 4;
#elif defined(EDGE_LAPLACIAN_MASK_3)
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			LaplacianMask[i][j] = 1;
		}
	}
	LaplacianMask[1][1] = -8;
#else
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			LaplacianMask[i][j] = -1;
		}
	}
	LaplacianMask[1][1] = 8;
#endif


	m_tempImage = OnMaskProcess(m_InputImage, LaplacianMask, 3);

	//m_tempImage = OnScale(m_tempImage, m_Re_height, m_Re_width);

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			if (m_tempImage[i][j] > 255.)
				m_tempImage[i][j] = 255.;
			else if (m_tempImage[i][j] < 0.)
				m_tempImage[i][j] = 0.;
		}
	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_Re_width + j] = (unsigned char)lround(m_tempImage[i][j]);
		}
	}
}


void CImgprocessingsDoc::OnLaplacianOfGaussian()
{
	CConstantDlg dlg;

	int i, j, MaskLen, x, y;
	double stdev, temp;
	double approxMask[5][5] = { { 0, 0,-1, 0, 0},
								{ 0,-1,-2,-1, 0},
								{-1,-2,16,-2,-1},
								{ 0,-1,-2,-1, 0},
								{ 0, 0,-1, 0, 0} };

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (dlg.DoModal() == IDOK)
	{
		//MaskLen = (int)(dlg.m_Constant * 8);
		MaskLen = (int)(dlg.m_Constant * 6);

		stdev = dlg.m_Constant;

		if (MaskLen % 2 == 0)
			MaskLen += 1;

		if (MaskLen < 3)
			MaskLen = 3;

		double** GaussianMask = Image2DMem(MaskLen, MaskLen);

		if (MaskLen == 5) {
			for (i = 0; i < 5; i++) {
				for (j = 0; j < 5; j++) {
					GaussianMask[i][j] = approxMask[i][j];
				}
			}
		}
		else {
			for (i = 0; i <= MaskLen / 2; i++) {
				x = MaskLen / 2 - i;
				for (j = 0; j <= MaskLen / 2; j++) {
					y = -(MaskLen / 2) + j;
					temp = (double)(x * x + y * y) / (double)(2 * stdev * stdev);
					GaussianMask[i][j] = 50 * exp(-temp) * (temp - 1) / (M_PI * stdev * stdev * stdev * stdev);
				}
			}

			for (i = 0; i < MaskLen / 2; i++) {
				for (j = 1; j <= MaskLen / 2; j++) {
					GaussianMask[i][MaskLen / 2 + j] = GaussianMask[i][MaskLen / 2 - j];
					GaussianMask[MaskLen / 2 + i + 1][j - 1] = GaussianMask[MaskLen / 2 - i - 1][j - 1];
					GaussianMask[MaskLen / 2 + i + 1][MaskLen / 2 + j] = GaussianMask[MaskLen / 2 - i - 1][MaskLen / 2 - j];
				}
				GaussianMask[MaskLen / 2][MaskLen / 2 + i + 1] = GaussianMask[MaskLen / 2][MaskLen / 2 - i - 1];
				GaussianMask[MaskLen / 2 + i + 1][MaskLen / 2] = GaussianMask[MaskLen / 2 - i - 1][MaskLen / 2];
			}
		}



		m_tempImage = OnMaskProcess(m_InputImage, GaussianMask, MaskLen);
		//m_tempImage = OnMaskProcessArr(m_InputImage, approxMask);


		//m_tempImage = OnScale(m_tempImage, m_Re_height, m_Re_width);

		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				if (m_tempImage[i][j] > 255.)
					m_tempImage[i][j] = 255.;
				else if (m_tempImage[i][j] < 0.)
					m_tempImage[i][j] = 0.;
			}
		}

		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				m_OutputImage[i * m_Re_width + j] = (unsigned char)lround(m_tempImage[i][j]);
			}
		}
	}
}


void CImgprocessingsDoc::OnDifferenceOfGaussian()
{
	CConstantDlg dlg;

	int i, j, MaskLen, x, y;
	double stdev1, stdev2;
	double aproxMask[9][9] = {  {0,0,0,-1,-1,-1,0,0,0}, 
								{0,-2,-3,-3,-3,-3,-3,-2,0}, 
								{0,-3,-2,-1,-1,-1,-2,-3,0}, 
								{-1,-3,-1,9,9,9,-1,-3,-1},
								{-1,-3,-1,9,19,9,-1,-3,-1}, 
								{-1,-3,-1,9,9,9,-1,-3,-1},
								{0,-3,-2,-1,-1,-1,-2,-3,0},
								{0,-2,-3,-3,-3,-3,-3,-2,0},
								{0,0,0,-1,-1,-1,0,0,0} };

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (dlg.DoModal() == IDOK)
	{
		MaskLen = (int)(dlg.m_Constant * 8);
		//MaskLen = (int)(dlg.m_Constant * 6);

		stdev2 = dlg.m_Constant;
		stdev1 = 1.6 * stdev2;

		if (MaskLen % 2 == 0)
			MaskLen += 1;

		if (MaskLen < 3)
			MaskLen = 3;

		double** GaussianMask = Image2DMem(MaskLen, MaskLen);

		if (MaskLen == 9) {
			for (i = 0; i < 9; i++) {
				for (j = 0; j < 9; j++) {
					GaussianMask[i][j] = aproxMask[i][j];
				}
			}
		}
		else {
			for (i = 0; i <= MaskLen / 2; i++) {
				x = MaskLen / 2 - i;
				for (j = 0; j <= MaskLen / 2; j++) {
					y = -(MaskLen / 2) + j;
					GaussianMask[i][j] = 100 * (exp(-((double)(x * x + y * y) / (double)(2 * stdev1 * stdev1))) / (2 * M_PI * stdev1 * stdev1) -
										 exp(-((double)(x * x + y * y) / (double)(2 * stdev2 * stdev2))) / (2 * M_PI * stdev2 * stdev2));
				}
			}

			for (i = 0; i < MaskLen / 2; i++) {
				for (j = 1; j <= MaskLen / 2; j++) {
					GaussianMask[i][MaskLen / 2 + j] = GaussianMask[i][MaskLen / 2 - j];
					GaussianMask[MaskLen / 2 + i + 1][j - 1] = GaussianMask[MaskLen / 2 - i - 1][j - 1];
					GaussianMask[MaskLen / 2 + i + 1][MaskLen / 2 + j] = GaussianMask[MaskLen / 2 - i - 1][MaskLen / 2 - j];
				}
				GaussianMask[MaskLen / 2][MaskLen / 2 + i + 1] = GaussianMask[MaskLen / 2][MaskLen / 2 - i - 1];
				GaussianMask[MaskLen / 2 + i + 1][MaskLen / 2] = GaussianMask[MaskLen / 2 - i - 1][MaskLen / 2];
			}
		}



		m_tempImage = OnMaskProcess(m_InputImage, GaussianMask, MaskLen);

		//m_tempImage = OnScale(m_tempImage, m_Re_height, m_Re_width);

		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				if (m_tempImage[i][j] > 255.)
					m_tempImage[i][j] = 255.;
				else if (m_tempImage[i][j] < 0.)
					m_tempImage[i][j] = 0.;
			}
		}

		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				m_OutputImage[i * m_Re_width + j] = (unsigned char)lround(m_tempImage[i][j]);
			}
		}
	}
}


void CImgprocessingsDoc::OnNearest()
{

	CConstantDlg dlg;

	int i, j;
	int ZoomRate;
	double** tempArray;

	if (dlg.DoModal() == IDOK) {
		ZoomRate = (int)dlg.m_Constant;
		if (ZoomRate < 1) {
			AfxMessageBox(L"ZoomRate must be larger than 1");
			return;
		}
	}


	m_Re_height = m_height * ZoomRate;
	m_Re_width = m_width * ZoomRate;
	m_Re_size = m_Re_height * m_Re_width;

	m_tempImage = Image2DMem(m_height, m_width);
	tempArray = Image2DMem(m_Re_height, m_Re_width);
	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_tempImage[i][j] = (double)m_InputImage[i * m_width + j];
		}
	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			tempArray[i][j] = m_tempImage[i / ZoomRate][j / ZoomRate];
		}
	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_Re_width + j] = (unsigned char)tempArray[i][j];
		}
	}
}


void CImgprocessingsDoc::OnBilinear()
{
	CConstantDlg dlg;

	int i, j, i_H, i_W;
	unsigned char newValue;
	double ZoomRate, r_H, r_W, s_H, s_W;
	double C1, C2, C3, C4;

	if (dlg.DoModal() == IDOK) {
		ZoomRate = dlg.m_Constant;
		if (ZoomRate < 1) {
			AfxMessageBox(L"ZoomRate must be larger than 1");
			return;
		}
	}

	m_Re_height = (int)(m_height * ZoomRate);
	m_Re_width = (int)(m_width * ZoomRate);
	m_Re_size = m_Re_height * m_Re_width;

	m_tempImage = Image2DMem(m_height, m_width);
	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_tempImage[i][j] = (double)m_InputImage[i * m_width + j];
		}
	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			r_H = i / ZoomRate;
			r_W = j / ZoomRate;

			i_H = (int)floor(r_H);
			i_W = (int)floor(r_W);

			s_H = r_H - i_H;
			s_W = r_W - i_W;

			if (i_H < 0 || i_W < 0 || i_H >= (m_height-1) || i_W >= (m_width-1)) {
				m_OutputImage[i * m_Re_width + j] = 255;
			}
			else {
				C1 = m_tempImage[i_H][i_W];
				C2 = m_tempImage[i_H][i_W+1];
				C3 = m_tempImage[i_H+1][i_W+1];
				C4 = m_tempImage[i_H+1][i_W];

				newValue = (unsigned char)lround((C1 * (1 - s_H) * (1 - s_W) + C2 * s_W * (1 - s_H) + C3 * s_W * s_H + C4 * (1 - s_W) * s_H));
				m_OutputImage[i * m_Re_width + j] = newValue;
			}
		}
	}
}


void CImgprocessingsDoc::OnBicubic()
{
	CConstantDlg dlg;

	int i, j, k, i_H, i_W;
	unsigned char newValue[4];
	double newOutputValue, ZoomRate, r_H, r_W, s_H, s_W;
	double C1, C2, C3, C4;

	if (dlg.DoModal() == IDOK) {
		ZoomRate = dlg.m_Constant;
		if (ZoomRate < 1) {
			AfxMessageBox(L"ZoomRate must be larger than 1");
			return;
		}
	}

	m_Re_height = (int)(m_height * ZoomRate);
	m_Re_width = (int)(m_width * ZoomRate);
	m_Re_size = m_Re_height * m_Re_width;

	m_tempImage = Image2DMem(m_height, m_width);
	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_tempImage[i][j] = (double)m_InputImage[i * m_width + j];
		}
	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			r_H = i / ZoomRate;
			r_W = j / ZoomRate;

			i_H = (int)floor(r_H);
			i_W = (int)floor(r_W);

			s_H = r_H - i_H;
			s_W = r_W - i_W;

			if (i_H < 1 || i_W < 1 || i_H >= (m_height - 2) || i_W >= (m_width - 2)) {
				m_OutputImage[i * m_Re_width + j] = 255;
			}
			else {
				for (k = -1; k < 3; k++) {
					C1 = m_tempImage[i_H + k][i_W - 1];
					C2 = m_tempImage[i_H + k][i_W];
					C3 = m_tempImage[i_H + k][i_W + 1];
					C4 = m_tempImage[i_H + k][i_W + 2];

					newValue[k+1] = (unsigned char)lround((C1 * OnBiCubicParam(1+s_W) + C2 * OnBiCubicParam(s_W) + C3 * OnBiCubicParam(1-s_W) + C4 * OnBiCubicParam(2-s_W)));
				}
				newOutputValue = (unsigned char)lround((newValue[0] * OnBiCubicParam(1 + s_H) + newValue[1] * OnBiCubicParam(s_H) + 
														newValue[2] * OnBiCubicParam(1 - s_H) + newValue[3] * OnBiCubicParam(2 - s_H)));
				
				if (newOutputValue > 255.)
					newOutputValue = 255.;
				else if (newOutputValue < 0.)
					newOutputValue = 0.;

				m_OutputImage[i * m_Re_width + j] = (unsigned char)newOutputValue;
				
			}
		}
	}
}


double CImgprocessingsDoc::OnBiCubicParam(double value)
{
	double ab = abs(value);
	double a = -1.;

	if (ab < 1) {
		return (a + 2.) * pow(ab, 3.) - (a + 3) * pow(ab, 2.) + 1;
	}
	else if (ab < 2) {
		return a * pow(ab, 3.) - 5 * a * pow(ab, 2.) + 8 * a * ab - 4 * a;
	}
	else {
		return 0.;
	}

	return 0.0;
}


void CImgprocessingsDoc::OnBSpline()
{
	CConstantDlg dlg;

	int i, j, k, i_H, i_W;
	unsigned char newValue[4];
	double newOutputValue, ZoomRate, r_H, r_W, s_H, s_W;
	double C1, C2, C3, C4;

	if (dlg.DoModal() == IDOK) {
		ZoomRate = dlg.m_Constant;
		if (ZoomRate < 1) {
			AfxMessageBox(L"ZoomRate must be larger than 1");
			return;
		}
	}

	m_Re_height = (int)(m_height * ZoomRate);
	m_Re_width = (int)(m_width * ZoomRate);
	m_Re_size = m_Re_height * m_Re_width;

	m_tempImage = Image2DMem(m_height, m_width);
	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_tempImage[i][j] = (double)m_InputImage[i * m_width + j];
		}
	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			r_H = i / ZoomRate;
			r_W = j / ZoomRate;

			i_H = (int)floor(r_H);
			i_W = (int)floor(r_W);

			s_H = r_H - i_H;
			s_W = r_W - i_W;

			if (i_H < 1 || i_W < 1 || i_H >= (m_height - 2) || i_W >= (m_width - 2)) {
				m_OutputImage[i * m_Re_width + j] = 255;
			}
			else {
				for (k = -1; k < 3; k++) {
					C1 = m_tempImage[i_H + k][i_W - 1];
					C2 = m_tempImage[i_H + k][i_W];
					C3 = m_tempImage[i_H + k][i_W + 1];
					C4 = m_tempImage[i_H + k][i_W + 2];

					newValue[k + 1] = (unsigned char)lround((C1 * OnBSplineParam(1 + s_W) + C2 * OnBSplineParam(s_W) + C3 * OnBSplineParam(1 - s_W) + C4 * OnBSplineParam(2 - s_W)));
				}
				newOutputValue = (unsigned char)lround((newValue[0] * OnBSplineParam(1 + s_H) + newValue[1] * OnBSplineParam(s_H) +
														newValue[2] * OnBSplineParam(1 - s_H) + newValue[3] * OnBSplineParam(2 - s_H)));


				m_OutputImage[i * m_Re_width + j] = (unsigned char)newOutputValue;

			}
		}
	}
}


double CImgprocessingsDoc::OnBSplineParam(double value)
{
	double ab = abs(value);

	if (ab < 1) {
		return pow(ab, 3.) / 2 - pow(ab, 2.) + 2./3.;
	}
	else if (ab < 2) {
		return pow(ab, 3.) / -6. + pow(ab, 2.) - 2 * ab + 4. / 3.;
	}
	else {
		return 0.;
	}

	return 0.0;
}


void CImgprocessingsDoc::OnMedianSub()
{
	CConstantDlg dlg;

	int i, j, n, m, M, index = 0;
	double* Mask, Value = 0;

	if (dlg.DoModal() == IDOK) {
		M = (int)dlg.m_Constant;
		if (M < 1) {
			AfxMessageBox(L"ZoomRate must be larger than 1");
			return;
		}
	}

	Mask = (double*)malloc(sizeof(double) * M * M);

	m_Re_height = (m_height + M - 1) / M;
	m_Re_width = (m_width + M - 1) / M;
	m_Re_size = m_Re_height * m_Re_width;

	m_tempImage = Image2DMem(m_height+M-1, m_width+M-1);
	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_tempImage[i][j] = (double)m_InputImage[i * m_width + j];
		}
	}

	/*
	index = 0;
	for (i = 0; i < m_height-(M-1); i+=M) {
		for (j = 0; j < m_width-(M-1); j+=M) {
			for (n = 0; n < M; n++) {
				for (m = 0; m < M; m++) {
					Mask[n * M + m] = m_tempImage[i + n][j + m];
				}
			}
			OnBubbleSort(Mask, M * M);
			Value = Mask[(int)(M * M / 2)];
			//m_OutputImage[index++] = (unsigned char)Value;
			m_OutputImage[i/M * m_Re_width + j/M] = (unsigned char)Value;
		}
	}
	*/

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			for (n = 0; n < M; n++) {
				for (m = 0; m < M; m++) {
					Mask[n * M + m] = m_tempImage[i * M + n][j * M + m];
				}
			}
			OnBubbleSort(Mask, M * M);
			Value = Mask[(int)(M * M / 2)];
			while ((i*M > m_height - M || j * M > m_width - M) && Value == 0) {
				Value = Mask[(int)(M * M / 2) + (++index)];
				if (index == M * M - 1) break;
			}

			m_OutputImage[i * m_Re_width + j] = (unsigned char)Value;

			index = 0;
		}
	}
}


void CImgprocessingsDoc::OnBubbleSort(double* A, int MAX)
{
	int i, j;

	for (i = 0; i < MAX; i++) {
		for (j = 0; j < MAX - 1; j++) {
			if (A[j] > A[j + 1]) {
				*((long long*)A + j) ^= *((long long*)A + j + 1);
				*((long long*)A + j + 1) ^= *((long long*)A + j);
				*((long long*)A + j) ^= *((long long*)A + j + 1);
			}
		}
	}
}


void CImgprocessingsDoc::OnMeanSub()
{
	CConstantDlg dlg;

	int i, j, k, n, m, M, index = 0;
	double* Mask, Value = 0., Sum = 0.;

	if (dlg.DoModal() == IDOK) {
		M = (int)dlg.m_Constant;
		if (M < 1) {
			AfxMessageBox(L"ZoomRate must be larger than 1");
			return;
		}
	}

	Mask = (double*)malloc(sizeof(double) * M * M);

	m_Re_height = (m_height + M - 1) / M;
	m_Re_width = (m_width + M - 1) / M;
	m_Re_size = m_Re_height * m_Re_width;

	m_tempImage = Image2DMem(m_height + M - 1, m_width + M - 1);
	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_tempImage[i][j] = (double)m_InputImage[i * m_width + j];
		}
	}

	/*
	index = 0;
	for (i = 0; i < m_height-(M-1); i+=M) {
		for (j = 0; j < m_width-(M-1); j+=M) {
			for (n = 0; n < M; n++) {
				for (m = 0; m < M; m++) {
					Mask[n * M + m] = m_tempImage[i + n][j + m];
				}
			}
			
			for(k=0; k<M*M; k++){
				Sum += Mask[k];
			}

			Value = Sum / (M*M);
			//m_OutputImage[index++] = (unsigned char)Value;
			m_OutputImage[i/M * m_Re_width + j/M] = (unsigned char)Value;
			Sum = 0.;
		}
	}
	*/

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			
			if (m_width % M != 0 && (i * M > m_height - M || j * M > m_width - M)) {
				for (n = i*M; n < m_height && n - i*M < M; n++) {
					for (m = j*M; m < m_width && m - j*M < M; m++) {
						Mask[index++] = m_tempImage[n][m];
					}
				}

				for (k = 0; k < index; k++) {
					Sum += Mask[k];
				}
				Value = Sum / index;
			}
			else {
				for (n = 0; n < M; n++) {
					for (m = 0; m < M; m++) {
						Mask[n * M + m] = m_tempImage[i * M + n][j * M + m];
					}
				}

				for (k = 0; k < M * M; k++) {
					Sum += Mask[k];
				}
				Value = Sum / (M * M);
			}
			
			m_OutputImage[i * m_Re_width + j] = (unsigned char)Value;

			Sum = 0.; index = 0;
		}
	}
}


void CImgprocessingsDoc::OnTranslation()
{
	int i, j;
	int h_pos = 30, w_pos = 130;
	int h_pos_abs, w_pos_abs;
	double **tempArray;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_tempImage = Image2DMem(m_height, m_width);
	tempArray = Image2DMem(m_Re_height, m_Re_width);
	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_tempImage[i][j] = (double)m_InputImage[i * m_width + j];
		}
	}

	h_pos_abs = h_pos >= 0 ? 0 : -h_pos;
	w_pos_abs = w_pos >= 0 ? 0 : -w_pos;

	for (i = h_pos_abs; i < m_height - h_pos - h_pos_abs; i++) {
		for (j = w_pos_abs; j < m_width - w_pos - w_pos_abs; j++) {
			tempArray[i + h_pos][j + w_pos] = m_tempImage[i][j];
		}
	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_Re_width + j] = (unsigned char)tempArray[i][j];
		}
	}

	free(tempArray[0]);
	free(tempArray);
	free(m_tempImage[0]);
	free(m_tempImage);
}


void CImgprocessingsDoc::OnMirrorHor()
{
	int i, j;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_OutputImage[i * m_width + m_width - 1 -j] = m_InputImage[i*m_width+j];
		}
	}
}


void CImgprocessingsDoc::OnMirrorVer()
{
	int i, j;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_OutputImage[(m_height - 1 -i) * m_width + j] = m_InputImage[i * m_width + j];
		}
	}
}


void CImgprocessingsDoc::OnMirrorXy()
{
	int i, j;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	/*
	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_OutputImage[j * m_width + i] = m_InputImage[i * m_width + j];
		}
	}
	*/

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_OutputImage[(m_height-1-j) * m_width + (m_width-1-i)] = m_InputImage[i * m_width + j];
		}
	}

}


void CImgprocessingsDoc::OnMirrorHorVer()
{
	int i, j;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_OutputImage[(m_height - 1 - i) * m_width + m_width - 1 - j] = m_InputImage[i * m_width + j];
		}
	}
}


void CImgprocessingsDoc::OnRotation()
{
	CConstantNoLimitDlg dlg;

	int i, j, CenterH, CenterW, ReCenterH, ReCenterW, newH, newW, H, degree = 45;
	double Radian, ** tempArray, value;

	if (dlg.DoModal() == IDOK) {
		degree = (int)dlg.m_Constant;
	}

	Radian = (double)degree * M_PI / 180.;

	DecideLen:
	if (0 <= Radian && Radian <= M_PI/2) {
		m_Re_height = (int)(m_height * cos(Radian) + m_width * cos(M_PI / 2 - Radian));
		m_Re_width = (int)(m_height * cos(M_PI / 2 - Radian) + m_width * cos(Radian));
	}
	else if (Radian < 0) {
		Radian += M_PI/2;
		goto DecideLen;
	}
	else {
		Radian -= M_PI/2;
		goto DecideLen;
	}
	Radian = (double)degree * M_PI / 180.;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	H = m_Re_height - 1;
	CenterH = m_height / 2;
	CenterW = m_width / 2;
	ReCenterH = m_Re_height / 2;
	ReCenterW = m_Re_width / 2;

	m_tempImage = Image2DMem(m_height, m_width);
	tempArray = Image2DMem(m_Re_height, m_Re_width);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_tempImage[i][j] = m_InputImage[i * m_width + j];
		}
	}


	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			newW = (int)(((H-i) - ReCenterH) * sin(Radian) + (j - ReCenterW) * cos(Radian) + CenterW);
			newH = (m_height-1) - (int)(((H-i) - ReCenterH) * cos(Radian) - (j - ReCenterW) * sin(Radian) + CenterH);

			if (newH < 0 || newH >= m_height || newW < 0 || newW >= m_width) {
				if (i == m_Re_height - 1 || j == m_Re_width - 1 || i == 0 || j == 0) {
					value = 0;;
				}
				else {
					value = 255;
				}
			}
			else {
				value = m_tempImage[newH][newW];
			}
			tempArray[i][j] = value;
		}
	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_Re_width + j] = (unsigned char)tempArray[i][j];
		}
	}

	free(tempArray[0]);
	free(tempArray);
	free(m_tempImage[0]);
	free(m_tempImage);
}


void CImgprocessingsDoc::OnWarping()
{
	int i, j, k, pointx = m_width / 2, pointy = m_height / 2;
	int pointWx = 64, pointWy = 64, X, Y;
	double u, h, d, weight, t_x, t_y, totalWeight, srcx, srcy, temp , a = 0.001, b = 2.0, p = 0.75;

	typedef struct {
		int Lx;
		int Ly;
		int L1x;
		int L1y;
	}cntrLine;

	/*
	FILE* file;
	fopen_s(&file, "C:\\Users\\darak\\OneDrive\\Desktop\\check.txt", "wb+");
	if (file == (FILE*)0) return;
	*/

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	
	cntrLine cLine[8], cLineSrc[8];
	int cntrLnum = sizeof(cLine) / sizeof(cntrLine);

	
	//제어선 초기화
	for (i = 0; i < cntrLnum; i++) {
		cLine[i].Lx = pointx + pointWx;
		cLine[i].Ly = pointy + pointWy;
		cLine[i].L1x = (i % 2) * (m_Re_width - 1);
		cLine[i].L1y = (i / 2) * (m_Re_height - 1);
	}

	cLine[4].Lx = pointx + pointWx;
	cLine[4].Ly = pointy + pointWy;
	cLine[4].L1x = 128;
	cLine[4].L1y = 0;

	cLine[5].Lx = pointx + pointWx;
	cLine[5].Ly = pointy + pointWy;
	cLine[5].L1x = 0;
	cLine[5].L1y = 128;

	cLine[6].Lx = pointx + pointWx;
	cLine[6].Ly = pointy + pointWy;
	cLine[6].L1x = 256;
	cLine[6].L1y = 128;

	cLine[7].Lx = pointx + pointWx;
	cLine[7].Ly = pointy + pointWy;
	cLine[7].L1x = 128;
	cLine[7].L1y = 256;

	for (i = 0; i < cntrLnum; i++) {
		cLineSrc[i].Lx = pointx;
		cLineSrc[i].Ly = pointy;
		cLineSrc[i].L1x = (i % 2) * (m_Re_width - 1);
		cLineSrc[i].L1y = (i / 2) * (m_Re_height - 1);
	}
	
	cLineSrc[4].Lx = pointx + pointWx;
	cLineSrc[4].Ly = pointy + pointWy;
	cLineSrc[4].L1x = 128;
	cLineSrc[4].L1y = 0;

	cLineSrc[5].Lx = pointx + pointWx;
	cLineSrc[5].Ly = pointy + pointWy;
	cLineSrc[5].L1x = 0;
	cLineSrc[5].L1y = 128;

	cLineSrc[6].Lx = pointx + pointWx;
	cLineSrc[6].Ly = pointy + pointWy;
	cLineSrc[6].L1x = 256;
	cLineSrc[6].L1y = 128;

	cLineSrc[7].Lx = pointx + pointWx;
	cLineSrc[7].Ly = pointy + pointWy;
	cLineSrc[7].L1x = 128;
	cLineSrc[7].L1y = 256;

	//계산
	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			
			t_x = 0; t_y = 0; totalWeight = 0;
			for (k = 0; k < cntrLnum; k++) {
				
				//수직 교차점의 위치 계산
				u = (double)((j - cLine[k].Lx) * (cLine[k].L1x - cLine[k].Lx) + (i - cLine[k].Ly) * (cLine[k].L1y - cLine[k].Ly)) /
					(pow(cLine[k].L1x - cLine[k].Lx, 2) + pow(cLine[k].L1y - cLine[k].Ly, 2));

				//제이선으로부터의 수직 변위 계산
				h = (double)((i - cLine[k].Ly) * (cLine[k].L1x - cLine[k].Lx) - (j - cLine[k].Lx) * (cLine[k].L1y - cLine[k].Ly)) /
					sqrt(pow(cLine[k].L1x - cLine[k].Lx, 2) + pow(cLine[k].L1y - cLine[k].Ly, 2));

				//입력 영상에서의 대응 픽셀의 위치 계산
				srcx = cLineSrc[k].Lx + u * (cLineSrc[k].L1x - cLineSrc[k].Lx) - (double)(h*(cLineSrc[k].L1y - cLineSrc[k].Ly)) / 
						sqrt(pow(cLineSrc[k].L1x - cLineSrc[k].Lx, 2) + pow(cLineSrc[k].L1y - cLineSrc[k].Ly, 2));

				srcy = cLineSrc[k].Ly + u * (cLineSrc[k].L1y - cLineSrc[k].Ly) - (double)(h * (cLineSrc[k].L1x - cLineSrc[k].Lx)) /
					sqrt(pow(cLineSrc[k].L1x - cLineSrc[k].Lx, 2) + pow(cLineSrc[k].L1y - cLineSrc[k].Ly, 2));
				
				
				//픽셀과 제어선 사이의 거리
				if (u < 0) {
					d = sqrt(pow(j - cLine[k].Lx, 2) + pow(i - cLine[k].Ly, 2));
				}
				else if (0 <= u && u <= 1){//sqrt(pow(cLine[k].L1x - cLine[k].Lx, 2) + pow(cLine[k].L1y - cLine[k].Ly, 2))) {
					d = abs(h);
				}
				else if (u > 1){//sqrt(pow(cLine[k].L1x - cLine[k].Lx, 2) + pow(cLine[k].L1y - cLine[k].Ly, 2))) {
					d = sqrt(pow(j - cLine[k].L1x, 2) + pow(i - cLine[k].L1y, 2));
				}

				//제어선의 가중치 계산
				weight = pow(pow(sqrt(pow(cLine[k].L1x - cLine[k].Lx, 2) + pow(cLine[k].L1y - cLine[k].Ly, 2)), p) / (a + d), b);

				//입력 영상의 대응 픽셀과의 변위 누적
				t_x += (srcx - j) * weight;
				t_y += (srcy - i) * weight;
				totalWeight += weight;
				
			}
			
			//fprintf(file, "(%.4lf, %.4lf, %.4lf)\n", t_x , t_y, totalWeight);
			X = j + (int)lround(t_x / totalWeight);
			Y = i + (int)lround(t_y / totalWeight);
			
			if (X < 0) {
				X = 0;
			}
			else if (X >= m_width) {
				X = m_width - 1;
			}

			if (Y < 0) {
				Y = 0;
			}
			else if (Y >= m_height) {
				Y = m_height - 1;
			}
			

			
			m_OutputImage[i * m_Re_width + j] = m_InputImage[Y * m_width + X];
		}
	}
	//fclose(file);

}


void CImgprocessingsDoc::OnFrameSum()
{
	CFile File;
	CFileDialog OpenDlg(TRUE);

	int i;
	double a = 0.5;
	unsigned char* temp;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (OpenDlg.DoModal() == IDOK) {
		File.Open(OpenDlg.GetPathName(), CFile::modeRead);

		if (File.GetLength() == (unsigned)m_width * m_height) {
			temp = new unsigned char[m_size];

			File.Read(temp, m_size);
			File.Close();

			for (i = 0; i < m_size; i++) {
				if (a * m_InputImage[i] + (1 - a) * temp[i] > 255)
					m_OutputImage[i] = 255;
				else
					m_OutputImage[i] = a * m_InputImage[i] + (1 - a) * temp[i];
			}
		}
		else {
			AfxMessageBox(L"Image size not matched");
			return;
		}
	}
}


void CImgprocessingsDoc::OnFrameSub()
{
	CFile File;
	CFileDialog OpenDlg(TRUE);

	int i;
	unsigned char* temp;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (OpenDlg.DoModal() == IDOK) {
		File.Open(OpenDlg.GetPathName(), CFile::modeRead);

		if (File.GetLength() == (unsigned)m_width * m_height) {
			temp = new unsigned char[m_size];

			File.Read(temp, m_size);
			File.Close();

			for (i = 0; i < m_size; i++) {
				if (m_InputImage[i] - temp[i] < 0)
					m_OutputImage[i] = 0;
				else
					m_OutputImage[i] = m_InputImage[i] - temp[i];
			}
		}
		else {
			AfxMessageBox(L"Image size not matched");
			return;
		}
	}
}


void CImgprocessingsDoc::OnFrameMul()
{
	CFile File;
	CFileDialog OpenDlg(TRUE);

	int i;
	unsigned char* temp;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (OpenDlg.DoModal() == IDOK) {
		File.Open(OpenDlg.GetPathName(), CFile::modeRead);

		if (File.GetLength() == (unsigned)m_width * m_height) {
			temp = new unsigned char[m_size];

			File.Read(temp, m_size);
			File.Close();

			for (i = 0; i < m_size; i++) {
				if (m_InputImage[i] * temp[i] > 255)
					m_OutputImage[i] = 255;
				else
					m_OutputImage[i] = m_InputImage[i] * temp[i];
			}
		}
		else {
			AfxMessageBox(L"Image size not matched");
			return;
		}
	}
}


void CImgprocessingsDoc::OnFrameDiv()
{
	CFile File;
	CFileDialog OpenDlg(TRUE);

	int i;
	unsigned char* temp;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (OpenDlg.DoModal() == IDOK) {
		File.Open(OpenDlg.GetPathName(), CFile::modeRead);

		if (File.GetLength() == (unsigned)m_width * m_height) {
			temp = new unsigned char[m_size];

			File.Read(temp, m_size);
			File.Close();

			for (i = 0; i < m_size; i++) {
				if (m_InputImage[i] == 0)
					m_OutputImage[i] = 0;
				else if (temp[i] == 0)
					m_OutputImage[i] = 255;
				else
					m_OutputImage[i] = (unsigned char)(m_InputImage[i] / temp[i]);
			}
		}
		else {
			AfxMessageBox(L"Image size not matched");
			return;
		}
	}
}


void CImgprocessingsDoc::OnFrameMean()

{
	CFile File;
	CFileDialog OpenDlg(TRUE);

	int i;
	unsigned char* temp;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (OpenDlg.DoModal() == IDOK) {
		File.Open(OpenDlg.GetPathName(), CFile::modeRead);

		if (File.GetLength() == (unsigned)m_width * m_height) {
			temp = new unsigned char[m_size];

			File.Read(temp, m_size);
			File.Close();

			for (i = 0; i < m_size; i++) {
				m_OutputImage[i] = (unsigned char)((m_InputImage[i] + temp[i]) / 2);
			}
		}
		else {
			AfxMessageBox(L"Image size not matched");
			return;
		}
	}
}


void CImgprocessingsDoc::OnFrameAnd()
{
	CFile File;
	CFileDialog OpenDlg(TRUE);

	int i;
	unsigned char* temp;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (OpenDlg.DoModal() == IDOK) {
		File.Open(OpenDlg.GetPathName(), CFile::modeRead);

		if (File.GetLength() == (unsigned)m_width * m_height) {
			temp = new unsigned char[m_size];

			File.Read(temp, m_size);
			File.Close();

			for (i = 0; i < m_size; i++) {
				m_OutputImage[i] = (unsigned char)(m_InputImage[i] & temp[i]);
			}
		}
		else {
			AfxMessageBox(L"Image size not matched");
			return;
		}
	}
}


void CImgprocessingsDoc::OnFrameOr()
{
	CFile File;
	CFileDialog OpenDlg(TRUE);

	int i;
	unsigned char* temp;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (OpenDlg.DoModal() == IDOK) {
		File.Open(OpenDlg.GetPathName(), CFile::modeRead);

		if (File.GetLength() == (unsigned)m_width * m_height) {
			temp = new unsigned char[m_size];

			File.Read(temp, m_size);
			File.Close();

			for (i = 0; i < m_size; i++) {
				m_OutputImage[i] = (unsigned char)(m_InputImage[i] | temp[i]);
			}
		}
		else {
			AfxMessageBox(L"Image size not matched");
			return;
		}
	}
}


void CImgprocessingsDoc::OnFrameComb()
{
	CFile File;
	CFileDialog OpenDlg(TRUE);

	int i;
	unsigned char* temp, *masktemp, maskvalue;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	AfxMessageBox(L"합성할 영상 선택");

	if (OpenDlg.DoModal() == IDOK) {
		File.Open(OpenDlg.GetPathName(), CFile::modeRead);

		if (File.GetLength() == (unsigned)m_width * m_height) {
			temp = new unsigned char[m_size];

			File.Read(temp, m_size);
			File.Close();
		}
		else {
			AfxMessageBox(L"Image size not matched");
			return;
		}
	}

	AfxMessageBox(L"마스크 선택");

	if (OpenDlg.DoModal() == IDOK) {
		File.Open(OpenDlg.GetPathName(), CFile::modeRead);

		if (File.GetLength() == (unsigned)m_width * m_height) {
			masktemp = new unsigned char[m_size];
			File.Read(masktemp, m_size);
			File.Close();
		}
		else {
			AfxMessageBox(L"mask size not matched");
			return;
		}
	}

	for (i = 0; i < m_size; i++) {
		maskvalue = 255 - masktemp[i];
		m_OutputImage[i] = (m_InputImage[i] & masktemp[i]) | (temp[i] & maskvalue);
	}
}


void CImgprocessingsDoc::OnBinaryErosion(int num, BOOL useDlg, BOOL useMaskInput, double InMask[][3])
{
	CConstantDlg dlg;

	int i, j, n, m;
	double MaskDefault[3][3] = { {255., 255., 255.}, {255., 255., 255.}, {255., 255., 255.} };
	double** tempInput, s = 0., (*Mask)[3];

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_size;
	
	m_OutputImage = new unsigned char[m_Re_size];

	tempInput = Image2DMem(m_height + 2, m_width+2);

	if (useDlg) {
		if (dlg.DoModal() == IDOK) {
			num = (int)dlg.m_Constant;
			if (num < 1) {
				AfxMessageBox(L"num must be larger than 1");
				return;
			}
		}
	}

	if (useMaskInput) {
		Mask = InMask;
	}
	else {
		Mask = MaskDefault;
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			tempInput[i + 1][j + 1] = (double)m_InputImage[i * m_width + j];
		}
	}

	while (num-- >= 1) {
		for (i = 0; i < m_height; i++) {
			for (j = 0; j < m_width; j++) {
				
				for (n = 0; n < 3; n++) {
					for (m = 0; m < 3; m++) {
						if (Mask[n][m] == tempInput[i + n][j + m]) {
							s += 1.;
						}
					}
				}
	
				/*
				for (n = -1; n < 3; n++) {
					if (tempInput[i + 1 + (n % 2)][j + 1 + ((n - 1) % 2)] == 255) {
						s += 1.;
					}
				}
				*/
				if (s == 9.) {
					m_OutputImage[i * m_Re_width + j] = (unsigned char)255.;
				}
				else {
					m_OutputImage[i * m_Re_width + j] = (unsigned char)0.;
				}
				s = 0.;
			}
		}

		for (i = 0; i < m_height; i++) {
			for (j = 0; j < m_width; j++) {
				tempInput[i + 1][j + 1] = (double)m_OutputImage[i * m_width + j];
			}
		}
	}

	free(tempInput[0]);
	free(tempInput);
}


void CImgprocessingsDoc::OnBinaryDilation(int num, BOOL useDlg, BOOL useMaskInput, double InMask[][3])
{
	CConstantDlg dlg;

	int i, j, n, m;
	double MaskDefault[3][3] = { {0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.} };
	double** tempInput, s = 0., (*Mask)[3];

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_size;

	m_OutputImage = new unsigned char[m_Re_size];

	tempInput = Image2DMem(m_height + 2, m_width + 2);

	if (useDlg) {
		if (dlg.DoModal() == IDOK) {
			num = (int)dlg.m_Constant;
			if (num < 1) {
				AfxMessageBox(L"num must be larger than 1");
				return;
			}
		}
	}

	if (useMaskInput) {
		Mask = InMask;
	}
	else {
		Mask = MaskDefault;
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			tempInput[i + 1][j + 1] = (double)m_InputImage[i * m_width + j];
		}
	}

	while (num-- >= 1) {
		for (i = 0; i < m_height; i++) {
			for (j = 0; j < m_width; j++) {
				for (n = 0; n < 3; n++) {
					for (m = 0; m < 3; m++) {
						if (Mask[n][m] == tempInput[i + n][j + m]) {
							s += 1.;
						}
					}
				}
				if (s == 9.) {
					m_OutputImage[i * m_Re_width + j] = (unsigned char)0.;
				}
				else {
					m_OutputImage[i * m_Re_width + j] = (unsigned char)255.;
				}
				s = 0.;
			}
		}

		for (i = 0; i < m_height; i++) {
			for (j = 0; j < m_width; j++) {
				tempInput[i + 1][j + 1] = (double)m_OutputImage[i * m_width + j];
			}
		}
	}

	free(tempInput[0]);
	free(tempInput);
}


void CImgprocessingsDoc::OnBinarySkeleton()
{
	int i, j, num = 0, topItemNum = 10, check = 0;
	double** tempOutput, **tempErosion, **tempOpen;
	double Mask[3][3] = { {255., 255., 255.}, {255., 255., 255.}, {255., 255., 255.} };

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];
	m_tempImage = Image2DMem(m_height, m_width);
	tempOutput = (double**)malloc(sizeof(double*) * topItemNum);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_tempImage[i][j] = (double)m_InputImage[i * m_width + j];
		}
	}

	while(TRUE) {
		check = 0;
		num++;

		if (num > topItemNum) {
			topItemNum += 10;
			tempOutput = (double**)realloc(tempOutput, sizeof(double*) * topItemNum);
		}
		tempOutput[num - 1] = (double*)malloc(sizeof(double) * m_height * m_width);
		
		
		//침식
		OnBinaryErosion(1,FALSE,TRUE,Mask);
		
		for (i = 0; i < m_height; i++) {
			for (j = 0; j < m_width; j++) {
				tempOutput[num - 1][i * m_width + j] = m_InputImage[i * m_width + j];
				m_InputImage[i * m_width + j] = m_OutputImage[i * m_Re_width + j];
			}
		}

		//열림
		OnBinaryDilation(1, FALSE, FALSE, NULL);


		//차집합
		for (i = 0; i < m_height; i++) {
			for (j = 0; j < m_width; j++) {
				tempOutput[num-1][i * m_width + j] -= (double)m_OutputImage[i * m_width + j];
				if (m_OutputImage[i * m_width + j] != 0) {
					check = 1;
				}
			}
		}

		if (!check) {
			break;
		}
	}


	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_OutputImage[i * m_Re_width + j] = 0;
			m_InputImage[i * m_width + j] = (unsigned char)m_tempImage[i][j];
		}
	}

	while (num-- > 0) {
		for (i = 0; i < m_height; i++) {
			for (j = 0; j < m_width; j++) {
				m_OutputImage[i * m_Re_width + j] |= (int)tempOutput[num][i * m_width + j];
			}
		}
	}
}


void CImgprocessingsDoc::OnBinaryEdgeDetec()
{
	int i, j;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];
	m_OutputImageLB = new unsigned char[m_Re_size];
	m_OutputImageRB = new unsigned char[m_Re_size];

	OnBinaryErosion(1, FALSE, FALSE, NULL);

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImageLB[i * m_Re_width + j] = m_InputImage[i * m_width + j] - m_OutputImage[i * m_Re_width + j];
		}
	}

	OnBinaryDilation(1, FALSE, FALSE, NULL);

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImageRB[i * m_Re_width + j] =  m_OutputImage[i * m_Re_width + j] - m_InputImage[i * m_width + j];
		}
	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_Re_width + j] = m_OutputImageLB[i * m_Re_width + j] + m_OutputImageRB[i * m_Re_width + j];
		}
	}

	m_printImageBool = TRUE;
}


void CImgprocessingsDoc::OnGrayErosion()
{
	int i, j, n, m, h;
	double Mask[9], MIN = 10000.;
	double** tempInput, s = 0.;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	tempInput = Image2DMem(m_height + 2, m_width + 2);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			tempInput[i + 1][j + 1] = (double)m_InputImage[i * m_width + j];
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			MIN = 10000.;
			for (n = 0; n < 3; n++) {
				for (m = 0; m < 3; m++) {
					Mask[n * 3 + m] = tempInput[i + n][j + m];
				}
			}
			for (h = 0; h < 9; h++) {
				if (Mask[h] < MIN)
					MIN = Mask[h];
			}
			m_OutputImage[i * m_Re_width + j] = (unsigned char)MIN;
		}
	}
}


void CImgprocessingsDoc::OnGrayDilation()
{
	int i, j, n, m, h;
	double Mask[9], MAX = 0.;
	double** tempInput, s = 0.;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	tempInput = Image2DMem(m_height + 2, m_width + 2);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			tempInput[i + 1][j + 1] = (double)m_InputImage[i * m_width + j];
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			MAX = 0.;
			for (n = 0; n < 3; n++) {
				for (m = 0; m < 3; m++) {
					Mask[n * 3 + m] = tempInput[i + n][j + m];
				}
			}
			for (h = 0; h < 9; h++) {
				if (Mask[h] > MAX)
					MAX = Mask[h];
			}
			m_OutputImage[i * m_Re_width + j] = (unsigned char)MAX;
		}
	}
}
