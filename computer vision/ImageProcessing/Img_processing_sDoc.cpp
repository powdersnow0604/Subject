
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
	pDlg = new CWaveletTransformDlg(this);
}

CImgprocessingsDoc::~CImgprocessingsDoc()
{
	delete pDlg;
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
	else if (File.GetLength() == (unsigned long long)(340 * 300)) {
		m_height = 340;
		m_width = 300;
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


#define MASK_PROCESS_
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


double** CImgprocessingsDoc::OnMaskProcessArr(unsigned char* Target, double Mask[][3])
{
	int i, j, n, m;
	double** tempInputImage, ** tempOutputImage, S = 0.;

	tempInputImage = Image2DMem(m_height + 2, m_width + 2);
	tempOutputImage = Image2DMem(m_height, m_width);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			tempInputImage[i + 1][j + 1] = (double)Target[i * m_width + j];
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			for (n = 0; n < 3; n++) {
				for (m = 0; m < 3; m++) {
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
					if (fabs(m_tempImage[i + 1][j + 1] - m_tempImage[i + n][j + m]) >= max) {
						max = fabs(m_tempImage[i + 1][j + 1] - m_tempImage[i + n][j + m]);
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
	double ab = fabs(value);
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
	double ab = fabs(value);

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
	
	//m_Re_height = m_height;
	//m_Re_width = m_width;
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
	int x1, x2, y1, y2;
	int src_x1, src_x2, src_y1, src_y2;
	double u, h, d, weight, t_x, t_y, totalWeight, srcx, srcy , Value, a = 0.001, b = 2.0, p = 0.75; 
	double src_line_length, dest_line_length;

	typedef struct {
		int Px;
		int Py;
		int Qx;
		int Qy;
	}cntrLine;


	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	//제어선
	cntrLine cLine[4] = { {64, 64, 192, 64}, { 192,64,255,255 }, { 64,64,0,255 }, { 0,255,255,255 } };
	cntrLine cLineSrc[4] = { {64,64,192,64},{192,64,192,192},{64,64,64,192},{64,192,192,192} };
	int cntrLnum = sizeof(cLine) / sizeof(cntrLine);


	//계산
	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
		
			t_x = 0; t_y = 0; totalWeight = 0;
			for (k = 0; k < cntrLnum; k++) {
				
				x1 = cLine[k].Px;
				x2 = cLine[k].Qx;
				y1 = cLine[k].Py;
				y2 = cLine[k].Qy;

				src_x1 = cLineSrc[k].Px;
				src_x2 = cLineSrc[k].Qx;
				src_y1 = cLineSrc[k].Py;
				src_y2 = cLineSrc[k].Qy;

				dest_line_length = sqrt((double)(x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
				src_line_length = sqrt((double)(src_x2 - src_x1) * (src_x2 - src_x1) + (src_y2 - src_y1) * (src_y2 - src_y1));

				//수직 교차점의 위치 계산
				u = (double)((j - x1) * (x2 - x1) + (i - y1) * (y2 - y1)) / (double)((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));

				//제이선으로부터의 수직 변위 계산
				h = (double)((i - y1) * (x2 - x1) - (j - x1) * (y2 - y1)) / dest_line_length;

				//입력 영상에서의 대응 픽셀의 위치 계산
				srcx = src_x1 + u * (src_x2 - src_x1) - h * (src_y2 - src_y1) / src_line_length;

				srcy = src_y1 + u * (src_y2 - src_y1) + h * (src_x2 - src_x1) / src_line_length;
				
				//픽셀과 제어선 사이의 거리
				if (u < 0) {
					d = sqrt(pow(j - x1, 2) + pow(i - y1, 2));
				}
				else if (0 <= u && u <= 1){
					d = fabs(h);
				}
				else if (u > 1){
					d = sqrt(pow(j - x2, 2) + pow(i - y1, 2));
				}

				//제어선의 가중치 계산
				weight = pow(pow(dest_line_length, p) / (a + d), b);

				//입력 영상의 대응 픽셀과의 변위 누적
				t_x += (srcx - j) * weight;
				t_y += (srcy - i) * weight;
				totalWeight += weight;
				
			}
			
			X = j + (int)(t_x / totalWeight+0.5);
			Y = i + (int)(t_y / totalWeight+0.5);
			
			/*
			if (X < 0)	X = 0;
			else if (X > m_width-1)	X = m_width - 1;
			if (Y < 0)	Y = 0;
			else if (Y > m_height-1)	Y = m_height - 1;
			*/

			if (X < 0 || X > m_width - 1 || Y < 0 || Y > m_height - 1)
				Value = 0;
			else
				Value = m_InputImage[Y * m_width + X];
			
			m_OutputImage[i * m_Re_width + j] = Value;
		}
	}
}


void CImgprocessingsDoc::OnGeometryMorphing()
{
	CFile File;
	CFileDialog OpenDlg(TRUE);

	typedef struct {
		int Px;
		int Py;
		int Qx;
		int Qy;
	}cntrLine;

	int NUM_FRAMES = 10;
	double u; // 수직 교차점의 위치 
	double h; // 제어선으로부터 픽셀의 수직 변위
	double d; // 제어선과 픽셀 사이의 거리
	double tx, ty; // 결과영상 픽셀에 대응되는 입력 영상 픽셀 사이의 변위의 합
	double xp, yp; // 각 제어선에 대해 계산된 입력 영상의 대응되는 픽셀 위치
	double weight; // 각 제어선의 가중치
	double totalWeight; // 가중치의 합
	double a = 0.001, b = 2.0, p = 0.75;
	unsigned char** warpedImg;
	unsigned char** warpedImg2;
	int frame;
	double fweight;
	cntrLine warp_lines[23];
	double tx2, ty2, xp2, yp2;
	int dest_x1, dest_y1, dest_x2, dest_y2, source_x2, source_y2;
	int x1, x2, y1, y2, src_x1, src_y1, src_x2, src_y2;
	double src_line_length, src2_line_length, dest_line_length;
	int i, j;
	int num_lines = 23; // 제어선의 수
	int line, x, y, source_x, source_y;


	cntrLine source_lines[23] =
	{ {116,7,207,5},{34,109,90,21},{55,249,30,128},{118,320,65,261},
	{123,321,171,321},{179,319,240,264},{247,251,282,135},{281,114,228,8},
	{78,106,123,109},{187,115,235,114},{72,142,99,128},{74,150,122,154},
	{108,127,123,146},{182,152,213,132},{183,159,229,157},{219,131,240,154},
	{80,246,117,212},{127,222,146,223},{154,227,174,221},{228,252,183,213},
	{114,255,186,257},{109,258,143,277},{152,278,190,262} };

	cntrLine dest_lines[23] =
	{ {120,8,200,6},{12,93,96,16},{74,271,16,110},{126,336,96,290},
	{142,337,181,335},{192,335,232,280},{244,259,288,108},{285,92,212,13},
	{96,135,136,118},{194,119,223,125},{105,145,124,134},{110,146,138,151},
	{131,133,139,146},{188,146,198,134},{189,153,218,146},{204,133,221,140},
	{91,268,122,202},{149,206,159,209},{170,209,181,204},{235,265,208,199},
	{121,280,205,284},{112,286,160,301},{166,301,214,287} };

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);
	m_OutputImageLB = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);
	m_OutputImageRB = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	if (OpenDlg.DoModal() == IDOK) {
		File.Open(OpenDlg.GetPathName(), CFile::modeRead);

		if (File.GetLength() == (unsigned long long)m_width * m_height) {

			File.Read(m_OutputImage, m_size);
			File.Close();
		}
		else {
			AfxMessageBox(L"Image size not matched");
			return;
		}
	}

	warpedImg = OnMem2DAllocUnsigned(m_height, m_width);
	warpedImg2 = OnMem2DAllocUnsigned(m_height, m_width);

	for (i = 0; i < NUM_FRAMES; i++) {
		m_MorphedImg[i] = OnMem2DAllocUnsigned(m_height, m_width);
	}

	for (frame = 1; frame <= NUM_FRAMES; frame++) {
		// 중간 프레임에 대한 가중치 계산
		fweight = (double)(frame) / (NUM_FRAMES+1);

		// 중간 프레임에 대한 제어선 계산
		for (line = 0; line < num_lines; line++) {
			warp_lines[line].Px = (int)(source_lines[line].Px +
				(dest_lines[line].Px - source_lines[line].Px) * fweight);
			warp_lines[line].Py = (int)(source_lines[line].Py +
				(dest_lines[line].Py - source_lines[line].Py) * fweight);
			warp_lines[line].Qx = (int)(source_lines[line].Qx +
				(dest_lines[line].Qx - source_lines[line].Qx) * fweight);
			warp_lines[line].Qy = (int)(source_lines[line].Qy +
				(dest_lines[line].Qy - source_lines[line].Qy) * fweight);
		}

		for (y = 0; y < m_Re_height; y++) {
			for (x = 0; x < m_Re_width; x++) {

				totalWeight = 0.0;
				tx = 0.0;
				ty = 0.0;
				tx2 = 0.0;
				ty2 = 0.0;

				// 각 제어선에 대하여
				for (line = 0; line < num_lines; line++) {
					x1 = warp_lines[line].Px;
					y1 = warp_lines[line].Py;
					x2 = warp_lines[line].Qx;
					y2 = warp_lines[line].Qy;
					dest_line_length = sqrt((double)(x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));

					u = (double)((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) /
						(double)((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
					h = (double)((y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)) / dest_line_length;

					// 제어선과 픽셀 사이의 거리 계산
					if (u < 0) d = sqrt((double)(x - x1) * (x - x1) + (y - y1) * (y - y1));
					else if (u > 1) d = sqrt((double)(x - x2) * (x - x2) + (y - y2) * (y - y2));
					else d = fabs(h);

					src_x1 = source_lines[line].Px;
					src_y1 = source_lines[line].Py;
					src_x2 = source_lines[line].Qx;
					src_y2 = source_lines[line].Qy;
					src_line_length = sqrt((double)(src_x2 - src_x1) * (src_x2 - src_x1) + (src_y2 - src_y1) * (src_y2 - src_y1));

					dest_x1 = dest_lines[line].Px; 
					dest_y1 = dest_lines[line].Py;
					dest_x2 = dest_lines[line].Qx;
					dest_y2 = dest_lines[line].Qy;
					src2_line_length = sqrt((double)(dest_x2 - dest_x1) * (dest_x2 - dest_x1) + (dest_y2 - dest_y1) * (dest_y2 - dest_y1));

					// 입력 영상 1에서의 대응 픽셀 위치 계산
					xp = src_x1 + u * (src_x2 - src_x1) -
						h * (src_y2 - src_y1) / src_line_length;
					yp = src_y1 + u * (src_y2 - src_y1) +
						h * (src_x2 - src_x1) / src_line_length;
					// 입력 영상 2에서의 대응 픽셀 위치 계산
					xp2 = dest_x1 + u * (dest_x2 - dest_x1) -
						h * (dest_y2 - dest_y1) / src2_line_length;
					yp2 = dest_y1 + u * (dest_y2 - dest_y1) +
						h * (dest_x2 - dest_x1) / src2_line_length;

					// 제어선에 대한 가중치 계산
					weight = pow((pow((double)(src2_line_length), p) / (a + d)), b);

					// 입력 영상 1의 대응 픽셀과의 변위 계산
					tx += (xp - x) * weight;
					ty += (yp - y) * weight;

					// 입력 영상 2의 대응 픽셀과의 변위 계산
					tx2 += (xp2 - x) * weight;
					ty2 += (yp2 - y) * weight;

					totalWeight += weight;
				}

				// 입력 영상 1의 대응 픽셀 위치 계산
				source_x = x + (int)(tx / totalWeight + 0.5);
				source_y = y + (int)(ty / totalWeight + 0.5);

				// 입력 영상 2의 대응 픽셀 위치 계산
				source_x2 = x + (int)(tx2 / totalWeight + 0.5);
				source_y2 = y + (int)(ty2 / totalWeight + 0.5);

				// 영상의 경계를 벗어나는지 검사
				if (source_x < 0) source_x = 0;
				if (source_x > m_width - 1) source_x = m_width-1;
				if (source_y < 0) source_y = 0;
				if (source_y > m_height - 1) source_y = m_height-1;
				if (source_x2 < 0) source_x2 = 0;
				if (source_x2 > m_width - 1) source_x2 = m_width - 1;
				if (source_y2 < 0) source_y2 = 0;
				if (source_y2 > m_height - 1) source_y2 = m_height - 1;

				warpedImg[y][x] = m_InputImage[source_y*m_width + source_x];
				warpedImg2[y][x] = m_OutputImage[source_y2*m_Re_width + source_x2];
			}
		}

		// 모핑 결과 합병
		for (y = 0; y < m_Re_height; y++) {
			for (x = 0; x < m_Re_width; x++) {
				int val = (int)((1.0 - fweight) * warpedImg[y][x] +
					fweight * warpedImg2[y][x]);
				if (val < 0) val = 0;
				if (val > 255) val = 255;
				m_MorphedImg[frame - 1][y][x] = val;
			}
		}

	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImageLB[i * m_Re_width + j] = m_MorphedImg[4][i][j];
			m_OutputImageRB[i * m_Re_width + j] = m_MorphedImg[7][i][j];
		}
	}

	m_printImageBool = TRUE;

	OnDeleteImageUC(warpedImg);
	OnDeleteImageUC(warpedImg2);

	for (i = 0; i < NUM_FRAMES; i++) {
		OnDeleteImageUC(m_MorphedImg[i]);
	}
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
					m_OutputImage[i] = (unsigned char)(a * m_InputImage[i] + (1 - a) * temp[i]);
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
	double** tempOutput;
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


void CImgprocessingsDoc::OnLowPassFilter()
{
	int i, j;
	double LPF[3][3] = { {1. / 9., 1. / 9., 1. / 9.},{1. / 9., 1. / 9., 1. / 9.} ,{1. / 9., 1. / 9., 1. / 9.} };

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	m_tempImage = OnMaskProcessArr(m_InputImage, LPF);

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			if (m_tempImage[i][j] > 255.)
				m_OutputImage[i * m_Re_width + j] = 255;
			else if (m_tempImage[i][j] < 0)
				m_OutputImage[i * m_Re_width + j] = 0;
			else
				m_OutputImage[i * m_Re_width + j] = (unsigned char)m_tempImage[i][j];
		}
	}
}


void CImgprocessingsDoc::OnHighPassFilter()
{
	int i, j;
	double HPF[3][3] = { {-1. / 9., -1. / 9., -1. / 9.},{-1. / 9., 8. / 9., -1. / 9.} ,{-1. / 9., -1. / 9., -1. / 9.} };
	//double HPF[3][3] = { {-1. / 9., -1. / 9., -1. / 9.},{-1. / 9., A - 1. / 9., -1. / 9.} ,{-1. / 9., -1. / 9., -1. / 9.} };
	//double HPF[3][3] = { {1,-2,1},{-2,5,-2} ,{1,-2,1} };
	//double HPF[3][3] = { {0,-1,0},{-1,5,-1},{0,-1,0} };
	//double HPF[3][3] = { {-1,-1,-1},{-1,9,-1},{-1,-1,-1} };

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	m_tempImage = OnMaskProcessArr(m_InputImage, HPF);

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			if (m_tempImage[i][j] > 255.)
				m_OutputImage[i * m_Re_width + j] = 255;
			else if (m_tempImage[i][j] < 0)
				m_OutputImage[i * m_Re_width + j] = 0;
			else
				m_OutputImage[i * m_Re_width + j] = (unsigned char)m_tempImage[i][j];
		}
	}
}


void CImgprocessingsDoc::OnMeanFilter()
{
	int i, j, n, m;
	double** tempInputImage, S = 0.;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	tempInputImage = Image2DMem(m_height + 2, m_width + 2);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			tempInputImage[i + 1][j + 1] = (double)m_InputImage[i * m_width + j];
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			for (n = 0; n < 3; n++) {
				for (m = 0; m < 3; m++) {
					S += tempInputImage[i + n][j + m];
				}
			}
			m_OutputImage[i * m_Re_width + j] = (unsigned char)(S / 9.);
			S = 0.;
		}
	}
}


void CImgprocessingsDoc::OnMedianFilter()
{
	int i, j, n, m, h, weight = 0;
	double** tempInputImage, *Mask;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];
	Mask = new double[9 + weight];

	tempInputImage = Image2DMem(m_height + 2, m_width + 2);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			tempInputImage[i + 1][j + 1] = (double)m_InputImage[i * m_width + j];
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			for (n = 0; n < 3; n++) {
				for (m = 0; m < 3; m++) {
					Mask[n*3+m] = tempInputImage[i + n][j + m];
				}
			}
			for (h = 9; h < 9 + weight; h++) {
				Mask[h] = tempInputImage[i + 1][j + 1];
			}
			OnBubbleSort(Mask, 9+weight);
			m_OutputImage[i * m_Re_width + j] = (unsigned char)Mask[(9+weight)/2];
		}
	}
}


void CImgprocessingsDoc::OnMaxFilter()
{
	int i, j, n, m;
	double** tempInputImage, Mask[9];

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	tempInputImage = Image2DMem(m_height + 2, m_width + 2);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			tempInputImage[i + 1][j + 1] = (double)m_InputImage[i * m_width + j];
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			for (n = 0; n < 3; n++) {
				for (m = 0; m < 3; m++) {
					Mask[n * 3 + m] = tempInputImage[i + n][j + m];
				}
			}
			OnBubbleSort(Mask, 9);
			m_OutputImage[i * m_Re_width + j] = (unsigned char)Mask[8];
		}
	}
}


void CImgprocessingsDoc::OnMinFilter()
{
	int i, j, n, m;
	double** tempInputImage, Mask[9];

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	tempInputImage = Image2DMem(m_height + 2, m_width + 2);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			tempInputImage[i + 1][j + 1] = (double)m_InputImage[i * m_width + j];
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			for (n = 0; n < 3; n++) {
				for (m = 0; m < 3; m++) {
					Mask[n * 3 + m] = tempInputImage[i + n][j + m];
				}
			}
			OnBubbleSort(Mask, 9);
			m_OutputImage[i * m_Re_width + j] = (unsigned char)Mask[0];
		}
	}
}


void CImgprocessingsDoc::OnTrackClosedCurve()
{
	int i, j;
	double** edge, ** visited;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	edge = Image2DMem(m_height, m_width);
	visited = Image2DMem(m_height, m_width);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			//memset(visited[0], 0, sizeof(double) * m_height * m_width);
			if (m_InputImage[i * m_width + j] == 255 && visited[i][j] == 0) {
				visited[i][j] = 2;
				OnFollowClosedCurve(m_InputImage, visited, edge, 0, i, j);
				visited[i][j] = 1;
			}
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_OutputImage[i * m_Re_width + j] = (unsigned char)edge[i][j];
		}
	}
}


BOOL CImgprocessingsDoc::OnFollowClosedCurve(unsigned char* Target, double** visited, double** edge, BOOL sbool, int row, int col)
{
	int i, j;
	BOOL tempBool;

	if (sbool != 0) {
		visited[row][col] = 1;
	}

	if (sbool == 0)
		tempBool = 1;
	else
		tempBool = 2;

	for (i = -1; i < 2; i++) {
		if (row + i < 0 || row + i > m_height-1)
			continue;

		for (j = -1; j < 2; j++) {
			if (col + j < 0 || col + j > m_width-1)
				continue;

			if (visited[row+i][col+j] == 2 && sbool == 2) {
				edge[row][col] = 255.;
				return TRUE;
			}
			else if (Target[(row + i) * m_width + (col + j)] == 255 && visited[row + i][col + j] == 0) {
				if (OnFollowClosedCurve(Target, visited, edge, tempBool, row + i, col + j)) {
					edge[row][col] = 255.;
					return TRUE;
				}
			}
		}
	}

	return FALSE;
}


void CImgprocessingsDoc::OnFft2d()
{
	
	int i, j, row, col, Log2N, Num, C = 20;
	Complex* Data;

	unsigned char** temp;
	double Value, Absol;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];
	
	Num = m_width;
	Log2N = 0;

	while (Num >= 2) {
		Num >>= 1;
		Log2N++;
	}

	m_tempImage = Image2DMem(m_height, m_width);

	Data = new Complex[m_width];

	m_FFT = new Complex * [m_height];
	m_FFT[0] = new Complex[m_width*m_height];

	temp = new unsigned char* [m_height];
	temp[0] = new unsigned char[m_width * m_height];

	for (i = 1; i < m_height; i++) {
		m_FFT[i] = m_FFT[i - 1] + m_width;
		temp[i] = temp[i - 1] + m_width;
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			Data[j].Re = (double)m_InputImage[i * m_width + j];
			Data[j].Im = 0;
		}

		OnFft1d(Data, m_width, Log2N);

		for (j = 0; j < m_width; j++) {
			m_FFT[i][j] = Data[j];
		}
	}

	free(Data);

	Num = m_height;
	Log2N = 0;

	while (Num >= 2) {
		Num >>= 1;
		Log2N++;
	}

	Data = new Complex[m_height];

	for (i = 0; i < m_width; i++) {
		for (j = 0; j < m_height; j++) {
			Data[j] = m_FFT[j][i];
		}

		OnFft1d(Data, m_height, Log2N);

		for (j = 0; j < m_height; j++) {
			m_FFT[j][i] = Data[j];
		}
	}
	

	
	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			Value = hypot(m_FFT[i][j].Re, m_FFT[i][j].Im);
			Absol = C * log(1 + Value);

			if (Absol > 255.0)
				Absol = 255.0;
			if (Absol < 0.0)
				Absol = 0.0;

			m_tempImage[i][j] = Absol;
		}
	}
	
	for (i = 0; i < m_height; i += m_height / 2) {
		for (j = 0; j < m_width; j += m_width / 2) {
			for (row = 0; row < m_height / 2; row++) {
				for (col = 0; col < m_width / 2; col++) {
					temp[(m_height / 2 - 1) - row + i][(m_width / 2 - 1) - col + j] = 
						(unsigned char)m_tempImage[i + row][j + col];
				}
			}
		}
	}
	
	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_width + j] = temp[i][j];
		}
	}

	free(Data);
	free(temp[0]);
	free(temp);
}


void CImgprocessingsDoc::OnFft1d(Complex* X, int N, int Log2N)
{
	OnShuffle(X, N, Log2N);
	OnButterfly(X, N, 0);
}


void CImgprocessingsDoc::OnShuffle(Complex* X, int N, int Log2N)
{
	int i;
	Complex* temp;

	temp = new Complex[N];

	for (i = 0; i < N; i++) {
		temp[i] = X[OnReverseBitOrder(i, Log2N)];
	}

	for (i = 0; i < N; i++) {
		X[i] = temp[i];
	}

	delete[] temp;
}


void CImgprocessingsDoc::OnButterfly(Complex* X, int N, int mode)
{
	if (N == 1)
		return;
	int i, s = 1 - 2 * mode;
	Complex temp, I = { 0,-1 };

	OnButterfly(X, N / 2, mode);
	OnButterfly(X + N / 2, N / 2, mode);

	for (i = 0; i < N / 2; i++)
	{
		temp = OnComplexMul(OnTwiddleFactor(N, s * i), X[i + N / 2]);

		X[i + N / 2] = OnComplexSub(X[i], temp);
		
		X[i] = OnComplexAdd(X[i], temp);
	}
}


int CImgprocessingsDoc::OnReverseBitOrder(int index, int Log2N)
{
	int i, X, Y;

	Y = 0;

	for (i = 0; i < Log2N; i++) {
		X = (index & (1 << i)) >> i;
		Y = (Y << 1) | X;
	}

	return Y;
}


Complex CImgprocessingsDoc::OnComplexAdd(Complex A, Complex B)
{
	Complex res = { A.Re + B.Re, A.Im + B.Im };
	return res;
}


Complex CImgprocessingsDoc::OnComplexSub(Complex A, Complex B)
{
	Complex res = { A.Re - B.Re, A.Im - B.Im };
	return res;
}


Complex CImgprocessingsDoc::OnComplexMul(Complex A, Complex B)
{
	Complex res = { A.Re * B.Re - B.Im * A.Im, A.Re * B.Im + A.Im * B.Re };
	return res;
}


Complex CImgprocessingsDoc::OnTwiddleFactor(double N, double exp)
{
	Complex res = { cos(2 * M_PI / N * exp), -sin(2 * M_PI / N * exp) };

	return res;
}


void CImgprocessingsDoc::OnIfft2d()
{
	int i, j, Num, Log2N;
	Complex* Data;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	Num = m_width;
	Log2N = 0;
	while (Num >= 2) // 주파수 변환된 영상의 너비 계산
	{
		Num >>= 1;
		Log2N++;
	}

	Data = new Complex[m_height];
	m_IFFT = new Complex * [m_height];
	m_IFFT[0] = new Complex[m_width * m_height];

	for (i = 1; i < m_height; i++) {
		m_IFFT[i] = m_IFFT[i - 1] + m_width;
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) { // 한 행을 복사
			Data[j].Re = m_FFT[i][j].Re;
			Data[j].Im = m_FFT[i][j].Im;
		}

		OnIfft1d(Data, m_width, Log2N); // 1차원 IFFT

		for (j = 0; j < m_width; j++) {
			m_IFFT[i][j].Re = Data[j].Re; // 결과 저장
			m_IFFT[i][j].Im = Data[j].Im;
		}
	}

	free(Data);

	Num = m_height;
	Log2N = 0;
	while (Num >= 2) // 주파수 변환된 영상의 높이 계산
	{
		Num >>= 1;
		Log2N++;
	}

	Data = new Complex[m_height];

	for (i = 0; i < m_width; i++) {
		for (j = 0; j < m_height; j++) {
			Data[j].Re = m_IFFT[j][i].Re; // 한 열을 복사
			Data[j].Im = m_IFFT[j][i].Im;
		}

		OnIfft1d(Data, m_height, Log2N); // 1차원 IFFT       

		for (j = 0; j < m_height; j++) {
			m_IFFT[j][i].Re = Data[j].Re; // 결과 저장
			m_IFFT[j][i].Im = Data[j].Im;
		}
	}

	for (i = 0; i < m_width; i++) {
		for (j = 0; j < m_height; j++) {
			if (m_IFFT[i][j].Re > 255.)
				m_OutputImage[i * m_width + j] = 255;
			else if (m_IFFT[i][j].Re < 0)
				m_OutputImage[i * m_width + j] = 0;
			else
				m_OutputImage[i * m_width + j] = (unsigned char)lround(m_IFFT[i][j].Re); // 결과 출력
		}
	}

	free(Data);
}


void CImgprocessingsDoc::OnIfft1d(Complex* X, int N, int Log2N)
{
	OnShuffle(X, N, Log2N);
	OnButterfly(X, N, 1);

	for (int i = 0; i < N; i++)
	{
		X[i].Re /= N;
		X[i].Im /= N;
	}
}


void CImgprocessingsDoc::OnLpfFrequency()
{
	int i, j, x, y, row, col;
	double temp, D, N, Absol, Value, C = 20;
	double** tempIm;
	D = 32.;
	N = 4.;

	m_OutputImageLB = new unsigned char[m_width * m_height];
	m_OutputImageRB = new unsigned char[m_width * m_height];

	tempIm = Image2DMem(m_height, m_width);
	m_tempImage = Image2DMem(m_height, m_width);

	OnFft2d();

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_OutputImageLB[i * m_width + j] = m_OutputImage[i * m_width + j];
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			x = i;
			y = j;
			if (x > m_height / 2)
				x -= m_height;
			if (y > m_width / 2)
				y -= m_width;

			temp = 1. / (1. + pow(hypot(x, y) / D, 2 * N));

			m_FFT[i][j].Re *= temp;
			m_FFT[i][j].Im *= temp;
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			Value = hypot(m_FFT[i][j].Re, m_FFT[i][j].Im);
			Absol = C * log(1 + Value);

			if (Absol > 255.0)
				Absol = 255.0;
			if (Absol < 0.0)
				Absol = 0.0;

			m_tempImage[i][j] = Absol;
		}
	}

	for (i = 0; i < m_height; i += m_height / 2) {
		for (j = 0; j < m_width; j += m_width / 2) {
			for (row = 0; row < m_height / 2; row++) {
				for (col = 0; col < m_width / 2; col++) {
					tempIm[(m_height / 2 - 1) - row + i][(m_width / 2 - 1) - col + j] =
						(unsigned char)m_tempImage[i + row][j + col];
				}
			}
		}
	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImageRB[i * m_width + j] = (unsigned char)tempIm[i][j];
		}
	}

	OnIfft2d();

	free(tempIm[0]);
	free(tempIm);

	m_printImageBool = TRUE;
}


void CImgprocessingsDoc::OnHpfFrequency()
{
	int i, j, x, y, row, col;
	double temp, D, N, Absol, Value, C = 20;
	double** tempIm;
	D = 128.;
	N = 4.;

	m_OutputImageLB = new unsigned char[m_width * m_height];
	m_OutputImageRB = new unsigned char[m_width * m_height];

	tempIm = Image2DMem(m_height, m_width);
	m_tempImage = Image2DMem(m_height, m_width);

	OnFft2d();

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_OutputImageLB[i * m_width + j] = m_OutputImage[i * m_width + j];
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			x = i;
			y = j;
			if (x > m_height / 2)
				x -= m_height;
			if (y > m_width / 2)
				y -= m_width;

			temp = 1. / (1. + pow(D / (1+hypot(x, y)), 2 * N));

			m_FFT[i][j].Re *= temp;
			m_FFT[i][j].Im *= temp;
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			Value = hypot(m_FFT[i][j].Re, m_FFT[i][j].Im);
			Absol = C * log(1 + Value);

			if (Absol > 255.0)
				Absol = 255.0;
			if (Absol < 0.0)
				Absol = 0.0;

			m_tempImage[i][j] = Absol;
		}
	}

	for (i = 0; i < m_height; i += m_height / 2) {
		for (j = 0; j < m_width; j += m_width / 2) {
			for (row = 0; row < m_height / 2; row++) {
				for (col = 0; col < m_width / 2; col++) {
					tempIm[(m_height / 2 - 1) - row + i][(m_width / 2 - 1) - col + j] =
						(unsigned char)m_tempImage[i + row][j + col];
				}
			}
		}
	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImageRB[i * m_width + j] = (unsigned char)tempIm[i][j];
		}
	}

	OnIfft2d();

	free(tempIm[0]);
	free(tempIm);

	m_printImageBool = TRUE;
}


void CImgprocessingsDoc::OnBilateral()
{
	CConstantDlg dlg;

	int i, j, MaskLen, x, y, row, col;
	double stdR, similarity;
	double** GaussianMask, ** tempInputImage, ** tempOutputImage, S = 0., weight = 0.;
	

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	tempInputImage = Image2DMem(m_height, m_width);
	tempOutputImage = Image2DMem(m_height, m_width);

	if (dlg.DoModal() == IDOK)
	{
		stdR = 30;// 2 * dlg.m_Constant;
		//MaskLen = (int)(dlg.m_Constant * 8);
		MaskLen = (int)(dlg.m_Constant * 6);

		if (MaskLen % 2 == 0)
			MaskLen += 1;

		if (MaskLen < 3)
			MaskLen = 3;

		GaussianMask = Image2DMem(MaskLen, MaskLen);

		for (i = 0; i <= MaskLen / 2; i++) {
			x = MaskLen / 2 - i;
			for (j = 0; j <= MaskLen / 2; j++) {
				y = -(MaskLen / 2) + j;
				GaussianMask[i][j] = exp(-((double)(x * x + y * y) / (double)(2 * dlg.m_Constant * dlg.m_Constant))) / (2 * M_PI * dlg.m_Constant * dlg.m_Constant);
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

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			tempInputImage[i][j] = (double)m_InputImage[i * m_width + j];
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			for (x = 0; x < MaskLen; x++) {
				row = (i + x - MaskLen / 2);
				if (row < 0 || row > m_height-1)
					continue;

				for (y = 0; y < MaskLen; y++) {
					col = (j + y - MaskLen / 2);
					if (col < 0 || col > m_width-1)
						continue;

					similarity = exp(-(pow(tempInputImage[i][j]-tempInputImage[row][col], 2) / (2 * stdR * stdR)));
					S += GaussianMask[x][y] * tempInputImage[row][col] * similarity;
					weight += similarity * GaussianMask[x][y];
				}
			}
			tempOutputImage[i][j] = S/weight;
			S = 0.;
			weight = 0.;
		}
	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			if (tempOutputImage[i][j] > 255.)
				tempOutputImage[i][j] = 255.;
			else if (tempOutputImage[i][j] < 0.)
				tempOutputImage[i][j] = 0.;
		}
	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_Re_width + j] = (unsigned char)(tempOutputImage[i][j]+0.5);
		}
	}

	free(tempInputImage[0]);
	free(tempInputImage);
	free(tempOutputImage[0]);
	free(tempOutputImage);
}


void CImgprocessingsDoc::OnMeanFilterSat(int mode, double** Src, double** Dst, int height, int width, int radius)
{
	int i, j, UL, UR, BL, BR, divx, divy, r_2Mr = radius/2 - radius;
	double** SAT, S = 0.;

	if (mode) {
		SAT = Image2DMem(height, width);

		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				S += Src[i][j];
				SAT[i][j] = S;
			}
			S = 0.;
		}

		for (i = 0; i < width; i++) {
			for (j = 0; j < height; j++) {
				S += SAT[j][i];
				SAT[j][i] = S;
			}
			S = 0.;
		}

		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				UR = i + r_2Mr < 0 ? 0 : (int)SAT[i + r_2Mr][j + radius / 2 >= width ? width - 1 : j + radius / 2];

				BL = j + r_2Mr < 0 ? 0 : (int)SAT[i + radius / 2 >= height ? height - 1 : i + radius / 2][j + r_2Mr];

				UL = j + r_2Mr < 0 || i + r_2Mr < 0 ? 0 : (int)SAT[i + r_2Mr][j + r_2Mr];

				BR = (int)SAT[i + radius / 2 >= height ? height - 1 : i + radius / 2][j + radius / 2 >= width ? width - 1 : j + radius / 2];

				divx = (j + radius / 2 >= width ? width - 1 : j + radius / 2) - (j - radius / 2 < 0 ? 0 : j - radius / 2) + radius % 2;

				divy = (i + radius / 2 >= height ? height - 1 : i + radius / 2) - (i - radius / 2 < 0 ? 0 : i - radius / 2) + radius % 2;

				Dst[i][j] = (double)(BR - UR - BL + UL) / (double)(divx * divy);
			}
		}
	}
	else
	{
		m_Re_height = m_height;
		m_Re_width = m_width;
		m_Re_size = m_Re_height * m_Re_width;

		m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);
		SAT = Image2DMem(m_height, m_width);

		for (i = 0; i < m_height; i++) {
			for (j = 0; j < m_width; j++) {
				S += m_InputImage[i * m_width + j];
				SAT[i][j] = S;
			}
			S = 0.;
		}

		for (i = 0; i < m_width; i++) {
			for (j = 0; j < m_height; j++) {
				S += SAT[j][i];
				SAT[j][i] = S;
			}
			S = 0.;
		}

		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				UR = i + r_2Mr < 0 ? 0 : (int)SAT[i + r_2Mr][j + radius / 2 >= m_width ? m_width - 1 : j + radius / 2];

				BL = j + r_2Mr < 0 ? 0 : (int)SAT[i + radius / 2 >= m_height ? m_height - 1 : i + radius / 2][j + r_2Mr];

				UL = j + r_2Mr < 0 || i + r_2Mr < 0 ? 0 : (int)SAT[i + r_2Mr][j + r_2Mr];

				BR = (int)SAT[i + radius / 2 >= m_height ? m_height - 1 : i + radius / 2][j + radius / 2 >= m_width ? m_width - 1 : j + radius / 2];

				divx = (j + radius / 2 >= m_width ? m_width - 1 : j + radius / 2) - (j - radius / 2 < 0 ? 0 : j - radius / 2) + radius % 2;

				divy = (i + radius / 2 >= m_height ? m_height - 1 : i + radius / 2) - (i - radius / 2 < 0 ? 0 : i - radius / 2) + radius % 2;

				m_OutputImage[i * m_Re_width + j] = (unsigned char)((double)(BR - UR - BL + UL) / (double)(divx * divy) + 0.5);
			}
		}
	}

	free(SAT[0]);
	free(SAT);
}


void CImgprocessingsDoc::OnGuidedFilter()
{
	CFile File;
	CFileDialog OpenDlg(TRUE);

	int i, j, radius = 5;
	double epsilon = 0.01;
	double** meanI, ** meanP, ** tempA, ** tempB;
	unsigned char* temp;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = (unsigned char*)malloc(sizeof(unsigned char) * m_Re_size);

	meanI = Image2DMem(m_height, m_width);
	meanP = Image2DMem(m_height, m_width);
	tempA = Image2DMem(m_height, m_width);
	tempB = Image2DMem(m_height, m_width);

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

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			meanI[i][j] = temp[i * m_width + j];
			meanP[i][j] = m_InputImage[i * m_width + j];
			tempA[i][j] = temp[i * m_width + j] * temp[i * m_width + j];
			tempB[i][j] = temp[i * m_width + j] * m_InputImage[i * m_width + j];
		}
	}

	OnMeanFilterSat(1, meanI, meanI, m_height, m_width, radius);
	OnMeanFilterSat(1, meanP, meanP, m_height, m_width, radius);
	OnMeanFilterSat(1, tempA, tempA, m_height, m_width, radius);
	OnMeanFilterSat(1, tempB, tempB, m_height, m_width, radius);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			tempA[i][j] = (tempB[i][j] - meanI[i][j] * meanP[i][j]) / (tempA[i][j] - meanI[i][j] * meanI[i][j] + epsilon);
			tempB[i][j] = meanP[i][j] - tempA[i][j] * meanI[i][j];
		}
	}

	free(meanI[0]);
	free(meanI);
	free(meanP[0]);
	free(meanP);

	OnMeanFilterSat(1, tempA, tempA, m_height, m_width, radius);
	OnMeanFilterSat(1, tempB, tempB, m_height, m_width, radius);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			if (tempA[i][j] * (double)temp[i * m_width + j] + tempB[i][j] > 255.)
				m_OutputImage[i * m_Re_width + j] = 255;
			else if(tempA[i][j] * (double)temp[i * m_width + j] + tempB[i][j] < 0.)
				m_OutputImage[i * m_Re_width + j] = 0;
			else
				m_OutputImage[i * m_Re_width + j] = (unsigned char)(tempA[i][j] * (double)temp[i * m_width + j] + tempB[i][j] + 0.5);
		}
	}

	free(tempA[0]);
	free(tempA);
	free(tempB[0]);
	free(tempB);
	free(temp);
}


void CImgprocessingsDoc::OnDct()
{
	
	int i, j, Log2N, Num, C = 5;
	Complex* Data;

	double Value, Absol, Scale;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	Num = m_width;
	Log2N = 0;

	while (Num >= 2) {
		Num >>= 1;
		Log2N++;
	}

	m_tempImage = Image2DMem(m_height, m_width);

	Data = new Complex[m_width];

	m_DCT = new double * [m_height];
	m_DCT[0] = new double[m_width * m_height];

	for (i = 1; i < m_height; i++) {
		m_DCT[i] = m_DCT[i - 1] + m_width;
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width/2; j++) {
			Data[j].Re = (double)m_InputImage[i * m_width + 2*j];
			Data[j].Im = 0;
			Data[m_width-1-j].Re = (double)m_InputImage[i * m_width + 2 * j + 1];
			Data[m_width - 1 - j].Im = 0;
		}

		OnFft1d(Data, m_width, Log2N);

		for (j = 0; j < m_width; j++) {
			if (j == 0)
				Scale = sqrt(1. / (double)m_width);
			else
				Scale = sqrt(2. / (double)m_width);
			m_DCT[i][j] = Scale * OnComplexMul(Data[j], OnTwiddleFactor(2*m_width, j / 2.)).Re;
		}
	}

	free(Data);

	Num = m_height;
	Log2N = 0;

	while (Num >= 2) {
		Num >>= 1;
		Log2N++;
	}

	Data = new Complex[m_height];

	for (i = 0; i < m_width; i++) {
		for (j = 0; j < m_height/2; j++) {
			Data[j].Re = m_DCT[2*j][i];
			Data[j].Im = 0;
			Data[m_height-1-j].Re = m_DCT[2*j+1][i];
			Data[m_height-1-j].Im = 0;
		}

		OnFft1d(Data, m_height, Log2N);

		for (j = 0; j < m_height; j++) {
			if (j == 0)
				Scale = sqrt(1. / (double)m_width);
			else
				Scale = sqrt(2. / (double)m_width);
			m_DCT[j][i] = Scale * OnComplexMul(Data[j], OnTwiddleFactor(2 * m_width, j / 2.)).Re;
		}
	}



	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			Value = m_DCT[i][j];
			//Absol = C * log(1 + Value);
			Absol = Value;

			if (Absol > 255.0)
				Absol = 255.0;
			if (Absol < 0.0)
				Absol = 0.0;

			m_tempImage[i][j] = Absol;
		}
	}

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_width + j] = (unsigned char)m_tempImage[i][j];
		}
	}

	free(Data);
	

	
}


void CImgprocessingsDoc::OnIdct()
{
	
	int i, j, Num, Log2N;
	double Scale;
	Complex* Data;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	Data = new Complex[m_width];
	m_IDCT = new double* [m_height];
	m_IDCT[0] = new double[m_width * m_height];

	for (i = 1; i < m_height; i++) {
		m_IDCT[i] = m_IDCT[i - 1] + m_width;
	}


	Num = m_width;
	Log2N = 0;
	while (Num >= 2) 
	{
		Num >>= 1;
		Log2N++;
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) { 
			if (j == 0)
				Scale = sqrt(1. / (double)m_width);
			else
				Scale = sqrt(2. / (double)m_width);
			Data[j].Re = Scale * m_DCT[i][j];
			Data[j].Im = 0;
			Data[j] = OnComplexMul(Data[j], OnTwiddleFactor(2. * m_width, j / 2.));
		}

		OnFft1d(Data, m_width, Log2N); 

		for (j = 0; j < m_width/2; j++) {
			m_IDCT[i][2 * j] = Data[j].Re; 
			m_IDCT[i][2 * j + 1] = Data[m_width-1-j].Re; 
		}
	}

	free(Data);

	Num = m_height;
	Log2N = 0;
	while (Num >= 2) 
	{
		Num >>= 1;
		Log2N++;
	}

	Data = new Complex[m_height];

	for (i = 0; i < m_width; i++) {
		for (j = 0; j < m_height; j++) {
			if (j == 0)
				Scale = sqrt(1. / (double)m_width);
			else
				Scale = sqrt(2. / (double)m_width);
			Data[j].Re = Scale * m_IDCT[j][i];
			Data[j].Im = 0;
			Data[j] = OnComplexMul(Data[j], OnTwiddleFactor(2 * m_height, j / 2.));
		}

		OnFft1d(Data, m_height, Log2N);        

		for (j = 0; j < m_height/2; j++) {
			m_IDCT[2 * j][i] = Data[j].Re;
			m_IDCT[2 * j + 1][i] = Data[m_height - 1 - j].Re;
		}
	}

	for (i = 0; i < m_width; i++) {
		for (j = 0; j < m_height; j++) {
			if (m_IDCT[i][j] > 255.)
				m_OutputImage[i * m_width + j] = 255;
			else if (m_IDCT[i][j] < 0)
				m_OutputImage[i * m_width + j] = 0;
			else
				m_OutputImage[i * m_width + j] = (unsigned char)(m_IDCT[i][j]+0.5); 
		}
	}

	free(Data);
}


void CImgprocessingsDoc::OnWavletTransform()
{
	if (pDlg->GetSafeHwnd() == NULL)
		pDlg->Create(IDD_DIALOG7);

	pDlg->ShowWindow(SW_SHOW);
}


void CImgprocessingsDoc::OnWaveletEncode()
{	
	if (m_Level <= 0 || (pow(2, m_Level + 3) > (double)m_width) || (pow(2, m_Level + 3) > (double)m_height)) {
		AfxMessageBox(L"Not Support decomposition level");
		return;
	}

	int i, j, k, width, height;
	double* m_Conv1, * m_Conv2, * m_Conv3, * m_Conv4;
	double* m_Down1, * m_Down2, * m_Down3, * m_Down4;
	double* m_Hor, * m_Ver1, * m_Ver2;
	double** m_L, ** m_H, ** m_LL, ** m_LH, ** m_HL, ** m_HH, ** m_SLL, ** m_SLH, ** m_SHL, ** m_SHH;

	m_tempInput = Image2DMem(m_height, m_width);
	m_tempOutput = Image2DMem(m_height, m_width);
	m_ArrangeImage = OnMem2DAllocUnsigned(m_height, m_width);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_tempInput[i][j] = (double)m_InputImage[i * m_width + j];
		}
	}

	OnFilterTapGen();

	m_FilterH0 = new double[m_FilterTap];
	m_FilterH1 = new double[m_FilterTap];
	m_FilterG0 = new double[m_FilterTap];
	m_FilterG1 = new double[m_FilterTap];

	OnFilterGen(m_FilterH0, m_FilterH1, m_FilterG0, m_FilterG1);

	width = m_width;
	height = m_height;

	for (k = 0; k < m_Level; k++) {
		m_L = Image2DMem(height, width/2);
		m_H = Image2DMem(height, width/2);
		m_LL = Image2DMem(height/2, width/2);
		m_LH = Image2DMem(height/2, width/2);
		m_HL = Image2DMem(height/2, width/2);
		m_HH = Image2DMem(height/2, width/2);

		m_Hor = new double[width];

		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				m_Hor[j] = m_tempInput[i][j];
			}

			m_Conv1 = OnConvolution(m_Hor, m_FilterH0, width, 1);
			m_Conv2 = OnConvolution(m_Hor, m_FilterH1, width, 1);

			m_Down1 = OnWaveletDownSampling(m_Conv1, width);
			m_Down2 = OnWaveletDownSampling(m_Conv2, width);

			for (j = 0; j < width / 2; j++) {
				m_L[i][j] = m_Down1[j];
				m_H[i][j] = m_Down2[j];
			}
		}

		m_Ver1 = new double[height];
		m_Ver2 = new double[height];

		for (i = 0; i < width / 2; i++) {
			for (j = 0; j < height; j++) {
				m_Ver1[j] = m_L[j][i];
				m_Ver2[j] = m_H[j][i];
			}

			m_Conv1 = OnConvolution(m_Ver1, m_FilterH0, height, 1);
			m_Conv2 = OnConvolution(m_Ver1, m_FilterH1, height, 1);
			m_Conv3 = OnConvolution(m_Ver2, m_FilterH0, height, 1);
			m_Conv4 = OnConvolution(m_Ver2, m_FilterH1, height, 1);

			m_Down1 = OnWaveletDownSampling(m_Conv1, height);
			m_Down2 = OnWaveletDownSampling(m_Conv2, height);
			m_Down3 = OnWaveletDownSampling(m_Conv3, height);
			m_Down4 = OnWaveletDownSampling(m_Conv4, height);

			for (j = 0; j < height / 2; j++) {
				m_LL[j][i] = m_Down1[j];
				m_LH[j][i] = m_Down2[j];
				m_HL[j][i] = m_Down3[j];
				m_HH[j][i] = m_Down4[j];
			}
		}

		m_SLL = OnWaveletScale(m_LL, height / 2, width / 2);
		m_SLH = OnWaveletScale(m_LH, height / 2, width / 2);
		m_SHL = OnWaveletScale(m_HL, height / 2, width / 2);
		m_SHH = OnWaveletScale(m_HH, height / 2, width / 2);

		for (i = 0; i < height / 2; i++) {
			for (j = 0; j < width / 2; j++) {
				m_tempOutput[i][j] = m_LL[i][j];
				m_tempOutput[i][j+(width/2)] = m_HL[i][j];
				m_tempOutput[i+(height/2)][j] = m_LH[i][j];
				m_tempOutput[i+(height/2)][j+(width/2)] = m_HH[i][j];

				m_ArrangeImage[i][j] = (unsigned char)m_SLL[i][j];
				m_ArrangeImage[i][j + (width / 2)] = (unsigned char)m_SHL[i][j];
				m_ArrangeImage[i + (height / 2)][j] = (unsigned char)m_SLH[i][j];
				m_ArrangeImage[i + (height / 2)][j + (width / 2)] = (unsigned char)m_SHH[i][j];
			}
		}

		width /= 2;
		height /= 2;

		free(m_tempInput[0]);
		free(m_tempInput);

		m_tempInput = Image2DMem(height, width);

		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				m_tempInput[i][j] = m_LL[i][j];
			}
		}
		
		#pragma region free
		free(m_Hor);
		free(m_Ver1);
		free(m_Ver2);
		free(m_Conv1);
		free(m_Conv2);
		free(m_Conv3);
		free(m_Conv4);
		free(m_Down1);
		free(m_Down2);
		free(m_Down3);
		free(m_Down4);
		free(m_L[0]);
		free(m_H[0]);
		free(m_LL[0]);
		free(m_LH[0]);
		free(m_HL[0]);
		free(m_HH[0]);
		free(m_SLL[0]);
		free(m_SLH[0]);
		free(m_SHL[0]);
		free(m_SHH[0]);
		free(m_L);
		free(m_H);
		free(m_LL);
		free(m_LH);
		free(m_HL);
		free(m_HH);
		free(m_SLL);
		free(m_SLH);
		free(m_SHL);
		free(m_SHH);
		#pragma endregion
		
	}

	UpdateAllViews(NULL);	
}


void CImgprocessingsDoc::OnFilterTapGen()
{
	switch (pDlg->m_FilterCheck)
	{
	case 0: m_FilterTap = 2;
		break;
	case 1: m_FilterTap = 4;
		break;
	case 2: m_FilterTap = 6;
		break;
	case 3: m_FilterTap = 8;
		break;
	default: AfxMessageBox(L"Wrong Filter Tap");
		break;
	}
}


void CImgprocessingsDoc::OnFilterGen(double* m_H0, double* m_H1, double* m_G0, double* m_G1)
{
	int i;
	switch (m_FilterTap)
	{
	case 2:
		m_H0[0] = 0.70710678118655;
		m_H0[1] = 0.70710678118655;
		break;
	case 4:
		m_H0[0] = -0.12940952255092;
		m_H0[1] = 0.22414386804186;
		m_H0[2] = 0.83651630373747;
		m_H0[3] = 0.48296291314469;
		break;
	case 6:
		m_H0[0] = 0.03522629188210;
		m_H0[1] = -0.08544127388224;
		m_H0[2] = -0.13501102001039;
		m_H0[3] = 0.45987750211933;
		m_H0[4] = 0.80689150931334;
		m_H0[5] = 0.33267055295096;
		break;
	case 8:
		m_H0[0] = -0.01059740178500;
		m_H0[1] = 0.03288301166698;
		m_H0[2] = 0.03084138183599;
		m_H0[3] = -0.18703481171888;
		m_H0[4] = -0.02798376941698;
		m_H0[5] = 0.63088076792959;
		m_H0[6] = 0.71484657055254;
		m_H0[7] = 0.23037781330886;
		break;
	default:
		AfxMessageBox(L"Wrong Filter");
		return;
	}

	// H0 필터 계수를 이용해, H1, G0, G1 필터 계수 생성
	for (i = 0; i < m_FilterTap; i++)
		m_H1[i] = pow(-1, i + 1) * m_H0[m_FilterTap - i - 1];

	for (i = 0; i < m_FilterTap; i++)
		m_G0[i] = m_H0[m_FilterTap - i - 1];

	for (i = 0; i < m_FilterTap; i++)
		m_G1[i] = pow(-1, i) * m_H0[i];
}


double* CImgprocessingsDoc::OnWaveletDownSampling(double* m_Target, int size)
{
	int i;
	double* m_temp;

	m_temp = new double[size / 2];

	for (i = 0; i < size / 2; i++) {
		m_temp[i] = m_Target[2 * i];
	}

	return m_temp;
}


double* CImgprocessingsDoc::OnConvolution(double* m_Target, double* m_Filter, int size, int mode)
{
	int i, j;
	double* m_temp, * m_tempConv;
	double m_sum = 0.;

	m_temp = new double[size + m_FilterTap - 1];
	m_tempConv = new double[size];

	switch (mode) {
	case 1: 
		for (i = 0; i < size; i++) {
			m_temp[i] = m_Target[i];
		}

		for (i = 0; i < m_FilterTap - 1; i++) {
			m_temp[size + i] = m_Target[i];
		}

		break;
	case 2:
		for (i = 0; i < m_FilterTap - 1; i++) {
			m_temp[i] = m_Target[size - m_FilterTap + 1 + i];
		}

		for (i = m_FilterTap - 1; i < size + m_FilterTap - 1; i++) {
			m_temp[i] = m_Target[i - m_FilterTap + 1];
		}

		break; 
	}
	
	for (i = 0; i < size; i++) {
		for (j = 0; j < m_FilterTap; j++) {
			m_sum += (m_temp[j + i] * m_Filter[m_FilterTap - 1 - j]);
		}

		m_tempConv[i] = m_sum;
		m_sum = 0.;
	}

	free(m_temp);

	return m_tempConv;
}


unsigned char** CImgprocessingsDoc::OnMem2DAllocUnsigned(int height, int width)
{
	int i;
	unsigned char** temp;

	temp = (unsigned char**)malloc(sizeof(unsigned char*) * height);
	temp[0] = (unsigned char*)calloc((size_t)height * width, sizeof(unsigned char));

	for (i = 1; i < height; i++) {
		temp[i] = temp[i - 1] + width;
	}

	return temp;
}


double** CImgprocessingsDoc::OnWaveletScale(double** m_Target, int height, int width)
{
	int i, j;
	double min, max;
	double** temp;

	temp = Image2DMem(height, width);

	min = max = m_Target[0][0];

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (m_Target[i][j] < min) {
				min = m_Target[i][j];
			}
			else if (m_Target[i][j] > max) {
				max = m_Target[i][j];
			}
		}
	}

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			temp[i][j] = (m_Target[i][j] - min) * (255. / (max - min));
		}
	}

	return temp;
}


void CImgprocessingsDoc::OnWaveletDecode()
{
	int i, j, k;
	int width, height;
	double* tempLL, * tempLH, * tempHL, * tempHH, * tempL, * tempH;
	double** L, ** H;
	double* Up1, * Up2, * Up3, * Up4;
	double* Conv1, * Conv2, * Conv3, * Conv4;
	double** R;

	width = m_width / (int)(pow(2, m_Level));
	height = m_height / (int)(pow(2, m_Level));

	m_Recon = new double[m_width * m_height];

	for (k = m_Level; k > 0; k--) {
		if (width > m_width || height > m_height) { // 분해 종료
			return;
		}

		tempLL = new double[height];
		tempLH = new double[height];
		tempHL = new double[height];
		tempHH = new double[height];

		L = Image2DMem(height * 2, width);
		H = Image2DMem(height * 2, width);

		tempL = new double[width];
		tempH = new double[width];

		R = Image2DMem(height * 2, width * 2);

		for (i = 0; i < width; i++) {
			for (j = 0; j < height; j++) { // 정렬영상에서 처리하고자 하는 열을 분리
				tempLL[j] = m_tempOutput[j][i];
				tempLH[j] = m_tempOutput[j + height][i];
				tempHL[j] = m_tempOutput[j][i + width];
				tempHH[j] = m_tempOutput[j + height][i + width];
			}
			
			Up1 = OnWaveletUpSampling(tempLL, height); // 업 샘플링
			Up2 = OnWaveletUpSampling(tempLH, height);
			Up3 = OnWaveletUpSampling(tempHL, height);
			Up4 = OnWaveletUpSampling(tempHH, height);

			Conv1 = OnConvolution(Up1, m_FilterG0, height * 2, 2); // Convolution 연산
			Conv2 = OnConvolution(Up2, m_FilterG1, height * 2, 2);
			Conv3 = OnConvolution(Up3, m_FilterG0, height * 2, 2);
			Conv4 = OnConvolution(Up4, m_FilterG1, height * 2, 2);

			for (j = 0; j < height * 2; j++) {
				L[j][i] = Conv1[j] + Conv2[j];
				H[j][i] = Conv3[j] + Conv4[j];
			}
		}

		for (i = 0; i < height * 2; i++) {
			for (j = 0; j < width; j++) {
				tempL[j] = L[i][j]; // 횡 데이터 분리
				tempH[j] = H[i][j];
			}


			Up1 = OnWaveletUpSampling(tempL, width); // 업 샘플링
			Up2 = OnWaveletUpSampling(tempH, width);

			Conv1 = OnConvolution(Up1, m_FilterG0, width * 2, 2); //Convolution 연산
			Conv2 = OnConvolution(Up2, m_FilterG1, width * 2, 2);

			for (j = 0; j < width * 2; j++) {
				R[i][j] = Conv1[j] + Conv2[j];
			}
		}

		for (i = 0; i < height * 2; i++) {
			for (j = 0; j < width * 2; j++) {
				m_tempOutput[i][j] = R[i][j]; // 복원 데이터를 다시 정렬
			}
		}
		height = height * 2; // 영상의 크기를 두배 확장
		width = width * 2;

	}

	m_Re_width = m_width;
	m_Re_height = m_height;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_Recon[i * m_width + j] = R[i][j];
			m_OutputImage[i * m_width + j] = (unsigned char)R[i][j]; // 최종 복원된 결과를 출력
		}
	}

	UpdateAllViews(NULL);

	// 메모리 해제
	delete[] tempLL, tempLH, tempHL, tempHH, tempL, tempH;
	delete[] Up1, Up2, Up3, Up4;
	delete[] Conv1, Conv2, Conv3, Conv4;

	free(L[0]); free(H[0]); free(R[0]);
	free(L); free(H); free(R);
}


double* CImgprocessingsDoc::OnWaveletUpSampling(double* m_Target, int size)
{
	// 업 샘플링을 위한 함수
	int i;
	double* m_temp;

	m_temp = new double[size * 2];

	for (i = 0; i < size * 2; i++)
		m_temp[i] = 0.0; //초기화

	for (i = 0; i < size; i++)
		m_temp[2 * i] = m_Target[i]; // 업샘플링 처리

	return m_temp;
}


void CImgprocessingsDoc::OnSNR()
{
	double OrgSum, ErrSum, MeanErr, MeanOrg;
	int i;

	OrgSum = 0.0;
	ErrSum = 0.0;

	// calculate mean squared error
	for (i = 0; i < m_size; i++) {
		// 에러의 에너지 계산
		ErrSum += ((double)m_InputImage[i] - m_Recon[i]) * ((double)m_InputImage[i] - m_Recon[i]);
	}
	MeanErr = ErrSum / m_size; // 에러 에너지 평균


	for (i = 0; i < m_size; i++) {
		// 신호의 에너지 계산
		OrgSum += ((double)m_InputImage[i]) * ((double)m_InputImage[i]);
	}
	MeanOrg = OrgSum / m_size; // 신호 에너지 평균

	pDlg->m_Error = (float)MeanErr; // 에러 출력
	pDlg->m_SNR = (float)(10 * (double)log10(MeanOrg / MeanErr)); // 신호대 잡음비 계산
}


void CImgprocessingsDoc::OnSheer()
{
	int i, j;
	double xs = -1, ys = 0;

	m_Re_height = (int)(m_height + fabs(ys) * (m_height-1) + 0.5);
	m_Re_width = (int)(m_width + fabs(xs) * (m_width - 1) + 0.5);
	m_Re_size = m_Re_width * m_Re_height;

	m_OutputImage = (unsigned char*)calloc(m_Re_size, sizeof(unsigned char));
	m_OutputImageLB = (unsigned char*)calloc(m_Re_size, sizeof(unsigned char));
	m_OutputImageRB = (unsigned char*)calloc(m_Re_size, sizeof(unsigned char));


	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_Re_width + j] = 0;
			m_OutputImageLB[i * m_Re_width + j] = 0;
		}
	}
	
	if (xs * ys == 1) {
		for (i = 0; i < m_height; i++) {
			for (j = 0; j < m_width; j++) {
				m_OutputImage[((-ys < 0 ? i + m_Re_height - m_height : i) + (int)(j * -ys)) * m_Re_width + ((xs < 0 ? j + m_Re_width - m_width : j) + (int)((m_height - 1 - i) * xs))] = m_InputImage[i * m_width + j];
			}
		}
	}
	else {
		double div = 1 - xs * ys;
		int x, y;
		int xcps = xs < 0 ? m_Re_width - m_width : 0;
		int ycps = ys < 0 ? m_Re_height - m_height : 0;
		unsigned char value;
		for (i = 0; i < m_Re_height; i++) {
			for (j = 0; j < m_Re_width; j++) {
				y = (int)((-ys * j + (m_Re_height - 1 - i) + ys * xcps - ycps) / div);
				x = (int)((-xs * (m_Re_height - 1 - i) + j + xs * ycps - xcps) / div);
				if (y > m_height - 1 || x > m_width - 1 || y < 0 || x < 0) {
					value = 0;
				}
				else
					value = m_InputImage[(m_height-1-y) * m_width + x];
				
				m_OutputImage[i * m_Re_width + j] = value;
			}
		}

		for (i = 0; i < m_height; i++) {
			for (j = 0; j < m_width; j++) {
				m_OutputImageLB[((-ys < 0 ? i + m_Re_height - m_height : i) + (int)(j * -ys)) * m_Re_width + ((xs < 0 ? j + m_Re_width - m_width : j) + (int)((m_height - 1 - i) * xs))] = m_InputImage[i * m_width + j];
			}
		}

		m_printImageBool = TRUE;
	}
}


void CImgprocessingsDoc::OnBinaryClose()
{
	CConstantDlg dlg;

	int num;
	unsigned char* tempInput;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_width * m_Re_height;

	m_OutputImage = new unsigned char[m_Re_size];
	tempInput = new unsigned char[m_size];

	memcpy(tempInput, m_InputImage, (size_t)m_size);

	if (dlg.DoModal() == IDOK) {
		num = (int)dlg.m_Constant;
		if (num < 1) {
			AfxMessageBox(L"num must be larger than 1");
			return;
		}
	}

	OnBinaryDilation(num, FALSE, FALSE, NULL);

	memcpy(m_InputImage, m_OutputImage, (size_t)m_size);

	OnBinaryErosion(num, FALSE, FALSE, NULL);

	memcpy(m_InputImage, tempInput, (size_t)m_size);

	free(tempInput);
}


void CImgprocessingsDoc::OnBinaryOpen()
{
	CConstantDlg dlg;

	int num;
	unsigned char* tempInput;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_width * m_Re_height;

	m_OutputImage = new unsigned char[m_Re_size];
	tempInput = new unsigned char[m_size];

	memcpy(tempInput, m_InputImage, (size_t)m_size);

	if (dlg.DoModal() == IDOK) {
		num = (int)dlg.m_Constant;
		if (num < 1) {
			AfxMessageBox(L"num must be larger than 1");
			return;
		}
	}

	OnBinaryErosion(num, FALSE, FALSE, NULL);

	memcpy(m_InputImage, m_OutputImage, (size_t)m_size);

	OnBinaryDilation(num, FALSE, FALSE, NULL);

	memcpy(m_InputImage, tempInput, (size_t)m_size);

	free(tempInput);
}


double** CImgprocessingsDoc::OnWarpAffine(double** image, double **matrix, int height, int width, int re_height, int re_width, int orgX, int orgY)
{
	int i, j, x, y;
	double** tempImage, Value;
	double xOriginPos = orgX;
	double yOriginPos = re_height - 1 - orgY;

	tempImage = Image2DMem(re_height, re_width);

	for (i = 0; i < re_height; i++) {
		for (j = 0; j < re_width; j++) {
			x = (int)((j- xOriginPos) * matrix[0][0] + (yOriginPos -i) * matrix[0][1] + matrix[0][2]);
			y = (int)(height-1 - ((j- xOriginPos) * matrix[1][0] + (yOriginPos - i) * matrix[1][1] + matrix[1][2]));
			if (x < 0 || x > width - 1 || y < 0 || y > height - 1)
				if (j == 0 || j == re_width - 1 || i == 0 || i == re_height - 1)
					Value = 0;
				else
					Value = 255;
			else
				Value = image[y][x];
				
			tempImage[i][j] = Value;
		}
	}
	
	return tempImage;
}


double** CImgprocessingsDoc::OnForwardWarpAffine(double** image, double** matrix, int height, int width, int re_height, int re_width, int orgX, int orgY)
{
	int i, j, x, y;
	double** tempImage;
	double xOriginPos = orgX;
	double yOriginPos = re_height - 1 - orgY;

	tempImage = Image2DMem(re_height, re_width);

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			x = (int)(j * matrix[0][0] + (height - 1 - i) * matrix[0][1] + matrix[0][2] + xOriginPos);
			y = (int)(yOriginPos - (j * matrix[1][0] + (height - 1 - i) * matrix[1][1] + matrix[1][2]));
			if (x < 0 || x > re_width - 1 || y < 0 || y > re_height - 1)
				continue;
			else
				tempImage[y][x] = image[i][j];
		}
	}

	return tempImage;
}


void CImgprocessingsDoc::OnTranslationAffine()
{
	int i, j, dx = 50, dy = 50;
	double** matrix, **temp;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];
	m_tempImage = Image2DMem(m_height, m_width);

	matrix = Image2DMem(2, 3);
	matrix[0][0] = 1; matrix[0][2] = -dx;
	matrix[1][1] = 1; matrix[1][2] = -dy;

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_tempImage[i][j] = m_InputImage[i * m_width + j];
		}
	}

	temp = OnWarpAffine(m_tempImage, matrix, m_height, m_width, m_Re_height, m_Re_width, 0,0);

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_Re_width + j] = (unsigned char)temp[i][j];
		}
	}

	free(temp[0]);
	free(temp);
	free(m_tempImage[0]);
	free(m_tempImage);
}


void CImgprocessingsDoc::OnRotationAffine()
{
	int i, j;
	double** matrix, ** temp;
	double Radian, angle = 75, scale = 2;

	Radian = (double)angle * M_PI / 180.;
	DecideLen:
	if (0 <= Radian && Radian <= M_PI / 2) {
		m_Re_height = (int)(m_height / scale * cos(Radian) + m_width / scale *cos(M_PI / 2 - Radian));
		m_Re_width = (int)(m_height/scale * cos(M_PI / 2 - Radian) + m_width/scale * cos(Radian));
	}
	else if (Radian < 0) {
		Radian += M_PI / 2;
		goto DecideLen;
	}
	else {
		Radian -= M_PI / 2;
		goto DecideLen;
	}
	Radian = (double)angle * M_PI / 180.;
	
	//m_Re_height = m_height*2;
	//m_Re_width = m_width*2;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];
	m_tempImage = Image2DMem(m_height, m_width);

	matrix = OnGetRotationMatrix(m_width / 2, m_height / 2, m_Re_width/2, m_Re_height/2, angle, scale);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_tempImage[i][j] = m_InputImage[i * m_width + j];
		}
	}

	temp = OnWarpAffine(m_tempImage, matrix, m_height, m_width, m_Re_height, m_Re_width, 0,0);

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_Re_width + j] = (unsigned char)(temp[i][j]+0.5);
		}
	}

	free(temp[0]);
	free(temp);
	free(m_tempImage[0]);
	free(m_tempImage);
}


double** CImgprocessingsDoc::OnGetRotationMatrix(int cx, int cy, int re_cx, int re_cy, double angle, double scale)
{
	double radian = M_PI * angle / 180;
	double** matrix = Image2DMem(2, 3);
	
	matrix[0][0] = scale * cos(radian); matrix[0][1] = scale * sin(radian);
	matrix[0][2] = scale * (-re_cx * cos(radian) - re_cy * sin(radian)) + cx;

	matrix[1][0] = scale * -sin(radian); matrix[1][1] = scale * cos(radian);
	matrix[1][2] = scale * (re_cx * sin(radian) - re_cy * cos(radian)) + cy;
	
	/*
	scale = 1 / scale;
	matrix[0][0] = scale * cos(radian); matrix[0][1] = scale * -sin(radian);
	matrix[0][2] = scale * (-cx * cos(radian) + cy * sin(radian)) + re_cx;

	matrix[1][0] = scale * sin(radian); matrix[1][1] = scale * cos(radian);
	matrix[1][2] = scale * (-cx * sin(radian) - cy * cos(radian)) + re_cy;
	*/
	return matrix;
}


double** CImgprocessingsDoc::OnGetAffineTransform(double** src, double** dst)
{
	int i, check;
	double **matrix = mkMatrix(6, 7);

	for (i = 0; i < 3; i++) {
		matrix[i*2][0] = dst[i][0];
		matrix[i*2][1] = dst[i][1];
		matrix[i*2][4] = 1;
		matrix[i*2][6] = src[i][0];
		matrix[i*2 + 1][2] = dst[i][0];
		matrix[i*2 + 1][3] = dst[i][1];
		matrix[i*2 + 1][5] = 1;
		matrix[i*2 + 1][6] = src[i][1];
	}
	
	double** res = rref(matrix, &check);

	delMatrix(matrix);
	matrix = mkMatrix(2, 3);

	
	for (i = 0; i < 2; i++) {
		matrix[0][0 + i * 3] = res[0 + i * 2][6];
		matrix[0][1 + i * 3] = res[1 + i * 2][6];
		matrix[0][2 + i * 3] = res[4 + i][6];
	}

	delMatrix(res);

	return matrix;
}


void CImgprocessingsDoc::OnAffineTransform()
{
	int i, j, n, m;
	double** src, ** dst;
	double** matrix, ** temp;

	src = mkMatrix(3, 2);
	dst = mkMatrix(3, 2);

	src[0][0] = 100; src[0][1] = 50;
	src[1][0] = 200; src[1][1] = 50;
	src[2][0] = 100; src[2][1] = 100;

	dst[0][0] = 50; dst[0][1] = 50;
	dst[1][0] = 150; dst[1][1] = 50;
	dst[2][0] = 0; dst[2][1] = 100;

	for (i = 0; i < 3; i++) {
		for (n = -4; n < 5; n++) {
			if (m_height - 1 - ((int)src[i][1] + n) > m_height - 1 || m_height - 1 - ((int)src[i][1] + n) < 0)
				continue;
			for (m = -4; m < 5; m++) {
				if ((int)src[i][0] + m > m_width - 1 || (int)src[i][0] + m < 0)
					continue;
				m_InputImage[(m_height - 1 - ((int)src[i][1] + n)) * m_width + ((int)src[i][0] + m)] = 255;
			}
		}
	}

	m_Re_height = m_height;
	m_Re_width = m_width*2;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];
	m_tempImage = Image2DMem(m_height, m_width);

	matrix = OnGetAffineTransform(src, dst);

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_tempImage[i][j] = m_InputImage[i * m_width + j];
		}
	}

	temp = OnWarpAffine(m_tempImage, matrix, m_height, m_width, m_Re_height, m_Re_width, m_Re_width/2,0);

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_Re_width + j] = (unsigned char)temp[i][j];
		}
	}

	free(temp[0]);
	free(temp);
	free(m_tempImage[0]);
	free(m_tempImage);
	delMatrix(src);
	delMatrix(dst);
	delMatrix(matrix);
}


#pragma region Matrix
double** CImgprocessingsDoc::mkMatrix(size_t row, size_t column)
{
	double** matrix;
	matrix = (double**)malloc(sizeof(double*) * row);
	matrix[0] = (double*)calloc( (size_t)column * row, sizeof(double));
	for (int i = 1; i < row; i++)
	{
		matrix[i] = matrix[i - 1] + column;
	}
	return matrix;
}


void CImgprocessingsDoc::delMatrix(double** matrix)
{
	free(matrix[0]);
	free(matrix);
}


void CImgprocessingsDoc::delPivot(pivot* pivot)
{
	free(pivot->row);
	free(pivot->column);
}


double** CImgprocessingsDoc::rref_sol(double** mat, pivot* piv)
{
	size_t row = _msize(mat) / sizeof(double*);
	size_t column = _msize(mat[0]) / sizeof(double) / row - 1;
	pivot pivot;
	pivot.rank = 0;
	pivot.row = (int*)malloc(sizeof(int) * column);
	pivot.column = (int*)malloc(sizeof(int) * column);

	//duplicate
	double** matrix = mkMatrix(row, column + 1);
	memcpy(matrix[0], mat[0], _msize(mat[0]));

	// ~gaussion elimination
	for (int i = 0; i < row; i++)
	{
		// finding pivot
		int j;
		for (j = pivot.rank == 0 ? 0 : pivot.column[pivot.rank - 1] + 1; j < column; j++){
			if (matrix[i][j] != 0)
			{
				div1row(matrix[i], column, j);
				pivot.row[pivot.rank] = i;
				pivot.column[pivot.rank] = j;
				pivot.rank += 1;
				break;
			}
			else
			{
				for (int k = i + 1; k < row; k++)
				{
					if (matrix[k][j] != 0)
					{
						swap(matrix[i], matrix[k], column + 1);
						j--;
						break;
					}
				}
			}
		}
		if (j == column){
			for (int k = i; k < row; k++)
			{
				if (matrix[k][column] != 0)
				{
					pivot.rank = -1;
					if (piv)
					{
						piv->rank = pivot.rank;
					}
					delPivot(&pivot);
					delMatrix(matrix);
					return mat;
				}
			}
			break;
		}
		
		// gaussion elemination
		for (int j = i + 1; j < row; j++)
		{
			double temp = matrix[j][pivot.column[pivot.rank - 1]];
			for (int k = pivot.column[pivot.rank - 1]; k <= column; k++)
			{
				matrix[j][k] -= matrix[i][k] * temp;
			}
		}

	}

	// jordan elimination
	for (int i = 0; i < pivot.rank; i++)
	{
		for (int j = pivot.row[i] - 1; j >= 0; j--)
		{
			double temp = matrix[j][pivot.column[i]];
			for (int k = pivot.column[i]; k <= column; k++)
			{
				matrix[j][k] -= matrix[pivot.row[i]][k] * temp;
			}
		}
	}

	if (piv)
	{
		piv->rank = pivot.rank;
		piv->row = pivot.row;
		piv->column = pivot.column;
	}
	else
	{
		delPivot(&pivot);
	}
	return matrix;
}


void CImgprocessingsDoc::div1row(double* matrix, size_t column, int div)
{
	double temp = matrix[div];
	for (int i = div; i <= column; i++)
	{
		matrix[i] = matrix[i] / temp;
	}

}


void CImgprocessingsDoc::swap(double* a, double* b, size_t num)
{
	double* temp = (double*)malloc(sizeof(double) * num);
	memcpy(temp, a, sizeof(double) * num);
	memcpy(a, b, sizeof(double) * num);
	memcpy(b, temp, sizeof(double) * num);
	free(temp);
}


double** CImgprocessingsDoc::rref(double** matrix, int* check)
{
	pivot pivot;
	double** rref_matrix = rref_sol(matrix, &pivot);
	size_t row = _msize(rref_matrix) / sizeof(double*);
	size_t column = _msize(rref_matrix[0]) / sizeof(double) / row;

	if (pivot.rank == column - 1)
	{
		/*
		puts("unique solution\n");
		for (int i = 0; i < row; i++)
		{
			printf("   %lf\n", rref_matrix[i][column - 1]);
		}
		*/

		if (check != NULL)
			*check = 1;

		delPivot(&pivot);
		return rref_matrix;
		//delMatrix(rref_matrix);
	}
	else if (pivot.rank == -1)
	{
		if (check != NULL)
			*check = -1;

		return leastSquareMethod(matrix);
	}
	else
	{
		/*
		puts("infinite solution\n");
		int ppos = 0;
		for (int i = 0; i < column - 1; i++)
		{
			if (i == pivot.column[ppos])
			{
				ppos++;
				continue;
			}
			printf("x%d * [ ", i + 1);
			for (int j = 0; j < column - 1; j++)
			{


				if (j >= pivot.rank)
				{
					for (int k = j; k < column - 1; k++)
					{
						if (k == i)
							printf("%lf ", (double)1);
						else
							printf("%lf ", (double)0);
					}
					break;
				}
				else
				{
					if (j < row)
						printf("%lf ", -1 * rref_matrix[j][i]);
					else
						printf("%lf ", (double)0);
				}

			}
			puts("]T");
		}
		printf("const [ ");
		for (int j = 0; j < column - 1; j++)
		{
			if (j < row)
				printf("%lf ", rref_matrix[j][column - 1]);
			else
				printf("%lf ", (double)0);
		}
		puts("]T");
		*/

		if (check != NULL)
			*check = 0;

		delPivot(&pivot);
		return rref_matrix;
		//delMatrix(rref_matrix);
	}
}


double** CImgprocessingsDoc::matrixMul(double** a, double** b)
{
	size_t lm = _msize(a) / sizeof(double*);
	size_t ln = _msize(a[0]) / sizeof(double) / lm;

	size_t rm = _msize(b) / sizeof(double*);
	size_t rn = _msize(b[0]) / sizeof(double) / rm;

	if (ln != rm) return a;
	double** matrix = mkMatrix(lm, rn);
	for (int i = 0; i < lm; i++)
	{
		for (int j = 0; j < rn; j++)
		{
			matrix[i][j] = 0;
			for (int k = 0; k < ln; k++)
			{
				matrix[i][j] += a[i][k] * b[k][j];
			}
		}
	}

	return matrix;
}


double** CImgprocessingsDoc::transpose(double** matrix)
{
	size_t row = _msize(matrix) / sizeof(double*);
	size_t column = _msize(matrix[0]) / sizeof(double) / row;

	double** trans = mkMatrix(column, row);

	for (int i = 0; i < column; i++)
	{
		for (int j = 0; j < row; j++)
		{
			trans[i][j] = matrix[j][i];
		}
	}

	return trans;
}


double** CImgprocessingsDoc::augment(double** A, double** c)
{
	size_t row = _msize(A) / sizeof(double*);
	size_t column = _msize(A[0]) / sizeof(double) / row;

	double** aug = mkMatrix(row, column + 1);

	for (int i = 0; i < row; i++)
	{
		memcpy(aug[i], A[i], column * sizeof(double));
		aug[i][column] = c[i][0];
	}

	return aug;
}


double** CImgprocessingsDoc::decomposeAugment(double** matrix, double*** c)
{
	size_t row = _msize(matrix) / sizeof(double*);
	size_t column = _msize(matrix[0]) / sizeof(double) / row - 1;

	double** A = mkMatrix(row, column);
	*c = mkMatrix(row, (size_t)1);

	for (int i = 0; i < row; i++)
	{
		memcpy(A[i], matrix[i], column * sizeof(double));
		(*c)[i][0] = matrix[i][column];
	}

	return A;

}


double** CImgprocessingsDoc::leastSquareMethod(double** matrix)
{
	double** c;
	double** A = decomposeAugment(matrix, &c);
	double** AT = transpose(A);

	double** ATA = matrixMul(AT, A);
	double** ATc = matrixMul(AT, c);

	delMatrix(c);
	delMatrix(A);
	delMatrix(AT);

	double** aug = augment(ATA, ATc);

	delMatrix(ATA);
	delMatrix(ATc);

	//puts("[Least Square Method]");
	double** res = rref(aug, NULL);

	delMatrix(aug);

	return res;
}


double CImgprocessingsDoc::determinant(double** mat)
{
	int i, j, k;
	int row = (int)(_msize(mat) / sizeof(double*));
	int col = (int)(_msize(mat[0]) / sizeof(double) / row);
	double temp = 0, ** matrix;

	if (row != col)
		return 0;

	matrix = mkMatrix(row, col);
	memcpy(matrix[0], mat[0], sizeof(double) * row * col);

	for (i = 0; i < row; i++) {
		if (matrix[i][i] == 0) {
			for (j = i + 1; j < row; j++) {
				if (matrix[j][i] != 0) {
					add2row(matrix[i], matrix[j], col, i);
					goto gaussian_elimainattion;
				}
			}
			return 0;
		}

	gaussian_elimainattion:
		temp = 1;

		for (j = i + 1; j < row; j++) {
			temp = matrix[j][i] / matrix[i][i];
			for (k = i; k < col; k++)
				matrix[j][k] -= temp * matrix[i][k];
		}

	}

	temp = 1;
	for (i = 0; i < row; i++) {
		temp *= matrix[i][i];
	}

	delMatrix(matrix);

	return temp;
}


void CImgprocessingsDoc::add2row(double* matrix, double* addMat, int end, int start)
{
	int i;

	for (i = start; i < end; i++) {
		matrix[i] += addMat[i];
	}
}


double** CImgprocessingsDoc::inverse(double** mat)
{
	int i, j, k;
	int row = (int)(_msize(mat) / sizeof(double*));
	int col = (int)(_msize(mat[0]) / sizeof(double) / row);
	double det, ** matrix, ** invMat, temp;

	invMat = mkMatrix(row, col);

	for (i = 0; i < row; i++)
		invMat[i][i] = 1;

	if ((det = determinant(mat)) == 0) {
		return invMat;
	}

	matrix = mkMatrix(row, col);

	memcpy(matrix[0], mat[0], sizeof(double) * row * col);

	for (i = 0; i < row; i++) {
		if (matrix[i][i] == 0) {
			for (j = i + 1; j < row; j++) {
				if (matrix[j][i] != 0) {
					add2row(matrix[i], matrix[j], col, i);
					add2row(invMat[i], invMat[j], col, 0);
					goto gaussian_elimainattion;
				}
			}
		}

	gaussian_elimainattion:
		temp = matrix[i][i];
		div1row(matrix[i], (size_t)col - 1, i);
		for (j = 0; j < col; j++)
			invMat[i][j] /= temp;

		for (j = i + 1; j < row; j++) {
			temp = matrix[j][i];
			for (k = i; k < col; k++)
				matrix[j][k] -= temp * matrix[i][k];
			for (k = 0; k < col; k++)
				invMat[j][k] -= temp * invMat[i][k];
		}
	}

	for (i = row - 1; i > 0; i--) {
		for (j = i - 1; j >= 0; j--) {
			temp = matrix[j][i];
			matrix[j][i] = 0;
			for (k = 0; k < col; k++) {
				invMat[j][k] -= invMat[i][k] * temp;
			}
		}
	}

	delMatrix(matrix);

	return invMat;
}
#pragma endregion


void CImgprocessingsDoc::OnPerspectiveTransform()
{
	int i, j, n, m;
	double** src, ** dst, ** matrix, **temp;

	m_Re_height = m_height;
	m_Re_width = m_width;
	m_Re_size = m_Re_height * m_Re_width;

	m_OutputImage = new unsigned char[m_Re_size];
	m_tempImage = Image2DMem(m_height, m_width);

	src = mkMatrix(4, 2);
	dst = mkMatrix(4, 2);

	src[0][0] = 100; src[0][1] = 150;
	src[1][0] = 150; src[1][1] = 150;
	src[2][0] = 200; src[2][1] = 50;
	src[3][0] = 50; src[3][1] = 50;

	dst[0][0] = 0; dst[0][1] = m_Re_height/4;
	dst[1][0] = 0; dst[1][1] = m_Re_height/2-1;
	dst[2][0] = m_Re_width-1; dst[2][1] = m_Re_height-1;
	dst[3][0] = m_Re_width-1; dst[3][1] = 0;

	for (i = 0; i < 4; i++) {
		for (n = -4; n < 5; n++) {
			if (m_height - 1 - ((int)src[i][1] + n) > m_height - 1 || m_height - 1 - ((int)src[i][1] + n) < 0)
				continue;
			for (m = -4; m < 5; m++) {
				if ((int)src[i][0] + m > m_width - 1 || (int)src[i][0] + m < 0)
					continue;
				m_InputImage[(m_height - 1 - ((int)src[i][1] + n)) * m_width + ((int)src[i][0] + m)] = 255;
			}
		}
	}

	for (i = 0; i < m_height; i++) {
		for (j = 0; j < m_width; j++) {
			m_tempImage[i][j] = m_InputImage[i * m_width + j];
		}
	}

	matrix = OnGetPerspectiveTransform(src, dst);

	delMatrix(src);
	delMatrix(dst);

	temp = OnWarpPerspective(m_tempImage, matrix, m_height, m_width, m_Re_height, m_Re_width);

	for (i = 0; i < m_Re_height; i++) {
		for (j = 0; j < m_Re_width; j++) {
			m_OutputImage[i * m_Re_width + j] = (unsigned char)temp[i][j];
		}
	}

	delMatrix(matrix);
	delMatrix(temp);
	delMatrix(m_tempImage);
}


double** CImgprocessingsDoc::OnGetPerspectiveTransform(double** src, double** dst)
{
	int i, j;
	double** matrix = mkMatrix(8, 9);

	for (i = 0; i < 4; i++) {
		matrix[i][0] = src[i][0]; matrix[i][1] = src[i][1]; matrix[i][2] = 1;
		matrix[i][6] = -src[i][0] * dst[i][0]; matrix[i][7] = -src[i][1] * dst[i][0];
		matrix[i][8] = dst[i][0];

		matrix[i+4][3] = src[i][0]; matrix[i+4][4] = src[i][1]; matrix[i+4][5] = 1;
		matrix[i+4][6] = -src[i][0] * dst[i][1]; matrix[i+4][7] = -src[i][1] * dst[i][1];
		matrix[i+4][8] = dst[i][1];
	}

	double** homography = rref(matrix, NULL);

	delMatrix(matrix);

	matrix = mkMatrix(3, 3);

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			if (i == 2 && j == 2)
				break;
			matrix[i][j] = homography[i * 3 + j][8];
		}
	}

	matrix[2][2] = 1;

	delMatrix(homography);

	return matrix;
}


double** CImgprocessingsDoc::OnWarpPerspective(double** image, double** homography, int height, int width, int re_height, int re_width)
{
	int i, j, x, y;
	double weight;
	double** matrix, ** tempImage, Value;

	tempImage = mkMatrix(re_height, re_width);
	matrix = inverse(homography);

	/*
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			weight = j * matrix[2][0] + (height - 1 - i) * matrix[2][1] + 1;
			x = (int)((j * matrix[0][0] + (height - 1 - i) * matrix[0][1] + matrix[0][2]) / weight + 0.5);
			y = (int)(re_height-1 - ((j * matrix[1][0] + (height - 1 - i) * matrix[1][1] + matrix[1][2]) / weight) + 0.5);
			if (x<0 || x > re_width - 1 || y < 0 || y > re_height - 1)
				continue;
			else
				tempImage[y][x] = image[i][j];
		}
	}
	*/

	for (i = 0; i < re_height; i++) {
		for (j = 0; j < re_width; j++) {
			weight = j * matrix[2][0] + (re_height - 1 - i) * matrix[2][1] + matrix[2][2];
			x = (int)((j * matrix[0][0] + (re_height - 1 - i) * matrix[0][1] + matrix[0][2]) / weight + 0.5);
			y = (int)(height - 1 - ((j * matrix[1][0] + (re_height - 1 - i) * matrix[1][1] + matrix[1][2]) / weight) + 0.5);
			if (x < 0 || x > width - 1 || y < 0 || y > height - 1)
				if (j == 0 || j == re_width - 1 || i == 0 || i == re_height - 1)
					Value = 0;
				else
					Value = 255;
			else
				Value = image[y][x];

			tempImage[i][j] = Value;
		}
	}

	return tempImage;
}


void CImgprocessingsDoc::OnDeleteImage(double** image)
{
	free(image[0]);
	free(image);
}


void CImgprocessingsDoc::OnDeleteImageUC(unsigned char** image)
{
	free(image[0]);
	free(image);
}
