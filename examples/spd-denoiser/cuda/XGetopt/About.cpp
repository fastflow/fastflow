#include "stdafx.h"
#include "About.h"

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
	//{{AFX_MSG_MAP(CAboutDlg)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
	TRACE0("in CAboutDlg::CAboutDlg\n");
	//{{AFX_DATA_INIT(CAboutDlg)
	//}}AFX_DATA_INIT
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CAboutDlg)
	DDX_Control(pDX, IDC_ABOUT_EMAIL, m_ctrlEmail);
	//}}AFX_DATA_MAP
}

BOOL CAboutDlg::OnInitDialog() 
{
	CDialog::OnInitDialog();
	
	m_ctrlEmail.SetURL(_T("mailto:hdietrich2@hotmail.com"));
	m_ctrlEmail.SetLinkCursor(AfxGetApp()->LoadCursor(IDC_ABOUT_HYPERLINK));
	
	CenterWindow();

	return TRUE;
}
