// HyperLink.cpp : implementation file
//
// HyperLink static control. Will open the default browser with the given URL
// when the user clicks on the link.
//
// Copyright (C) 1997, 1998 Chris Maunder (chrismaunder@codeguru.com)
// All rights reserved. May not be sold for profit.
//
// Thanks to Pål K. Tønder for auto-size and window caption changes.
//
// "GotoURL" function by Stuart Patterson
// As seen in the August, 1997 Windows Developer's Journal.
// Copyright 1997 by Miller Freeman, Inc. All rights reserved.
// Modified by Chris Maunder to use TCHARs instead of chars.
//
// "Default hand cursor" from Paul DiLascia's Jan 1998 MSJ article.
//

#include "stdafx.h"
#include "HyperLink.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

#define TOOLTIP_ID 1

BEGIN_MESSAGE_MAP(CHyperLink, CStatic)
    //{{AFX_MSG_MAP(CHyperLink)
    ON_CONTROL_REFLECT(STN_CLICKED, OnClicked)
    ON_WM_CTLCOLOR_REFLECT()
	ON_WM_TIMER()
    ON_WM_SETCURSOR()
	ON_WM_DESTROY()
    //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CHyperLink

CHyperLink::CHyperLink()
{
    m_hLinkCursor       = NULL;                 // No cursor as yet
    m_crLinkColor      = RGB(  0,   0, 255);   // Blue
    m_crVisitedColor   = RGB( 85,  26, 139);   // Purple
    m_crHoverColor     = RGB( 255,  0,   0);   // Red
    m_bOverControl      = FALSE;                // Cursor not yet over control
    m_bVisited          = FALSE;                // Hasn't been visited yet.
    m_bUnderline        = TRUE;                 // Underline the link?
    m_bAdjustToFit      = TRUE;                 // Resize the window to fit the text?
    m_strURL.Empty();
}

CHyperLink::~CHyperLink()
{
	m_Font.DeleteObject();
	if (m_hLinkCursor)
		DestroyCursor(m_hLinkCursor);
}

/////////////////////////////////////////////////////////////////////////////
// CHyperLink message handlers

BOOL CHyperLink::PreTranslateMessage(MSG* pMsg) 
{
    m_ToolTip.RelayEvent(pMsg);
    return CStatic::PreTranslateMessage(pMsg);
}

void CHyperLink::OnClicked()
{
	m_bVisited = TRUE; //(result > HINSTANCE_ERROR);
	SetVisited();						 // Repaint to show visited Color
	GotoURL(m_strURL, SW_SHOW);
}

HBRUSH CHyperLink::CtlColor(CDC* pDC, UINT) 
{
    if (m_bOverControl)
        pDC->SetTextColor(m_crHoverColor);
    else if (m_bVisited)
        pDC->SetTextColor(m_crVisitedColor);
    else
        pDC->SetTextColor(m_crLinkColor);

    // transparent text.
    pDC->SetBkMode(TRANSPARENT);
    return (HBRUSH)GetStockObject(NULL_BRUSH);
}

BOOL CHyperLink::OnSetCursor(CWnd* /*pWnd*/, UINT /*nHitTest*/, UINT /*message*/) 
{
    if (m_hLinkCursor)
    {
        ::SetCursor(m_hLinkCursor);
        return TRUE;
    }
    return FALSE;
}

void CHyperLink::PreSubclassWindow() 
{
    // We want to get mouse clicks via STN_CLICKED
    DWORD dwStyle = GetStyle();
    ::SetWindowLong(GetSafeHwnd(), GWL_STYLE, dwStyle | SS_NOTIFY);
    
    // Set the URL as the window text
    if (m_strURL.IsEmpty())
        GetWindowText(m_strURL);

    // Check that the window text isn't empty. If it is, set it as the URL.
    CString strWndText;
    GetWindowText(strWndText);
    if (strWndText.IsEmpty()) 
	{
        ASSERT(!m_strURL.IsEmpty());    // Window and URL both NULL. DUH!
        SetWindowText(m_strURL);
    }

    // Create the font
    LOGFONT lf;
    GetFont()->GetLogFont(&lf);
    lf.lfUnderline = (unsigned char) m_bUnderline;
    m_Font.CreateFontIndirect(&lf);
    SetFont(&m_Font);

    PositionWindow();        // Adjust size of window to fit URL if necessary
    SetDefaultCursor();      // Try and load up a "hand" cursor

    // Create the tooltip
    CRect rect; 
    GetClientRect(rect);
    m_ToolTip.Create(this);
    m_ToolTip.AddTool(this, m_strURL, rect, TOOLTIP_ID);

	SetTimer(1, 100, NULL);

    CStatic::PreSubclassWindow();
}

/////////////////////////////////////////////////////////////////////////////
// CHyperLink operations

void CHyperLink::SetURL(CString strURL)
{
    m_strURL = strURL;

    if (::IsWindow(GetSafeHwnd())) 
	{
        PositionWindow();
        m_ToolTip.UpdateTipText(strURL, this, TOOLTIP_ID);
    }
}

CString CHyperLink::GetURL() const
{ 
    return m_strURL;   
}

void CHyperLink::SetColors(COLORREF crLinkColor, COLORREF crVisitedColor,
                            COLORREF crHoverColor /* = -1 */) 
{ 
    m_crLinkColor    = crLinkColor; 
    m_crVisitedColor = crVisitedColor;

	if (crHoverColor == -1)
		m_crHoverColor = ::GetSysColor(COLOR_HIGHLIGHT);
	else
		m_crHoverColor = crHoverColor;

    if (::IsWindow(m_hWnd))
        Invalidate(); 
}

COLORREF CHyperLink::GetLinkColor() const
{ 
    return m_crLinkColor; 
}

COLORREF CHyperLink::GetVisitedColor() const
{
    return m_crVisitedColor; 
}

COLORREF CHyperLink::GetHoverColor() const
{
    return m_crHoverColor;
}

void CHyperLink::SetVisited(BOOL bVisited /* = TRUE */) 
{ 
    m_bVisited = bVisited; 

    if (::IsWindow(GetSafeHwnd()))
        Invalidate(); 
}

BOOL CHyperLink::GetVisited() const
{ 
    return m_bVisited; 
}

void CHyperLink::SetLinkCursor(HCURSOR hCursor)
{ 
    m_hLinkCursor = hCursor;
    if (m_hLinkCursor == NULL)
        SetDefaultCursor();
}

HCURSOR CHyperLink::GetLinkCursor() const
{
    return m_hLinkCursor;
}

void CHyperLink::SetUnderline(BOOL bUnderline /* = TRUE */)
{
    m_bUnderline = bUnderline;

    if (::IsWindow(GetSafeHwnd()))
    {
        LOGFONT lf;
        GetFont()->GetLogFont(&lf);
        lf.lfUnderline = (unsigned char) m_bUnderline;

        m_Font.DeleteObject();
        m_Font.CreateFontIndirect(&lf);
        SetFont(&m_Font);

        Invalidate(); 
    }
}

BOOL CHyperLink::GetUnderline() const
{ 
    return m_bUnderline; 
}

void CHyperLink::SetAutoSize(BOOL bAutoSize /* = TRUE */)
{
    m_bAdjustToFit = bAutoSize;

    if (::IsWindow(GetSafeHwnd()))
        PositionWindow();
}

BOOL CHyperLink::GetAutoSize() const
{ 
    return m_bAdjustToFit; 
}


// Move and resize the window so that the window is the same size
// as the hyperlink text. This stops the hyperlink cursor being active
// when it is not directly over the text. If the text is left justified
// then the window is merely shrunk, but if it is centred or right
// justified then the window will have to be moved as well.
//

void CHyperLink::PositionWindow()
{
    if (!::IsWindow(GetSafeHwnd()) || !m_bAdjustToFit) 
        return;

    // Get the current window position
    CRect rect;
    GetWindowRect(rect);

    CWnd* pParent = GetParent();
    if (pParent)
        pParent->ScreenToClient(rect);

    // Get the size of the window text
    CString strWndText;
    GetWindowText(strWndText);

    CDC* pDC = GetDC();
    CFont* pOldFont = pDC->SelectObject(&m_Font);
    CSize Extent = pDC->GetTextExtent(strWndText);
    pDC->SelectObject(pOldFont);
    ReleaseDC(pDC);

    // Get the text justification via the window style
    DWORD dwStyle = GetStyle();

    // Recalc the window size and position based on the text justification
    if (dwStyle & SS_CENTERIMAGE)
        rect.DeflateRect(0, (rect.Height() - Extent.cy)/2);
    else
        rect.bottom = rect.top + Extent.cy;

    if (dwStyle & SS_CENTER)   
        rect.DeflateRect((rect.Width() - Extent.cx)/2, 0);
    else if (dwStyle & SS_RIGHT) 
        rect.left  = rect.right - Extent.cx;
    else // SS_LEFT = 0, so we can't test for it explicitly 
        rect.right = rect.left + Extent.cx;

    // Move the window
    SetWindowPos(NULL, rect.left, rect.top, rect.Width(), rect.Height(), SWP_NOZORDER);
}

/////////////////////////////////////////////////////////////////////////////
// CHyperLink implementation

// The following appeared in Paul DiLascia's Jan 1998 MSJ articles.
// It loads a "hand" cursor from the winhlp32.exe module
void CHyperLink::SetDefaultCursor()
{
    if (m_hLinkCursor == NULL)                // No cursor handle - load our own
    {
        // Get the windows directory
        CString strWndDir;
        GetWindowsDirectory(strWndDir.GetBuffer(MAX_PATH), MAX_PATH);
        strWndDir.ReleaseBuffer();

        strWndDir += _T("\\winhlp32.exe");
        // This retrieves cursor #106 from winhlp32.exe, which is a hand pointer
        HMODULE hModule = LoadLibrary(strWndDir);
        if (hModule) 
		{
            HCURSOR hHandCursor = ::LoadCursor(hModule, MAKEINTRESOURCE(106));
            if (hHandCursor)
                m_hLinkCursor = CopyCursor(hHandCursor);
        }
        FreeLibrary(hModule);
    }
}

LONG CHyperLink::GetRegKey(HKEY key, LPCTSTR subkey, LPTSTR retdata)
{
	HKEY hkey;
	LONG retval = RegOpenKeyEx(key, subkey, 0, KEY_QUERY_VALUE, &hkey);
	
	*retdata = 0;

	if (retval == ERROR_SUCCESS) 
	{
		long datasize = MAX_PATH;
		TCHAR data[MAX_PATH];
		retval = RegQueryValue(hkey, NULL, data, &datasize);
		if (retval == ERROR_SUCCESS) 
		{
			lstrcpy(retdata, data);
			RegCloseKey(hkey);
		}
	}
	
	return retval;
}

void CHyperLink::ReportError(int nError)
{
    CString str;

	CString strError;
	strError.Format(_T("Unknown error (%d) occurred."), nError);
    str = _T("Unable to open hyperlink:\n\n") + strError;
    AfxMessageBox(str, MB_ICONEXCLAMATION | MB_OK);
}

HINSTANCE CHyperLink::GotoURL(LPCTSTR url, int showcmd)
{
	TCHAR key[MAX_PATH + MAX_PATH];

	// First try ShellExecute()
	HINSTANCE result = ShellExecute(NULL, _T("open"), url, NULL,NULL, showcmd);

	// If it failed, get the .htm regkey and lookup the program
	if ((UINT)result <= HINSTANCE_ERROR) 
	{
		if (GetRegKey(HKEY_CLASSES_ROOT, _T(".htm"), key) == ERROR_SUCCESS) 
		{
			lstrcat(key, _T("\\shell\\open\\command"));

			if (GetRegKey(HKEY_CLASSES_ROOT,key,key) == ERROR_SUCCESS) 
			{
				TCHAR *pos;
				pos = _tcsstr(key, _T("\"%1\""));
				if (pos == NULL) 
				{					                   // No quotes found
					pos = _tcsstr(key, _T("%1"));	   // Check for %1, without quotes 
					if (pos == NULL)				   // No parameter at all...
						pos = key + _tcslen(key) - 1;
					else
						*pos = _T('\0');				   // Remove the parameter
				}
				else
					*pos = _T('\0');					   // Remove the parameter

				_tcscat(pos, _T(" "));
				_tcscat(pos, url);
				//result = (HINSTANCE) WinExec(key, showcmd);
			}
		}
	}

	return result;
}

void CHyperLink::OnTimer(UINT)
{
	CPoint point;
	::GetCursorPos(&point);
	CRect rect;
	GetWindowRect(&rect);
	
	if (rect.PtInRect(point))		 // Cursor is currently over control
	{
		m_bOverControl = TRUE;
	}
	else
	{
		m_bOverControl = FALSE;
	}
	RedrawWindow();
}

void CHyperLink::OnDestroy()
{
	KillTimer(1);
	CStatic::OnDestroy();
}