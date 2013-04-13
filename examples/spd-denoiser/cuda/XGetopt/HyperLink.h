// HyperLink.h : header file
//
//
// HyperLink static control. Will open the default browser with the given URL
// when the user clicks on the link.
//
// Copyright Chris Maunder, 1997, 1998
// Feel free to use and distribute. May not be sold for profit. 

#ifndef HYPERLINK_H
#define HYPERLINK_H

/////////////////////////////////////////////////////////////////////////////
// CHyperLink window

class CHyperLink : public CStatic
{
// Construction/destruction
public:
    CHyperLink();
    virtual ~CHyperLink();

// Attributes
public:

// Operations
public:

	void SetURL(CString strURL);
	CString GetURL() const;

	void SetColors(COLORREF crLinkColor, COLORREF crVisitedColor, 
					COLORREF crHoverColor = -1);
	COLORREF GetLinkColor() const;
	COLORREF GetVisitedColor() const;
	COLORREF GetHoverColor() const;

	void SetVisited(BOOL bVisited = TRUE);
	BOOL GetVisited() const;

	void SetLinkCursor(HCURSOR hCursor);
	HCURSOR GetLinkCursor() const;

	void SetUnderline(BOOL bUnderline = TRUE);
	BOOL GetUnderline() const;

	void SetAutoSize(BOOL bAutoSize = TRUE);
	BOOL GetAutoSize() const;

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CHyperLink)
public:
	virtual BOOL PreTranslateMessage(MSG* pMsg);
protected:
	virtual void PreSubclassWindow();
	//}}AFX_VIRTUAL

// Implementation
protected:
	HINSTANCE GotoURL(LPCTSTR url, int showcmd);
	void ReportError(int nError);
	LONG GetRegKey(HKEY key, LPCTSTR subkey, LPTSTR retdata);
	void PositionWindow();
	void SetDefaultCursor();

// Protected attributes
protected:
    COLORREF m_crLinkColor, m_crVisitedColor;       // Hyperlink Colors
    COLORREF m_crHoverColor;                        // Hover Color
    BOOL     m_bOverControl;                        // cursor over control?
    BOOL     m_bVisited;                            // Has it been visited?
    BOOL     m_bUnderline;                          // underline hyperlink?
    BOOL     m_bAdjustToFit;                        // Adjust window size to fit text?
    CString  m_strURL;                              // hyperlink URL
    CFont    m_Font;                                // Underline font if necessary
	HCURSOR  m_hLinkCursor; 						// Cursor for hyperlink
	CToolTipCtrl m_ToolTip; 						// The tooltip

	// Generated message map functions
protected:
	//{{AFX_MSG(CHyperLink)
	afx_msg HBRUSH CtlColor(CDC* pDC, UINT nCtlColor);
	afx_msg BOOL OnSetCursor(CWnd* pWnd, UINT nHitTest, UINT message);
	afx_msg void OnTimer(UINT nIDEvent);
	afx_msg void OnDestroy();
	//}}AFX_MSG
	afx_msg void OnClicked();
	DECLARE_MESSAGE_MAP()
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Developer Studio will insert additional declarations immediately before the previous line.

#endif //HYPERLINK_H
