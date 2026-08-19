import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Diabetes Clinical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .header-section {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(180deg, rgba(79, 195, 247, 0.1) 0%, transparent 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
    }
    
    .header-icon {
        font-size: 4rem;
        display: block;
        margin-bottom: 0.5rem;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4fc3f7, #81d4fa, #4fc3f7);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease-in-out infinite;
        letter-spacing: -0.5px;
    }
    
    @keyframes gradient {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .sub-title {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.1rem;
        font-weight: 300;
        margin-top: 0.5rem;
        letter-spacing: 0.5px;
    }
    
    .sub-title span {
        color: #4fc3f7;
        font-weight: 500;
    }
    
    .input-section {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0 2rem 0;
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 1.2rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4fc3f7 !important;
        box-shadow: 0 0 20px rgba(79, 195, 247, 0.15) !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.3) !important;
    }
    
    .btn-container {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin-top: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4fc3f7, #0288d1) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(79, 195, 247, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(79, 195, 247, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    .btn-clear > button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: rgba(255, 255, 255, 0.7) !important;
        box-shadow: none !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .btn-clear > button:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }
    
    .answer-section {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .answer-label {
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    .answer-box {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1.5rem;
        color: #e0e0e0;
        font-size: 1.05rem;
        line-height: 1.8;
        border-left: 3px solid #4fc3f7;
    }
    
    .answer-box strong {
        color: #4fc3f7;
    }
    
    .confidence-section {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-top: 1rem;
        padding: 0.75rem 1.25rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .confidence-label {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.85rem;
    }
    
    .confidence-value {
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .confidence-bar {
        flex: 1;
        height: 6px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 3px;
        overflow: hidden;
        margin: 0 0.5rem;
    }
    
    .confidence-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.8s ease;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .status-confident {
        background: rgba(102, 187, 106, 0.2);
        color: #66bb6a;
        border: 1px solid rgba(102, 187, 106, 0.3);
    }
    
    .status-low {
        background: rgba(239, 83, 80, 0.2);
        color: #ef5350;
        border: 1px solid rgba(239, 83, 80, 0.3);
    }
    
    .sources-section {
        margin-top: 1.5rem;
    }
    
    .sources-header {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .source-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    
    .source-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(255, 255, 255, 0.1);
        transform: translateX(4px);
    }
    
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .source-number {
        color: rgba(255, 255, 255, 0.3);
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .source-score {
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.05);
    }
    
    .source-score-high {
        color: #66bb6a;
    }
    
    .source-score-medium {
        color: #ffd54f;
    }
    
    .source-score-low {
        color: #ef5350;
    }
    
    .source-doc {
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    
    .source-doc strong {
        color: rgba(255, 255, 255, 0.6);
    }
    
    .source-content {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .sidebar-section {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .sidebar-label {
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    
    .sidebar-value {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    
    .quick-btn {
        width: 100%;
        text-align: left !important;
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        padding: 0.6rem 1rem !important;
        color: rgba(255, 255, 255, 0.7) !important;
        font-size: 0.85rem !important;
        transition: all 0.3s ease !important;
        margin-bottom: 0.3rem !important;
    }
    
    .quick-btn:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        color: white !important;
        border-color: rgba(79, 195, 247, 0.3) !important;
        transform: translateX(4px);
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: rgba(255, 255, 255, 0.2);
        font-size: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.03);
        margin-top: 2rem;
    }
    
    .footer a {
        color: rgba(255, 255, 255, 0.3);
        text-decoration: none;
    }
    
    .footer a:hover {
        color: rgba(79, 195, 247, 0.6);
    }
    
    .spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-top: 2px solid #4fc3f7;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        margin-right: 10px;
        vertical-align: middle;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .stSlider > div > div > div {
        background: #4fc3f7 !important;
    }
    
    .stSlider > div > div > div > div {
        background: #4fc3f7 !important;
    }
    
    .stAlert {
        background: rgba(239, 83, 80, 0.1) !important;
        border: 1px solid rgba(239, 83, 80, 0.2) !important;
        color: #ef5350 !important;
        border-radius: 12px !important;
    }
    
    .stSuccess {
        background: rgba(102, 187, 106, 0.1) !important;
        border: 1px solid rgba(102, 187, 106, 0.2) !important;
        color: #66bb6a !important;
        border-radius: 12px !important;
    }
    
    .stExpander {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
    }
    
    .stExpander > div > div {
        color: rgba(255, 255, 255, 0.8) !important;
    }
    
    .stExpander:hover {
        border-color: rgba(79, 195, 247, 0.2) !important;
    }
    
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border: 1px solid rgba(255, 193, 7, 0.2);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: #ffd54f;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    .badge-container {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 0.5rem 0;
    }
    
    .badge {
        background: rgba(79, 195, 247, 0.1);
        border: 1px solid rgba(79, 195, 247, 0.15);
        border-radius: 20px;
        padding: 0.15rem 0.8rem;
        font-size: 0.7rem;
        color: rgba(255, 255, 255, 0.5);
        letter-spacing: 0.5px;
    }
    
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .answer-box {
            font-size: 0.95rem;
            padding: 1rem;
        }
        .header-section {
            padding: 1rem 0;
        }
    }
</style>
""", unsafe_allow_html=True)

if "question" not in st.session_state:
    st.session_state.question = ""
if "asked" not in st.session_state:
    st.session_state.asked = False

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem 0;">
        <span style="font-size:2.5rem;">🩺</span>
        <h3 style="color:rgba(255,255,255,0.9);font-weight:600;margin:0.25rem 0 0 0;">Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-label">Number of Sources</div>
        <div class="sidebar-value">Top-<span id="topk_display">5</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    top_k = st.slider("", 1, 10, 5, label_visibility="collapsed")
    
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-label">System Status</div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            data = r.json()
            chunks = data.get('chunks', 0)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.5rem;padding:0.25rem 0;">
                <span style="color:#66bb6a;">●</span>
                <span style="color:rgba(255,255,255,0.7);">Online</span>
                <span style="color:rgba(255,255,255,0.3);font-size:0.8rem;margin-left:auto;">{chunks} chunks</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:0.5rem;padding:0.25rem 0;">
                <span style="color:#ef5350;">●</span>
                <span style="color:rgba(255,255,255,0.5);">Offline</span>
            </div>
            """, unsafe_allow_html=True)
    except:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.5rem;padding:0.25rem 0;">
            <span style="color:#ef5350;">●</span>
            <span style="color:rgba(255,255,255,0.5);">Cannot connect</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="sidebar-label" style="margin-bottom:0.75rem;">Quick Questions</div>
    """, unsafe_allow_html=True)
    
    quick_questions = [
        "What is the recommended screening for diabetes?",
        "What is the target blood pressure for diabetes?",
        "What are the first-line medications for diabetes?",
        "What is the recommended HbA1c target?"
    ]
    
    for q in quick_questions:
        if st.button(q, key=f"quick_{q[:20]}", use_container_width=True):
            st.session_state.question = q
            st.session_state.asked = True
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("""
    <div style="padding:0.5rem 0;">
        <div class="badge-container">
            <span class="badge">USPSTF 2021</span>
            <span class="badge">WHO 2008</span>
            <span class="badge">RAG</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="main-container">
    <div class="header-section">
        <span class="header-icon">🩺</span>
        <div class="main-title">Diabetes Clinical Assistant</div>
        <div class="sub-title">
            Evidence-based answers from <span>USPSTF</span> and <span>WHO</span> diabetes guidelines
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([0.5, 5, 0.5])
with col2:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    question = st.text_input(
        "",
        value=st.session_state.question,
        placeholder="Ask a clinical question about diabetes...",
        label_visibility="collapsed",
        key="question_input"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    with col_btn1:
        ask_clicked = st.button("🔍 Ask", type="primary", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("✕ Clear", use_container_width=True)
    
    if clear_clicked:
        st.session_state.question = ""
        st.session_state.asked = False
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

if ask_clicked and question:
    st.session_state.question = question
    st.session_state.asked = True

if st.session_state.asked and st.session_state.question:
    with st.spinner("Searching guidelines..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"question": st.session_state.question, "top_k": top_k},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                st.markdown("""
                <div class="answer-section">
                    <div class="answer-label">✧ Answer</div>
                """, unsafe_allow_html=True)
                
                answer_text = data.get("answer", "No answer available.")
                
                if "don't have enough information" in answer_text.lower():
                    st.markdown(f"""
                    <div class="answer-box" style="border-left-color:#ffd54f;">
                        <span style="font-size:1.2rem;margin-right:0.5rem;">📋</span>
                        {answer_text}
                    </div>
                    <div class="warning-box">
                        ⚠️ Try rephrasing your question or using different keywords.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="answer-box">
                        {answer_text}
                    </div>
                    """, unsafe_allow_html=True)
                
                confidence = data.get("confidence", 0.0)
                is_confident = data.get("is_confident", False)
                
                conf_pct = int(confidence * 100)
                bar_color = "#66bb6a" if conf_pct > 70 else "#ffd54f" if conf_pct > 50 else "#ef5350"
                status_text = "High Confidence" if conf_pct > 70 else "Moderate Confidence" if conf_pct > 50 else "Low Confidence"
                status_class = "status-confident" if conf_pct > 50 else "status-low"
                
                st.markdown(f"""
                <div class="confidence-section">
                    <span class="confidence-label">Confidence</span>
                    <span class="confidence-value" style="color:{bar_color};">{conf_pct}%</span>
                    <div class="confidence-bar">
                        <div class="confidence-bar-fill" style="width:{conf_pct}%;background:{bar_color};"></div>
                    </div>
                    <span class="status-badge {status_class}">{status_text}</span>
                </div>
                """, unsafe_allow_html=True)
                
                sources = data.get("sources", [])
                if sources:
                    st.markdown("""
                    <div class="sources-section">
                        <div class="sources-header">📚 Sources</div>
                    """, unsafe_allow_html=True)
                    
                    for i, source in enumerate(sources, 1):
                        score = source.get("score", 0.0)
                        doc = source.get("document", "Unknown")
                        page = source.get("page", "?")
                        content = source.get("content", "No content available")
                        
                        if score > 0.7:
                            score_class = "source-score-high"
                            score_label = "High"
                        elif score > 0.5:
                            score_class = "source-score-medium"
                            score_label = "Medium"
                        else:
                            score_class = "source-score-low"
                            score_label = "Low"
                        
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-header">
                                <span class="source-number">#{i}</span>
                                <span class="source-score {score_class}">{score_label} • {score:.3f}</span>
                            </div>
                            <div class="source-doc">
                                <strong>📄 {doc}</strong> — Page {page}
                            </div>
                            <div class="source-content">
                                {content}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                st.error(f"⚠️ Error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            st.error("⏰ Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to API. Make sure the backend is running.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("""
<div class="footer">
    <span>🩺 Clinical decision support only</span>
    <span style="margin:0 0.5rem;">·</span>
    <span>Always consult a healthcare professional</span>
    <br>
    <span style="font-size:0.7rem;opacity:0.5;">Powered by RAG · USPSTF & WHO Guidelines</span>
</div>
""", unsafe_allow_html=True)