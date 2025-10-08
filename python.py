import streamlit as st
import pandas as pd
import numpy as np
import json
import math
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh Giá Phương Án Kinh Doanh (NPV/IRR)",
    layout="wide"
)

st.title("Ứng dụng Đánh Giá Phương Án Kinh Doanh 💰")

# --- Hàm tính toán chính: NPV, IRR, PP, DPP ---

@st.cache_data
def calculate_metrics(investment, lifespan, revenue_per_year, cost_per_year, wacc_rate, tax_rate):
    """
    Xây dựng dòng tiền và tính toán các chỉ số đánh giá dự án.
    Args:
        investment (float): Vốn đầu tư ban đầu (năm 0, outflow).
        lifespan (int): Dòng đời dự án (số năm).
        revenue_per_year (float): Doanh thu hàng năm (giả định cố định).
        cost_per_year (float): Chi phí hoạt động hàng năm (giả định cố định).
        wacc_rate (float): Tỷ suất chiết khấu (WACC).
        tax_rate (float): Thuế suất thu nhập doanh nghiệp.

    Returns:
        tuple: (cash_flow_df, metrics_dict)
    """
    # 1. Xây dựng Dòng tiền (Cash Flow)
    years = list(range(lifespan + 1))
    
    # Giả định: Investment là chi phí duy nhất ở Năm 0
    initial_cash_flow = -investment
    
    # Tính toán dòng tiền hoạt động (Operating Cash Flow - OCF) cho các năm sau
    EBIT = revenue_per_year - cost_per_year
    TAX = EBIT * tax_rate
    EAT = EBIT - TAX # Earnings After Tax
    # OCF = EAT + Depreciation - Changes in NWC. 
    # Ở đây giả định đơn giản: OCF = EAT (hoặc OCF = EBIT * (1-Tax) nếu bỏ qua Depreciation)
    # Ta sẽ dùng Cash Flow = Lợi nhuận ròng + Khấu hao. 
    # Để đơn giản hóa, ta giả định OCF = EAT. (Thường dùng cho phân tích nhanh)
    
    # Ta dùng công thức: OCF = (Doanh thu - Chi phí) * (1 - Thuế)
    # Giả định không có khấu hao. Nếu có khấu hao, OCF = (Doanh thu - Chi phí - Khấu hao)*(1-T) + Khấu hao
    # Dùng cách đơn giản nhất: Lợi nhuận Ròng = (Doanh thu - Chi phí) * (1 - Thuế)
    
    net_income = (revenue_per_year - cost_per_year) * (1 - tax_rate)
    
    # Dòng tiền cho các năm 1 đến lifespan
    periodic_cash_flows = [net_income] * lifespan
    
    # Tổng dòng tiền: [CF_0, CF_1, CF_2, ...]
    cash_flows = [initial_cash_flow] + periodic_cash_flows
    
    # Tạo DataFrame hiển thị
    cash_flow_df = pd.DataFrame({
        'Năm': years,
        'Dòng tiền': cash_flows
    })
    
    # 2. Tính toán các chỉ số
    # NPV (Net Present Value)
    try:
        npv_value = np.npv(wacc_rate, cash_flows)
    except Exception:
        npv_value = float('nan')
        
    # IRR (Internal Rate of Return)
    try:
        # IRR requires a switch from negative to positive cash flow
        if any(cf > 0 for cf in cash_flows):
            irr_value = np.irr(cash_flows)
        else:
            irr_value = float('nan') # Dự án không có dòng tiền dương
    except Exception:
        irr_value = float('nan')
        
    # PP (Payback Period - Thời gian hoàn vốn)
    cumulative_cf = np.cumsum(cash_flows)
    pp_year = 0
    pp_fraction = 0
    for i in range(1, lifespan + 1):
        if cumulative_cf[i] >= 0:
            pp_year = i - 1
            # PP = Năm cuối dòng tiền âm + (Dòng tiền lũy kế âm cuối năm / Dòng tiền năm hoàn vốn)
            if i > 0 and cash_flows[i] > 0:
                 # Lấy giá trị tuyệt đối của dòng tiền lũy kế cuối năm âm (năm trước đó)
                remaining_amount = abs(cumulative_cf[i-1]) if cumulative_cf[i-1] < 0 else 0 
                pp_fraction = remaining_amount / cash_flows[i]
            break
    payback_period = pp_year + pp_fraction
    if payback_period == 0 and investment > 0 and net_income <= 0: # Trường hợp không hoàn vốn
        payback_period = float('inf')

    # DPP (Discounted Payback Period - Thời gian hoàn vốn có chiết khấu)
    discount_factors = [1 / (1 + wacc_rate)**t for t in years]
    discounted_cf = [cash_flows[i] * discount_factors[i] for i in years]
    
    cumulative_discounted_cf = np.cumsum(discounted_cf)
    dpp_year = 0
    dpp_fraction = 0
    for i in range(1, lifespan + 1):
        if cumulative_discounted_cf[i] >= 0:
            dpp_year = i - 1
            # DPP = Năm cuối dòng tiền chiết khấu lũy kế âm + (Dòng tiền chiết khấu lũy kế âm cuối năm / Dòng tiền chiết khấu năm hoàn vốn)
            if i > 0 and discounted_cf[i] > 0:
                remaining_amount = abs(cumulative_discounted_cf[i-1]) if cumulative_discounted_cf[i-1] < 0 else 0
                dpp_fraction = remaining_amount / discounted_cf[i]
            break
    discounted_payback_period = dpp_year + dpp_fraction
    if discounted_payback_period == 0 and investment > 0 and net_income <= 0: # Trường hợp không hoàn vốn
        discounted_payback_period = float('inf')

    
    metrics_dict = {
        "NPV": npv_value,
        "IRR": irr_value,
        "PP": payback_period,
        "DPP": discounted_payback_period
    }
    
    return cash_flow_df, metrics_dict

# --- Hàm gọi AI để Trích xuất Dữ liệu (Nhiệm vụ 1) ---
def extract_financial_data_ai(document_text, api_key):
    """
    Sử dụng Gemini API với JSON Schema để trích xuất các thông tin tài chính.
    """
    if not api_key:
        return None, "Lỗi: Vui lòng nhập Khóa API."

    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        system_prompt = (
            "Bạn là một chuyên gia phân tích dự án kinh doanh. Nhiệm vụ của bạn là đọc "
            "nội dung tài liệu kinh doanh và trích xuất sáu thông số tài chính quan trọng. "
            "Hãy đảm bảo rằng tất cả các giá trị được trả về dưới dạng số (float hoặc integer) "
            "theo cấu trúc JSON chính xác. Tỷ lệ (WACC và Thuế) phải được chuyển thành số thập phân (ví dụ: 10% là 0.10). "
            "Nếu không tìm thấy giá trị, hãy sử dụng giá trị mặc định hợp lý (ví dụ: Thuế 0.2, WACC 0.1)."
        )

        user_query = (
            f"Vui lòng trích xuất các thông số sau từ nội dung tài liệu: "
            "Vốn đầu tư ban đầu (Investment), Dòng đời dự án (Lifespan, tính bằng năm), "
            "Doanh thu ước tính hàng năm (Revenue_Per_Year), Chi phí hoạt động hàng năm (Cost_Per_Year), "
            "Tỷ suất chiết khấu (WACC_Rate) và Thuế suất (Tax_Rate)."
            f"\n\nNội dung tài liệu:\n---\n{document_text}\n---"
        )
        
        # Định nghĩa JSON Schema bắt buộc
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "Investment": {"type": "NUMBER", "description": "Vốn đầu tư ban đầu (ví dụ: 100000000)."},
                "Lifespan": {"type": "INTEGER", "description": "Số năm của dự án (ví dụ: 5)."},
                "Revenue_Per_Year": {"type": "NUMBER", "description": "Doanh thu trung bình hàng năm."},
                "Cost_Per_Year": {"type": "NUMBER", "description": "Chi phí hoạt động trung bình hàng năm."},
                "WACC_Rate": {"type": "NUMBER", "description": "Tỷ suất chiết khấu (WACC), ví dụ 0.15 cho 15%."},
                "Tax_Rate": {"type": "NUMBER", "description": "Thuế suất TNDN, ví dụ 0.2 cho 20%."}
            },
            "required": ["Investment", "Lifespan", "Revenue_Per_Year", "Cost_Per_Year", "WACC_Rate", "Tax_Rate"]
        }

        response = client.models.generate_content(
            model=model_name,
            contents=[{"parts": [{"text": user_query}]}],
            config={
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
                "response_schema": response_schema
            }
        )
        
        # Xử lý đầu ra JSON
        try:
            extracted_data = json.loads(response.text)
            return extracted_data, None
        except json.JSONDecodeError:
            return None, f"Lỗi phân tích JSON từ AI: {response.text}"
        
    except APIError as e:
        return None, f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return None, f"Đã xảy ra lỗi không xác định: {e}"

# --- Hàm gọi AI để Phân tích Chỉ số (Nhiệm vụ 4) ---
def get_project_analysis(metrics_data, wacc, api_key):
    """Gửi các chỉ số đã tính toán đến Gemini API và nhận nhận xét."""
    if not api_key:
        return "Lỗi: Không thể phân tích vì thiếu Khóa API."
        
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'

        metrics_markdown = pd.Series(metrics_data).to_markdown()

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính/đầu tư cấp cao. Dựa trên các chỉ số hiệu quả dự án sau, hãy đưa ra một đánh giá toàn diện, chuyên nghiệp và có tính thuyết phục về tính khả thi của dự án. 
        Đánh giá cần tập trung vào:
        1. Tính khả thi của dự án (Dựa trên NPV và IRR so với WACC).
        2. Mức độ rủi ro (Dựa trên thời gian hoàn vốn PP và DPP).
        3. Kết luận và khuyến nghị rõ ràng (Chấp nhận hay Từ chối dự án).
        
        WACC (Tỷ suất chiết khấu): {wacc:.2%}
        
        Các Chỉ số Hiệu quả Dự án:
        {metrics_markdown}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định trong quá trình phân tích: {e}"


# --- Cấu trúc Giao diện Streamlit ---

# Sidebar để nhập API Key
with st.sidebar:
    st.subheader("Cấu hình API Key")
    # Sử dụng st.secrets nếu triển khai trên Streamlit Cloud
    api_key_default = st.secrets.get("GEMINI_API_KEY", "")
    api_key = st.text_input(
        "Nhập Khóa Gemini API", 
        type="password", 
        value=api_key_default,
        help="Khóa API của Google Gemini, cần thiết cho việc trích xuất dữ liệu và phân tích."
    )
    st.markdown("---")

# --- Nhiệm vụ 1: Lọc Dữ liệu bằng AI ---
st.header("1. Trích xuất Thông số Dự án từ Tài liệu")

st.info(
    "**Lưu ý:** Vui lòng sao chép toàn bộ nội dung (text) từ file Word (đã bao gồm các thông tin về Vốn đầu tư, Dòng đời, Doanh thu, Chi phí, WACC, Thuế) và dán vào ô bên dưới."
)

document_content = st.text_area(
    "Dán nội dung Tài liệu Phương án Kinh doanh tại đây:",
    height=300,
    placeholder="Ví dụ: 'Dự án có Vốn đầu tư là 10 tỷ VND, hoạt động trong 5 năm. Doanh thu hàng năm 3 tỷ, chi phí 1 tỷ. Tỷ suất chiết khấu (WACC) là 12%. Thuế suất TNDN 20%.'"
)

# Khởi tạo state để lưu trữ dữ liệu đã trích xuất
if 'extracted_params' not in st.session_state:
    st.session_state.extracted_params = None

col_extract, col_placeholder = st.columns([1, 4])
with col_extract:
    extract_button = st.button("Tạo tác Lọc Dữ liệu (AI)")

if extract_button and document_content:
    if not api_key:
        st.error("Vui lòng nhập Khóa Gemini API trước khi thực hiện trích xuất.")
    else:
        with st.spinner("Đang gửi tài liệu cho AI để trích xuất thông số..."):
            extracted_data, error = extract_financial_data_ai(document_content, api_key)

            if extracted_data:
                st.session_state.extracted_params = extracted_data
                st.success("Trích xuất thông số thành công!")
            elif error:
                st.error(f"Lỗi trích xuất: {error}")

# Hiển thị thông số đã trích xuất
if st.session_state.extracted_params:
    st.subheader("Thông số Dự án đã Trích xuất:")
    params = st.session_state.extracted_params
    
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    
    col1.metric("Vốn đầu tư (Investment)", f"{params['Investment']:,.0f} VND")
    col2.metric("Dòng đời dự án (Lifespan)", f"{params['Lifespan']} năm")
    col3.metric("Doanh thu Hàng năm", f"{params['Revenue_Per_Year']:,.0f} VND")
    col4.metric("Chi phí Hàng năm", f"{params['Cost_Per_Year']:,.0f} VND")
    col5.metric("WACC (Tỷ suất chiết khấu)", f"{params['WACC_Rate']:.2%}")
    col6.metric("Thuế suất (Tax Rate)", f"{params['Tax_Rate']:.2%}")
    
    st.markdown("---")

# --- Nhiệm vụ 2 & 3: Xây dựng Dòng tiền và Tính toán Chỉ số ---
st.header("2. Xây dựng Dòng tiền & 3. Tính toán Chỉ số Hiệu quả")

if st.session_state.extracted_params:
    
    params = st.session_state.extracted_params
    
    try:
        cash_flow_df, metrics = calculate_metrics(
            investment=params['Investment'],
            lifespan=params['Lifespan'],
            revenue_per_year=params['Revenue_Per_Year'],
            cost_per_year=params['Cost_Per_Year'],
            wacc_rate=params['WACC_Rate'],
            tax_rate=params['Tax_Rate']
        )
        
        # 2. Xây dựng Bảng dòng tiền
        st.subheader("Bảng Dòng tiền của Dự án:")
        st.dataframe(
            cash_flow_df.style.format({'Dòng tiền': '{:,.0f} VND'}), 
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        # 3. Tính toán các chỉ số
        st.subheader("Các Chỉ số Đánh giá Hiệu quả Dự án:")
        
        col_npv, col_irr, col_pp, col_dpp = st.columns(4)
        
        # Format NPV
        npv_color = 'green' if metrics['NPV'] > 0 else 'red'
        col_npv.markdown(
            f"""<div style="background-color: #f0f2f6; padding: 10px; border-radius: 8px; border-left: 5px solid {npv_color};">
            <p style="margin: 0; font-size: 14px; color: #333;">**NPV (Giá trị hiện tại ròng)**</p>
            <h3 style="margin: 0; color: {npv_color};">{metrics['NPV']:,.0f} VND</h3>
            </div>""", unsafe_allow_html=True
        )

        # Format IRR
        irr_value_str = f"{metrics['IRR']:.2%}" if not math.isinf(metrics['IRR']) and not math.isnan(metrics['IRR']) else "Không xác định"
        irr_color = 'green' if not math.isinf(metrics['IRR']) and not math.isnan(metrics['IRR']) and metrics['IRR'] > params['WACC_Rate'] else 'red'
        col_irr.markdown(
            f"""<div style="background-color: #f0f2f6; padding: 10px; border-radius: 8px; border-left: 5px solid {irr_color};">
            <p style="margin: 0; font-size: 14px; color: #333;">**IRR (Tỷ suất sinh lời nội tại)**</p>
            <h3 style="margin: 0; color: {irr_color};">{irr_value_str}</h3>
            </div>""", unsafe_allow_html=True
        )

        # Format PP
        pp_color = 'blue'
        pp_value_str = f"{metrics['PP']:.2f} năm" if not math.isinf(metrics['PP']) else "Không hoàn vốn"
        col_pp.markdown(
            f"""<div style="background-color: #f0f2f6; padding: 10px; border-radius: 8px; border-left: 5px solid {pp_color};">
            <p style="margin: 0; font-size: 14px; color: #333;">**PP (Thời gian hoàn vốn)**</p>
            <h3 style="margin: 0; color: {pp_color};">{pp_value_str}</h3>
            </div>""", unsafe_allow_html=True
        )
        
        # Format DPP
        dpp_color = 'blue'
        dpp_value_str = f"{metrics['DPP']:.2f} năm" if not math.isinf(metrics['DPP']) else "Không hoàn vốn"
        col_dpp.markdown(
            f"""<div style="background-color: #f0f2f6; padding: 10px; border-radius: 8px; border-left: 5px solid {dpp_color};">
            <p style="margin: 0; font-size: 14px; color: #333;">**DPP (Hoàn vốn có chiết khấu)**</p>
            <h3 style="margin: 0; color: {dpp_value_str}</h3>
            </div>""", unsafe_allow_html=True
        )

        st.session_state.metrics = metrics # Lưu metrics vào session state
        st.session_state.wacc = params['WACC_Rate']
        
    except Exception as e:
        st.error(f"Lỗi tính toán dòng tiền và chỉ số: {e}. Vui lòng kiểm tra lại thông số trích xuất.")
        st.session_state.metrics = None
        st.session_state.wacc = None

# --- Nhiệm vụ 4: Phân tích Chỉ số bằng AI ---
st.header("4. Phân tích Hiệu quả Dự án (AI)")

if st.session_state.extracted_params and st.session_state.metrics:
    if st.button("Yêu cầu AI Phân tích Chỉ số Hiệu quả"):
        if not api_key:
            st.error("Vui lòng nhập Khóa Gemini API để thực hiện phân tích.")
        else:
            with st.spinner('Đang gửi dữ liệu và chờ Gemini AI phân tích...'):
                ai_result = get_project_analysis(
                    st.session_state.metrics, 
                    st.session_state.wacc, 
                    api_key
                )
                
                st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                st.info(ai_result)
else:
    st.warning("Vui lòng trích xuất thông số dự án và tính toán dòng tiền trước.")

st.markdown("---")
st.caption("Ứng dụng được xây dựng bởi Gemini AI (Sử dụng Streamlit và Gemini API).")
