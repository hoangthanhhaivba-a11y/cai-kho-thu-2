col3.metric("PP (Thời gian hoàn vốn)", f"{pp:.2f} năm" if isinstance(pp, float) else pp)
        col4.metric("DPP (Hoàn vốn có chiết khấu)", f"{dpp:.2f} năm" if isinstance(dpp, float) else dpp)

        # --- Chức năng 5: Yêu cầu AI Phân tích ---
        st.markdown("---")
        st.subheader("5. Phân tích Hiệu quả Dự án (AI)")
        
        if st.button("Yêu cầu AI Phân tích Chỉ số 🧠"):
            if api_key:
                with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                    ai_result = get_ai_evaluation(metrics_data, wacc, api_key)
                    st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                    st.info(ai_result)
            else:
                 st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng kiểm tra cấu hình Secrets.")

    except Exception as e:
        st.error(f"Có lỗi xảy ra khi tính toán chỉ số: {e}. Vui lòng kiểm tra các thông số đầu vào.")

else:
    st.info("Vui lòng tải lên file Word và nhấn nút 'Trích xuất Dữ liệu Tài chính bằng AI' để bắt đầu.")
