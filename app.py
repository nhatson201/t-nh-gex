from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Cấu hình giao diện trang web
st.set_page_config(
    page_title="CME QuikStrike Quantitative Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Ứng Dụng Phân Tích Định Lượng CME QuikStrike Options")
st.markdown(
    "Tải lên tệp HTML dữ liệu QuikStrike để trích xuất tự động: **Max Pain,"
    " Call/Put Wall, Net GEX & Gamma Flip**."
)

# 1. Khu vực tải file lên web
uploaded_file = st.file_uploader(
    "Chọn file dữ liệu HTML của QuikStrike", type=["html", "htm", "docx", "txt"]
)

if uploaded_file is not None:
  # Đọc nội dung file tải lên
  html_content = uploaded_file.getvalue().decode("utf-8", errors="ignore")

  # 2. Hàm bóc tách dữ liệu từ thẻ <area fields=...>
  @st.cache_data
  def parse_uploaded_html(content):
    soup = BeautifulSoup(content, "html.parser")
    area_tags = soup.find_all("area", attrs={"fields": True})

    if not area_tags:
      return None

    records = []
    for area in area_tags:
      fields_str = area.get("fields", "")
      if not fields_str:
        continue

      row_dict = {}
      for item in fields_str.split("~"):
        if "|" in item:
          key, value = item.split("|", 1)
          clean_val = value.replace(",", ".").strip()
          try:
            if "." in clean_val:
              row_dict[key] = float(clean_val)
            else:
              row_dict[key] = int(clean_val)
          except ValueError:
            row_dict[key] = value.strip()
      if row_dict:
        records.append(row_dict)

    df = pd.DataFrame(records)

    # Chuẩn hóa cột Strike
    if "strike" not in df.columns and "title" in df.columns:
      df["strike"] = (
          df["title"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)
      )

    # Chuẩn hóa các cột Greeks và Open Interest
    numeric_cols = ["oiCall", "oiPut", "gamma", "deltaCall", "deltaPut"]
    for col in numeric_cols:
      if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "."), errors="coerce"
        ).fillna(0)
      else:
        df[col] = 0.0

    return df.sort_values(by="strike").reset_index(drop=True)


  df_parsed = parse_uploaded_html(html_content)

  if df_parsed is not None and not df_parsed.empty:
    st.success(
        f"✅ Bóc tách dữ liệu thành công! Tìm thấy {len(df_parsed)} mức Strike."
    )

    # Nhập thông số Spot Price và Multiplier tùy chỉnh trên giao diện web
    col_input1, col_input2 = st.columns(2)
    with col_input1:
      spot_price = st.number_input(
          "Giá cơ sở hiện tại (Spot/Future):", value=3898.9, step=1.0
      )
    with col_input2:
      contract_multiplier = st.number_input(
          "Hệ số nhân hợp đồng (Multiplier):", value=100, step=10
      )

    # 3. Tính toán các chỉ số định lượng
    strikes = df_parsed["strike"].values
    call_ois = df_parsed["oiCall"].values
    put_ois = df_parsed["oiPut"].values
    gammas = df_parsed["gamma"].values

    # Call Wall & Put Wall (Mức Strike có OI lớn nhất)
    call_wall = strikes[np.argmax(call_ois)]
    put_wall = strikes[np.argmax(put_ois)]

    # Max Pain
    total_payouts = []
    for s in strikes:
      call_payout = np.maximum(0, s - strikes) * call_ois
      put_payout = np.maximum(0, strikes - s) * put_ois
      total_payouts.append(np.sum(call_payout + put_payout))
    max_pain = strikes[np.argmin(total_payouts)]

    # Net Gamma Exposure (GEX)
    df_parsed["call_gex"] = (
        gammas * call_ois * (spot_price**2) * 0.01 * contract_multiplier
    )
    df_parsed["put_gex"] = (
        gammas * put_ois * (spot_price**2) * 0.01 * contract_multiplier
    )
    net_gex = (df_parsed["call_gex"] - df_parsed["put_gex"]).sum()

    # Gamma Flip Level
    test_spots = np.linspace(spot_price * 0.95, spot_price * 1.05, 200)
    net_gex_curve = []
    for s in test_spots:
      cg = gammas * call_ois * (s**2) * 0.01 * contract_multiplier
      pg = gammas * put_ois * (s**2) * 0.01 * contract_multiplier
      net_gex_curve.append(np.sum(cg - pg))
    net_gex_curve = np.array(net_gex_curve)

    zero_crossings = np.where(np.diff(np.sign(net_gex_curve)))[0]
    flip_levels = []
    for idx in zero_crossings:
      p1, p2 = test_spots[idx], test_spots[idx + 1]
      g1, g2 = net_gex_curve[idx], net_gex_curve[idx + 1]
      flip_levels.append(p1 - g1 * (p2 - p1) / (g2 - g1))

    # 4. Hiển thị Dashboard kết quả trên Web
    st.markdown("---")
    st.subheader("🎯 KẾT QUẢ PHÂN TÍCH ĐỊNH LƯỢNG CHÍNH")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🎯 Max Pain", f"{max_pain:,.1f}")
    m2.metric("🟢 Call Wall", f"{call_wall:,.1f}")
    m3.metric("🔴 Put Wall", f"{put_wall:,.1f}")
    m4.metric("📊 Net GEX ($)", f"${net_gex:,.0f}")
    m5.metric(
        "⚡ Gamma Flip",
        f"{flip_levels[0]:,.1f}" if flip_levels else "Không có",
    )

    # 5. Trực quan hóa Biểu đồ
    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
      st.markdown("### Phân Bổ Open Interest & Max Pain")
      fig, ax = plt.subplots(figsize=(8, 5))
      x = np.arange(len(strikes))
      width = 0.35
      ax.bar(
          x - width / 2,
          call_ois,
          width,
          label="Call OI",
          color="#2ecc71",
          alpha=0.85,
      )
      ax.bar(
          x + width / 2,
          put_ois,
          width,
          label="Put OI",
          color="#e74c3c",
          alpha=0.85,
      )
      if max_pain in strikes:
        mp_idx = np.where(strikes == max_pain)[0][0]
        ax.axvline(
            x=mp_idx,
            color="#f1c40f",
            linestyle="--",
            linewidth=2,
            label=f"Max Pain ({max_pain})",
        )
      ax.set_xlabel("Strike Price")
      ax.set_ylabel("Open Interest")
      ax.set_xticks(x[:: max(1, len(x) // 10)])
      ax.set_xticklabels(
          [str(int(s)) for s in strikes[:: max(1, len(x) // 10)]], rotation=45
      )
      ax.legend()
      ax.grid(axis="y", linestyle=":", alpha=0.6)
      st.pyplot(fig)

    with col_chart2:
      st.markdown("### Đường Cong Net GEX & Gamma Flip")
      fig2, ax2 = plt.subplots(figsize=(8, 5))
      ax2.plot(
          test_spots,
          net_gex_curve / 1e6,
          label="Net GEX (M USD)",
          color="#2980b9",
          linewidth=2.5,
      )
      ax2.axhline(0, color="black", linewidth=1, alpha=0.7)
      ax2.axvline(
          x=spot_price,
          color="#e67e22",
          linestyle="--",
          linewidth=2,
          label=f"Spot ({spot_price})",
      )
      for flip in flip_levels:
        ax2.axvline(
            x=flip,
            color="#9b59b6",
            linestyle="-.",
            linewidth=2,
            label=f"Flip ({flip:,.1f})",
        )
      ax2.set_xlabel("Spot Price")
      ax2.set_ylabel("Net GEX (Million USD)")
      ax2.legend()
      ax2.grid(True, linestyle=":", alpha=0.6)
      st.pyplot(fig2)

    # Hiển thị bảng dữ liệu chi tiết
    with st.expander("📂 Xem bảng dữ liệu Options chi tiết"):
      st.dataframe(df_parsed)

  else:
    st.error(
        "[!] Không tìm thấy cấu trúc dữ liệu thẻ <area fields> hợp lệ trong"
        " file này."
    )
