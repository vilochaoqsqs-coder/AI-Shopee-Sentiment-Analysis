# 🛍️ Phân tích cảm xúc từ đánh giá khách hàng trên nền tảng Shopee sử dụng kỹ thuật học máy và mô hình Transformer (PhoBERT)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-PhoBERT-yellow?logo=huggingface&logoColor=white)
![NLP](https://img.shields.io/badge/Task-Vietnamese%20Sentiment%20Analysis-success)
![Course](https://img.shields.io/badge/H%E1%BB%8Dc%20ph%E1%BA%A7n-Tr%C3%AD%20Tu%E1%BB%87%20Nh%C3%A2n%20T%E1%BA%A1o%20(N02)-orange)

## 📌 1. Giới thiệu đề tài
Thương mại điện tử tại Việt Nam đang phát triển mạnh mẽ, trong đó **Shopee** là một trong những nền tảng hàng đầu với lượng người dùng và đánh giá sản phẩm rất lớn. Việc phân tích cảm xúc của khách hàng từ các đánh giá có ý nghĩa quan trọng giúp doanh nghiệp nắm bắt phản hồi thị trường và cải thiện chất lượng sản phẩm, dịch vụ.

Đề tài tập trung xây dựng hệ thống tự động phân loại cảm xúc đánh giá tiếng Việt (bao gồm cả văn bản không dấu, teencode đặc thù thương mại điện tử) ở mức độ câu thành **3 lớp cảm xúc (`Positive` / `Neutral` / `Negative`)** bằng cách fine-tune mô hình ngôn ngữ tiền huấn luyện **PhoBERT**.

---

## 👥 2. Nhóm sinh viên thực hiện (Lớp N02)

| Mã SV | Họ và tên | Nhiệm vụ phụ trách |
| :---: | :--- | :--- |
| **BIT250128** | **Vi Lộc Hào** | Thu thập, tiền xử lý dữ liệu, tạo nhãn 3 lớp (`label_3class`), kiểm tra chất lượng dữ liệu và chuẩn bị thuyết trình |
| **BIT250060** | **Nguyễn Hải Đăng** | Tổng hợp đề tài, xây dựng pipeline tổng thể, tích hợp chương trình demo |
| **BIT250027** | **Nguyễn Văn Tài Anh** | Nghiên cứu và fine-tune mô hình PhoBERT, đánh giá hiệu suất mô hình cảm xúc |
| **BIT250219** | **Nguyễn Hoàng Bảo Long** | Xây dựng giao diện demo, trực quan hóa dữ liệu và kết quả, hỗ trợ đánh giá mô hình |

---

## 📊 3. Thông tin Bộ dữ liệu (Dataset)

* **Nguồn dữ liệu:** *Shopee Vietnamese Product Reviews Sentiment* (Kaggle), gồm **9.599 đánh giá** thu thập từ Shopee.vn trong giai đoạn 2023–2026.
* **Các trường dữ liệu chính:**
  * `id`: Mã định danh đánh giá
  * `review`: Nội dung đánh giá của khách hàng (bao gồm cả teencode, văn bản không dấu)
  * `rating`: Số sao đánh giá (từ 1 đến 5 sao)
  * `label`: Nhãn cảm xúc gốc (`positive` / `negative`)
  * `label_3class`: Nhãn cảm xúc 3 lớp (`positive` / `neutral` / `negative`) sau khi chuẩn hóa.
* **Quy tắc gán nhãn 3 lớp (`label_3class`):**
  * `rating = 3` $\rightarrow$ **Neutral**
  * Nhãn gốc `positive` (và `rating != 3`) $\rightarrow$ **Positive**
  * Còn lại $\rightarrow$ **Negative**

### Phân bố nhãn sau tiền xử lý:
| Nhãn cảm xúc (`label_3class`) | Số lượng mẫu | Tỷ lệ (%) |
| :--- | :---: | :---: |
| 🔴 **Negative** (Tiêu cực) | 4.274 | 44,5% |
| 🟢 **Positive** (Tích cực) | 3.543 | 36,9% |
| 🟡 **Neutral** (Trung tính) | 1.782 | 18,6% |
| **Tổng cộng** | **9.599** | **100%** |

---

## 🗂️ 4. Cấu trúc thư mục dự án

```text
AI-Shopee-Sentiment-Analysis/
├── data/
│   ├── raw/                                 # Dữ liệu gốc chưa qua xử lý
│   │   ├── shopee_reviews_dataset.jsonl     # Bộ dữ liệu gốc 9.599 đánh giá Shopee
│   │   └── aug_unaccented_reviews.jsonl     # Dữ liệu bổ sung văn bản không dấu
│   └── processed/                           # Dữ liệu đã làm sạch và gán nhãn 3 lớp
│       ├── shopee_reviews_cleaned_3class.csv
│       ├── train.csv                        # Tập huấn luyện (Train set)
│       ├── val.csv                          # Tập kiểm định (Validation set)
│       └── test.csv                         # Tập kiểm thử (Test set)
├── notebooks/
│   ├── docdulieu.ipynb                      # Notebook khám phá, làm sạch dữ liệu & tạo nhãn 3 lớp
│   └── phobert_finetuning.ipynb             # Notebook huấn luyện và đánh giá mô hình PhoBERT
├── src/                                     # Mã nguồn xử lý pipeline và mô hình
├── app/                                     # Mã nguồn giao diện thử nghiệm (Streamlit / Gradio)
├── models/                                  # Lưu trữ cấu hình / trọng số mô hình tốt nhất
├── docs/                                    # Báo cáo tiểu luận và slide thuyết trình
├── requirements.txt                         # Danh sách thư viện Python cần thiết
└── README.md                                # Tài liệu hướng dẫn dự án

