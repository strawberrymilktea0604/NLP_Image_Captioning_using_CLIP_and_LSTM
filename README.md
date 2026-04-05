# 🖼️ CLIP-LSTM Image Captioning
![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![OpenAI CLIP](https://img.shields.io/badge/OpenAI--CLIP-412991) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white) ![License](https://img.shields.io/badge/License-Academic-green)

Dự án nghiên cứu & thực nghiệm Deep Learning cho bài toán **Image Captioning** (sinh mô tả văn bản từ hình ảnh). Hệ thống kết hợp sức mạnh trích xuất đặc trưng đa phương thức của **CLIP (ViT-B/32)** với khả năng mô hình hóa chuỗi của **LSTM tích hợp Attention**, tạo thành pipeline End-to-End minh bạch, dễ tái tạo và phù hợp cho mục đích học thuật/nghiên cứu.

---

## ✨ Tính năng nổi bật
- 🔄 **Pipeline End-to-End:** Toàn bộ quy trình từ tiền xử lý dữ liệu, trích xuất đặc trưng, huấn luyện đến đánh giá được đóng gói trong một Jupyter Notebook duy nhất.
- ⚡ **CLIP Feature Caching:** Tiền tính toán và lưu đặc trưng ảnh vào RAM, giảm ~70-80% thời gian xử lý so với việc trích xuất lại mỗi epoch.
- 🎯 **Attention Mechanism:** Cho phép Decoder tập trung động vào các vùng đặc trưng quan trọng của ảnh tại từng bước sinh từ, cải thiện độ chính xác ngữ nghĩa.
- 🔍 **Beam Search Decoding:** Thay thế Greedy Search bằng tìm kiếm chùm (`beam_size=5`) kết hợp **Length Penalty** & **Repetition Penalty** để hạn chế câu cụt/lặp từ.
- 🛑 **Early Stopping & Checkpointing:** Tự động dừng training khi validation loss bão hòa (`patience=2`), lưu lại `best_model.pth` và `vocab.pkl` để phục hồi hoặc suy luận sau này.
- 📊 **Đánh giá Đa chiều:** Tính toán các chỉ số BLEU (1-4) chuẩn công nghiệp và trực quan hóa so sánh Ground Truth vs Prediction.

---

## 🧠 Kiến trúc & Công nghệ

### 🛠️ Công nghệ cốt lõi
| Thành phần | Công nghệ / Thư viện | Vai trò |
|:---|:---|:---|
| **Vision Encoder** | `OpenAI CLIP (ViT-B/32)` | Trích xuất đặc trưng ảnh đa phương thức, đóng băng trọng số (Transfer Learning) |
| **Language Decoder** | `PyTorch LSTM` | Sinh chuỗi từ tuần tự, mô hình hóa phụ thuộc ngắn-trung hạn |
| **Cơ chế tập trung** | `Additive Attention` | Gán trọng số động cho các vùng ảnh tại mỗi bước thời gian |
| **Tối ưu hóa** | `Adam`, `CrossEntropyLoss` | Huấn luyện ổn định, áp dụng Teacher Forcing & ignore `<PAD>` |
| **Xử lý ngôn ngữ** | `NLTK`, `Pandas`, `NumPy` | Tokenize, xây dựng Vocabulary, quản lý metadata |

### 🏗️ Luồng kiến trúc (Architecture Flow)
Mô hình tuân theo thiết kế **Encoder-Decoder** kinh điển cho tác vụ Vision-Language:
1. **Encoder (CLIP):** Nhận ảnh đầu vào → Tiền xử lý (Resize, Normalize) → Trích xuất vector đặc trưng cố định (512D).
2. **Decoder (LSTM + Attention):** Nhận token `<SOS>` → Tại mỗi bước $t$, Attention tính trọng số giữa Hidden State của LSTM và đặc trưng CLIP → LSTM dự đoán xác suất từ tiếp theo.
3. **Inference:** Sử dụng Beam Search để duyệt cây xác suất, áp dụng penalty → Trả về chuỗi caption tối ưu.

![Sơ đồ quy trình hoạt động của mô hình CLIP kết hợp cơ chế học đa phương thức và ứng dụng cho tác vụ sinh mô tả ảnh](docs/ảnh/imagecaptioning.png)
![Sơ đồ kiến trúc mô hình CLIP (Contrastive Language–Image Pre-training) với Image Encoder và Text Encoder](docs/ảnh/CLIP.png)
![Sơ đồ cơ chế Attention trong mô hình Sequence-to-Sequence, minh họa cách Decoder tập trung vào các vùng đặc trưng của Encoder](docs/ảnh/attention.png)

---

## 📁 Cấu trúc thư mục
```text
📦 NLP_Image_Captioning_using_CLIP_and_LSTM/
├── 📄 clip_lstm_test.ipynb      # 🚀 Entry Point: Chứa toàn bộ pipeline huấn luyện, đánh giá & demo
├── 📁 docs/                     # 📚 Tài liệu học thuật & Tài nguyên minh họa
│   ├── 📄 BaoCaoBTLFinal.pdf    # 📘 Báo cáo chi tiết (Lý thuyết, Thực nghiệm, Phân tích lỗi)
│   └── 📁 ảnh/                  # 🖼️ Sơ đồ kiến trúc, biểu đồ loss & kết quả sinh caption mẫu
└── 📄 README.md                 # 📖 Tài liệu hướng dẫn sử dụng & Cấu trúc dự án
```
> ⚠️ **Lưu ý:** Thư mục `data/` (chứa ảnh Flickr8k & `captions.txt`) không được commit do dung lượng lớn. Vui lòng tải dataset theo hướng dẫn bên dưới.

---

## ⚙️ Hướng dẫn Cài đặt & Sử dụng

### 1️⃣ Chuẩn bị môi trường
Khuyến nghị sử dụng **Python 3.9+** trên **Google Colab** hoặc Jupyter Lab/Notebook.
```bash
# Cài đặt các thư viện phụ thuộc
pip install torch torchvision openai-clip pandas numpy matplotlib nltk tqdm pillow
```

### 2️⃣ Chuẩn bị dữ liệu
Tạo thư mục `data/` tại root project với cấu trúc:
```text
data/
├── images/          # Chứa 8,000 ảnh .jpg của tập Flickr8k
└── captions.txt     # File metadata (Định dạng: <tên_ảnh>#<stt> \t <caption>)
```
Cập nhật đường dẫn `DATA_DIR`, `IMAGE_DIR`, `CAPTIONS_FILE` trong các cell cấu hình đầu tiên của notebook.

### 3️⃣ Chạy Pipeline
Mở file `clip_lstm_test.ipynb` bằng Jupyter hoặc VS Code, chạy tuần tự các cell từ trên xuống:
1. **Tiền xử lý & Caching:** Xây dựng Vocabulary, token hóa caption, tiền trích xuất đặc trưng CLIP.
2. **Huấn luyện:** Khởi tạo mô hình, chạy training loop (tối đa 15 epochs, `lr=3e-4`, `batch_size=32`).
3. **Đánh giá & Demo:** Load checkpoint tốt nhất, sinh caption trên tập test bằng Beam Search, tính BLEU và hiển thị trực quan.

---

## 📊 Kết quả & Đánh giá

### 📈 Quá trình huấn luyện
- **Early Stopping** kích hoạt tại **Epoch 11**.
- Mô hình đạt **Validation Loss thấp nhất: `3.1911`** tại Epoch 9.
- Đường cong Loss hội tụ ổn định, không xuất hiện hiện tượng Overfitting nghiêm trọng.

![Biểu đồ hàm mất mát (Loss) trong quá trình huấn luyện mô hình, phản ánh xu hướng giảm đều của training và validation loss qua các epoch](docs/ảnh/loss_plot.png)

### 📊 Chỉ số định lượng (BLEU Metrics)
Đánh giá trên tập Test sử dụng `nltk.corpus_bleu`:
| Chỉ số | Giá trị | Nhận xét |
|:---:|:---:|:---|
| **BLEU-1** | `0.2706` | Bắt đúng từ khóa cơ bản (chủ ngữ, động từ) |
| **BLEU-2** | `0.0850` | Bắt được một số cụm 2 từ phổ biến |
| **BLEU-3** | `0.0382` | Giảm mạnh, phản ánh hạn chế về cấu trúc câu |
| **BLEU-4** | `0.0201` | Thấp, cho thấy khoảng cách lớn về ngữ pháp và ngữ cảnh dài so với Ground Truth |

### 🖼️ Kết quả định tính (Qualitative Demo)
![Kết quả suy luận mẫu thứ nhất: So sánh ảnh đầu vào, caption thực tế và caption do mô hình sinh ra](docs/ảnh/result.png)
![Kết quả suy luận mẫu thứ hai: Minh họa thêm các trường hợp dự đoán của mô hình trên tập kiểm thử](docs/ảnh/result2.png)

---

## 🚧 Hạn chế & Hướng phát triển

### ⚠️ Hạn chế hiện tại
1. **Bias dữ liệu:** Caption có xu hướng chung chung, lặp lại các mẫu phổ biến trong tập Flickr8k, dễ sai lệch với vật thể/hành động cụ thể.
2. **Xử lý ngữ cảnh dài:** Kiến trúc LSTM gặp khó khăn trong việc nắm bắt phụ thuộc xa và quan hệ không gian phức tạp giữa nhiều vật thể.
3. **Chỉ số BLEU-4 thấp:** Phản ánh giới hạn của mô hình trong việc sinh câu dài, tự nhiên và đúng ngữ pháp.
4. **Thiếu file quản lý môi trường:** Hiện tại chưa có `requirements.txt` hoặc `Dockerfile`, dễ gây xung đột phiên bản khi triển khai trên máy mới.

### 🚀 Hướng phát triển (Future Work)
- 🔹 **Nâng cấp Decoder:** Thay thế LSTM bằng **Transformer Decoder** hoặc tích hợp Pre-trained LLM (ví dụ: GPT-2/Phi-3) để cải thiện khả năng sinh ngôn ngữ tự nhiên.
- 🔹 **Multi-Head Attention:** Chuyển từ Additive Attention sang Self/Multi-Head Attention để mô hình hóa tốt hơn quan hệ không gian giữa các vùng ảnh.
- 🔹 **Mở rộng Dataset:** Chuyển sang thử nghiệm trên **MS-COCO** hoặc **Conceptual Captions** để tăng độ đa dạng ngữ cảnh.
- 🔹 **Kỹ thuật Fine-tuning:** Thử nghiệm Unfreeze một phần CLIP Vision Encoder hoặc áp dụng LoRA để thích ứng đặc trưng tốt hơn với tập dữ liệu đích.

---

## 👥 Nhóm tác giả & Báo cáo
Đồ án được thực hiện bởi sinh viên **Lớp 67CS1**, Khoa Công nghệ Thông tin, Trường Đại học Xây dựng Hà Nội.

| Thành viên | MSSV |
|:---|:---|
| Nguyễn Hải Cường | 0174067 |
| Lã Minh Khánh | 4004267 |
| Trịnh Quỳnh Anh | 0279367 |
| Phạm Hồng Thái | 0127067 |

👨‍🏫 **Giảng viên hướng dẫn:** ThS. Nguyễn Đình Quý  
📅 **Hoàn thành:** 14/05/2025  
📄 **Báo cáo chi tiết:** Xem tại [`docs/BaoCaoBTLFinal.pdf`](docs/BaoCaoBTLFinal.pdf)

> 📜 *Tài liệu và mã nguồn được chia sẻ phục vụ mục đích học thuật & nghiên cứu. Vui lòng ghi nguồn khi tham khảo hoặc phát triển tiếp.*
