# 🖼️ CLIP + LSTM Image Captioning
![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![OpenAI CLIP](https://img.shields.io/badge/OpenAI--CLIP-412991) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white) ![License](https://img.shields.io/badge/License-Academic-green)

Dự án nghiên cứu/đồ án môn **Xử lý ngôn ngữ tự nhiên** về bài toán **tạo chú thích ảnh tự động (Image Captioning)**.  
Hệ thống kết hợp **CLIP (ViT-B/32)** để trích xuất đặc trưng ảnh và **LSTM + Attention** để sinh caption theo chuỗi.

![Pipeline Image Captioning](docs/ảnh/imagecaptioning.png)
![CLIP Architecture](docs/ảnh/CLIP.png)
![Attention Mechanism](docs/ảnh/attention.png)

---

## ✨ Tổng quan nhanh

- **Encoder:** CLIP `ViT-B/32`
- **Decoder:** LSTM nhiều lớp
- **Cơ chế hỗ trợ:** Attention
- **Giải mã caption:** Beam Search
- **Tiền xử lý văn bản:** NLTK
- **Đánh giá:** BLEU-1 đến BLEU-4

Dự án được triển khai chủ yếu trong notebook `clip_lstm_test.ipynb`, đi kèm báo cáo chi tiết trong `docs/BaoCaoBTLFinal.pdf`.

---

## 🔧 Tính năng chính

- Trích xuất đặc trưng ảnh bằng CLIP và tái sử dụng đặc trưng đã cache để tăng tốc huấn luyện.
- Xây dựng vocabulary với các token đặc biệt: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`.
- Huấn luyện mô hình sinh caption bằng **Cross-Entropy Loss**.
- Dùng **Attention** để mô hình tập trung vào đặc trưng quan trọng khi sinh từng từ.
- Dùng **Beam Search** để giảm câu quá ngắn, lặp từ hoặc lựa chọn tham lam.
- Hỗ trợ **early stopping** và lưu checkpoint.

---

## 🧠 Kiến trúc mô hình

Luồng xử lý chính:

1. Ảnh đầu vào được resize/normalize bằng `preprocess` của CLIP.
2. CLIP mã hóa ảnh thành embedding đặc trưng.
3. Vector ảnh được đưa vào LSTM để khởi tạo trạng thái ẩn.
4. Decoder sinh từng từ của caption theo từng bước thời gian.
5. Attention và Beam Search được dùng trong quá trình suy luận để cải thiện chất lượng caption.

---

## 📁 Cấu trúc thư mục

```text
NLP_Image_Captioning_using_CLIP_and_LSTM-main/
├── clip_lstm_test.ipynb
├── README.md
└── docs/
    ├── BaoCaoBTLFinal.pdf
    └── ảnh/
        ├── CLIP.png
        ├── CNN.png
        ├── RNN.png
        ├── attention.png
        ├── beamsearch.png
        ├── imagecaptioning.png
        ├── loss_plot.png
        ├── lstm.png
        ├── result.png
        ├── result2.png
        └── transfer-learning.png
```

> Lưu ý: dữ liệu gốc như `dataset/Images/` và `dataset/captions.txt` không nằm trong repo.

---

## 🗂️ Dữ liệu sử dụng

Dự án sử dụng bộ dữ liệu **Flickr8k**:

- khoảng **8.000 ảnh**
- mỗi ảnh có nhiều caption do con người viết

Trong notebook, dữ liệu được đọc từ:

- `dataset/Images/`
- `dataset/captions.txt`

### Tiền xử lý dữ liệu

- Đọc và gom nhóm caption theo từng ảnh
- Tokenize caption bằng NLTK
- Loại từ hiếm theo ngưỡng tần suất
- Thêm token đặc biệt `<SOS>` và `<EOS>`
- Padding caption theo batch
- Chia dữ liệu theo tỉ lệ:
  - **Train:** 80%
  - **Validation:** 10%
  - **Test:** 10%

---

## 🧪 Cấu hình huấn luyện

Các siêu tham số chính được dùng trong notebook:

- `embed_size = 256`
- `hidden_size = 512`
- `batch_size = 32`
- `learning_rate = 3e-4`
- `num_epochs = 15`
- `early_stopping_patience = 2`
- `attention = True`
- `beam_size = 5`
- `use_clip_cache = True`

---

## 🚀 Cách chạy

### 1) Cài đặt thư viện

```bash
pip install torch torchvision openai-clip pandas numpy matplotlib nltk tqdm pillow
```

### 2) Chuẩn bị dữ liệu

Tạo thư mục `dataset/` ở thư mục gốc:

```text
dataset/
├── Images/
└── captions.txt
```

Sau đó kiểm tra lại các đường dẫn trong notebook:

```python
CAPTIONS_FILE = 'dataset/captions.txt'
IMAGES_FOLDER = 'dataset/Images'
CHECKPOINT_DIR = 'checkpoints'
```

### 3) Chạy notebook

Mở `clip_lstm_test.ipynb` và chạy lần lượt các cell:

1. import thư viện và tải CLIP
2. load dữ liệu Flickr8k
3. xây vocabulary
4. cache CLIP features
5. huấn luyện mô hình
6. đánh giá BLEU và sinh caption mẫu

---

## 📊 Kết quả thực nghiệm

Theo báo cáo trong `docs/BaoCaoBTLFinal.pdf`:

- **Early stopping** kích hoạt ở **epoch 11**
- **Validation loss tốt nhất:** `3.1911` tại **epoch 9**

### BLEU score

| Chỉ số | Giá trị |
|---|---:|
| BLEU-1 | 0.2706 |
| BLEU-2 | 0.0850 |
| BLEU-3 | 0.0382 |
| BLEU-4 | 0.0201 |

![Loss plot](docs/ảnh/loss_plot.png)
![Result 1](docs/ảnh/result.png)
![Result 2](docs/ảnh/result2.png)

### Nhận xét

- Mô hình học được xu hướng giảm loss khá ổn định.
- Caption sinh ra vẫn còn thiên về các mẫu phổ biến.
- BLEU-4 còn thấp, cho thấy mô hình chưa nắm bắt tốt cấu trúc ngôn ngữ dài và ngữ cảnh phức tạp.

---

## ⚠️ Hạn chế

- LSTM còn hạn chế khi học phụ thuộc dài.
- Dữ liệu Flickr8k tương đối nhỏ cho bài toán sinh caption.
- Một số caption sinh ra bị lặp hoặc mô tả chưa sát ảnh.
- Repo hiện chưa có `requirements.txt` hoặc `Dockerfile`.

---

## 🔮 Hướng phát triển

- Thử **Transformer Decoder** thay cho LSTM.
- Tăng cường **Multi-Head Attention**.
- Mở rộng sang bộ dữ liệu lớn hơn như **MS COCO** hoặc **Conceptual Captions**.
- Thử fine-tune thêm một phần encoder CLIP hoặc dùng LoRA/adapters.

---

## 👥 Thành viên nhóm

Đồ án được thực hiện bởi sinh viên **Lớp 67CS1**, Khoa Công nghệ Thông tin, Trường Đại học Xây dựng Hà Nội.

| Thành viên | MSSV |
|---|---:|
| Nguyễn Hải Cường | 0174067 |
| Lã Minh Khánh | 4004267 |
| Trịnh Quỳnh Anh | 0279367 |
| Phạm Hồng Thái | 0127067 |

**Giảng viên hướng dẫn:** ThS. Nguyễn Đình Quý  
**Ngày hoàn thành:** 14/05/2025

---

## 📄 Tài liệu tham khảo trong repo

- `docs/BaoCaoBTLFinal.pdf`
- `clip_lstm_test.ipynb`
- các hình minh họa trong `docs/ảnh/`

---

## 📌 Ghi chú

Dự án được chia sẻ phục vụ mục đích học tập và nghiên cứu. Khi tham khảo hoặc phát triển tiếp, vui lòng ghi nguồn phù hợp.
