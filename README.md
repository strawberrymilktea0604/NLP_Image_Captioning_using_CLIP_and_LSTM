# NLP Image Captioning using CLIP and LSTM

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-7b61ff)
![LSTM](https://img.shields.io/badge/LSTM-Attention-00a67d)
![NLP](https://img.shields.io/badge/NLP-Image%20Captioning-ff8c42)

> Đồ án nhỏ của nhóm mình về bài toán **tạo chú thích ảnh tự động**.  
> Mục tiêu là lấy một bức ảnh đầu vào, trích xuất đặc trưng bằng **CLIP**, rồi để **LSTM + Attention** sinh ra một câu mô tả tương đối tự nhiên.

<p align="center">
  <img src="docs/ảnh/imagecaptioning.png" alt="CLIP overview" width="900">
</p>

## Tóm tắt ngắn

Project này là một thử nghiệm kết hợp giữa **Computer Vision** và **NLP**.  
Phần encoder dùng **CLIP (ViT-B/32)** để lấy đặc trưng hình ảnh, còn phần decoder dùng **LSTM** để sinh caption từng từ một. Trong quá trình chạy thử, nhóm có dùng thêm **Attention** và **Beam Search** để caption đỡ bị quá ngắn hoặc quá “cứng”.

Nói thật là kết quả chưa phải kiểu “rất xịn”, nhưng mô hình vẫn học được mối liên hệ cơ bản giữa ảnh và câu mô tả. Phần này nhóm mình giữ lại đầy đủ trong báo cáo để nhìn ra cả điểm mạnh lẫn điểm chưa ổn của mô hình.

## Người thực hiện

- **Nguyễn Hải Cường** – 0174067-67CS1
- **Lã Minh Khánh** – 4004267-67CS1
- **Trịnh Quỳnh Anh** – 0279367-67CS1
- **Phạm Hồng Thái** – 0127067-67CS1

**Giảng viên hướng dẫn:** ThS. **Nguyễn Đình Quý**  
**Môn học:** Xử lý ngôn ngữ tự nhiên

---

## 1. Ý tưởng chính

Bài toán image captioning có thể hiểu đơn giản là:

- Input: một bức ảnh
- Output: một câu mô tả nội dung ảnh

Trong đồ án này, nhóm mình chọn hướng đi khá phổ biến:

1. Dùng **CLIP** để lấy vector đặc trưng của ảnh.
2. Đưa vector đó vào **LSTM** để sinh caption.
3. Dùng **Attention** để mô hình chú ý tốt hơn tới các phần quan trọng của ảnh.
4. Dùng **Beam Search** khi suy luận để câu sinh ra đỡ bị “lụm cụm”.

<p align="center">
  <img src="docs/ảnh/CLIP.png" alt="CLIP architecture" width="900">
</p>

<p align="center">
  <img src="docs/ảnh/attention.png" alt="Attention decoder" width="760">
</p>

---

## 2. Mô hình và cách làm

### Encoder
- Dùng **CLIP ViT-B/32**
- Lấy embedding ảnh làm đầu vào cho decoder
- Có cache feature để tăng tốc lúc train

### Decoder
- **Embedding layer**
- **LSTM**
- **Attention**
- **Linear layer** để dự đoán token tiếp theo

### Sinh caption
- Khi train: dùng caption thật để dạy mô hình dự đoán từ kế tiếp
- Khi test: mô hình tự sinh từng từ cho đến khi gặp `<EOS>`
- Có dùng **Beam Search** để chọn caption hợp lý hơn thay vì greedy decoding

<p align="center">
  <img src="docs/ảnh/beamsearch.png" alt="Beam search" width="760">
</p>

---

## 3. Kết quả thực nghiệm

### Tham số chính
- `embed_size = 256`
- `hidden_size = 512`
- `batch_size = 32`
- `learning_rate = 3e-4`
- `num_epochs = 15`
- `attention = True`
- `beam_size = 5`
- `early_stopping_patience = 2`

### Kết quả huấn luyện
Theo báo cáo trong thư mục `docs`, mô hình dừng sớm ở **epoch 11** vì validation loss không còn cải thiện trong 2 epoch liên tiếp.  
Best validation loss đạt khoảng **3.1911** ở **epoch 9**.

<p align="center">
  <img src="docs/ảnh/loss_plot.png" alt="Training loss" width="760">
</p>

### BLEU score
| Metric | Value |
|---|---:|
| BLEU-1 | 0.2706 |
| BLEU-2 | 0.0850 |
| BLEU-3 | 0.0382 |
| BLEU-4 | 0.0201 |

Nhìn chung, caption sinh ra vẫn còn khá chung chung và đôi lúc bị lệch nội dung ảnh. Nhưng với một đồ án học phần, phần này giúp nhóm mình thấy khá rõ vấn đề của mô hình và chỗ cần cải thiện nếu làm tiếp.

### Một vài kết quả sinh caption
<p align="center">
  <img src="docs/ảnh/result2.png" alt="Generated captions" width="900">
</p>

---

## 4. Cấu trúc thư mục

```text
NLP_Image_Captioning_using_CLIP_and_LSTM-main/
├── clip_lstm_test.ipynb
├── docs/
│   ├── BaoCaoBTLFinal.pdf
│   └── ảnh/
│       ├── CLIP.png
│       ├── attention.png
│       ├── beamsearch.png
│       ├── loss_plot.png
│       └── result2.png
└── README.md
```

> Trong notebook, phần đọc dữ liệu đang trỏ tới `dataset/captions.txt` và `dataset/Images/`.  
> Nghĩa là nếu chạy lại từ đầu thì cần chuẩn bị đúng cấu trúc dữ liệu đó trước.

---

## 5. Cách chạy lại

### Cài thư viện
Dự án dùng notebook nên cách đơn giản nhất là tạo môi trường Python rồi cài các gói cần thiết:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib pillow tqdm nltk
pip install git+https://github.com/openai/CLIP.git
```

### Tải dữ liệu NLTK
```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
```

### Chạy notebook
1. Đưa dữ liệu vào đúng thư mục `dataset/`
2. Mở `clip_lstm_test.ipynb`
3. Chạy lần lượt các cell từ trên xuống
4. Quan sát loss, caption sinh ra và phần đánh giá BLEU

---

## 6. Tech stack

- Python
- PyTorch
- OpenAI CLIP
- LSTM
- Attention
- Beam Search
- NLTK
- NumPy
- Pandas
- Matplotlib
- Pillow
- tqdm
- Jupyter Notebook

---

## 7. Tài liệu tham khảo

- Báo cáo PDF: `docs/BaoCaoBTLFinal.pdf`
- Hình minh họa CLIP, Attention, Beam Search và kết quả sinh caption: `docs/ảnh/`

---

## 8. Lời cảm ơn

Nhóm mình xin gửi lời cảm ơn đến **ThS. Nguyễn Đình Quý** đã hướng dẫn môn học và góp ý trong quá trình làm đề tài.  
Cảm ơn các thành viên trong nhóm đã cùng thử nghiệm, chỉnh sửa notebook và hoàn thiện báo cáo.
