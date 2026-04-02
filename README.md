# 🖼️ NLP Image Captioning sử dụng CLIP và LSTM

## 👥 Thông tin chung
| Hạng mục | Thông tin |
|----------|-----------|
| **📚 Tên đề tài** | NLP Image Captioning using CLIP and LSTM |
| **👨‍🏫 Giáo viên hướng dẫn** | ThS. Nguyễn Đình Quý |
| **👥 Thành viên thực hiện** | Nguyễn Hải Cường - 0174067 - 67CS1<br>Lã Minh Khánh - 4004267 - 67CS1<br>Trịnh Quỳnh Anh - 0279367 - 67CS1<br>Phạm Hồng Thái - 0127067 - 67CS1 |

---

## 🎯 1. Tổng quan dự án
Dự án nghiên cứu và triển khai hệ thống **Tự động mô tả hình ảnh (Image Captioning)**, bài toán cốt lõi trong giao thoa giữa Xử lý Ngôn ngữ Tự nhiên (NLP) và Thị giác Máy tính (CV). Hệ thống xây dựng pipeline hoàn chỉnh từ tiền xử lý dữ liệu, trích xuất đặc trưng, huấn luyện mô hình đến đánh giá định lượng và trực quan hóa, sử dụng tập dữ liệu **Flickr8k**. Mục tiêu chính là tạo ra một baseline ổn định, dễ mở rộng, có khả năng sinh ra các câu mô tả tự nhiên, chính xác và đúng ngữ cảnh hình ảnh.

---

## 📚 2. Cơ sở lý thuyết
- **Vision Encoder (CLIP ViT-B/32):** Sử dụng mô hình đa phương thức Contrastive Language-Image Pre-training (CLIP) đã được tiền huấn luyện. CLIP ánh xạ ảnh và văn bản vào cùng một không gian embedding, giúp trích xuất vector đặc trưng 512-dim mang ngữ nghĩa cao. Trong dự án, backbone Vision được đóng băng (frozen) để tiết kiệm tài nguyên và duy trì tính tổng quát.
- **Language Decoder (LSTM + Attention):** Bộ giải mã tuần tự xử lý chuỗi token theo từng bước thời gian, duy trì bộ nhớ dài hạn. Cơ chế **Attention** được tích hợp giúp mô hình tính trọng số động cho đặc trưng hình ảnh tại mỗi bước sinh từ, cho phép "tập trung" chọn lọc vào các vùng thông tin liên quan nhất khi dự đoán từ tiếp theo.
- **Chiến lược tối ưu & Suy luận:** 
  - Hàm mất mát `CrossEntropyLoss` (ignore `<PAD>`), tối ưu bằng `Adam` (LR: 3e-4).
  - **Beam Search** (`beam_size=5`) kết hợp Length Penalty & Repetition Penalty trong giai đoạn inference để hạn chế lặp từ và cân bằng độ dài câu.
  - **Early Stopping** (`patience=2`) ngăn chặn overfitting và lưu checkpoint tốt nhất tự động.

---

## 🏗️ 3. Cấu trúc thiết kế hệ thống
Hệ thống được thiết kế theo kiến trúc **Encoder-Decoder** cải tiến, tối ưu hiệu năng thông qua cơ chế Cache đặc trưng trước khi huấn luyện:

```
Ảnh đầu vào → CLIP Encoder (Cache 1 lần → 512-dim)
                              ↓
                  State Mapping Layer (h0, c0)
                              ↓
Token hóa (<SOS>, từ, <EOS>) → Embedding (256-d) → LSTM + Attention
                              ↓
                    Linear Projection → Softmax → Từ tiếp theo
```

**Luồng thực thi chính:**
1. `Vocabulary` & `Flickr8kDataset`: Token hóa caption bằng NLTK, xây dựng bộ từ vựng (`freq_threshold=5`), padding đồng nhất.
2. `cache_clip_features`: Duyệt tập Train/Val, trích xuất và lưu đặc trưng CLIP vào RAM/Tensor, giảm thời gian tính toán mỗi epoch.
3. `train_epoch` & `validate`: Tính loss, cập nhật trọng số, giám sát bằng validation loss, tự động lưu `best_model.pth`.
4. `generate_caption_beam_search` & `calculate_bleu_scores`: Sinh chuỗi, đánh giá chỉ số BLEU-1 đến BLEU-4 và visualization.

![Cơ chế Attention](docs/ảnh/attention.png)

---

## 📊 4. Kết quả và nhận xét

### 🔹 Thông số huấn luyện
| Thành phần | Giá trị / Cấu hình |
|------------|-------------------|
| Tập dữ liệu | Flickr8k (Split 80% Train / 10% Val / 10% Test) |
| Batch Size | 32 |
| Learning Rate | 3e-4 (Adam Optimizer) |
| Max Epochs | 15 |
| Vocab Size | ~3005 |
| Beam Size | 5 |

### 🔹 Kết quả đánh giá
- **Chỉ số BLEU-4 đạt `0.0201`**: Cho thấy mô hình baseline đã nắm bắt được cú pháp cơ bản và cấu trúc câu đơn giản. Tuy nhiên, khả năng liên kết ngữ nghĩa dài hạn giữa đối tượng trong ảnh và hành động/mối quan hệ vẫn còn hạn chế.
- **Hiệu năng pipeline:** Cơ chế cache đặc trưng giúp quá trình huấn luyện nhanh, ổn định. Code được module hóa rõ ràng, dễ bảo trì và thử nghiệm các biến thể mô hình.

![Sự biến thiên của giá trị loss trên tập huấn luyện và validation qua các epoch](docs/ảnh/loss_plot.png)
![Kết quả huấn luyện mô hình](docs/ảnh/result.png)
![Kết quả tạo chú thích ảnh](docs/ảnh/result2.png)

### 🔹 Nhận xét
Kết quả BLEU-4 thấp phản ánh đúng đặc điểm của baseline CLIP+LSTM khi backbone Vision bị đóng băng hoàn toàn. Dù vậy, kiến trúc này cung cấp nền tảng vững chắc, luồng dữ liệu mạch lạc và đầy đủ các thành phần cần thiết cho một hệ thống Image Captioning thực tiễn.

---

## 🚀 5. Hướng cải thiện
Dựa trên phân tích mã nguồn và kết quả thực nghiệm, các hướng phát triển ưu tiên bao gồm:
1. **Fine-tune CLIP:** Mở khóa một phần trọng số CLIP với Learning Rate thấp (~1e-5) để đặc trưng hình ảnh thích nghi tốt hơn với tập Flickr8k, cải thiện đáng kể BLEU.
2. **Nâng cấp Decoder:** Chuyển từ LSTM sang **Transformer Decoder** hoặc mô hình sinh kiểu GPT để xử lý phụ thuộc ngữ nghĩa dài hạn và ngữ cảnh tốt hơn.
3. **Teacher Forcing Decay:** Tự động giảm dần tỉ lệ Teacher Forcing theo epoch, giúp mô hình tự tin sinh từ độc lập và giảm lỗi tích lũy (exposure bias).
4. **Tăng cường dữ liệu (Data Augmentation):** Áp dụng `RandomResizedCrop`, `ColorJitter`, `HorizontalFlip` để nâng cao khả năng tổng quát hóa.
5. **Monitoring & Experiment Tracking:** Tích hợp `WandB` hoặc `TensorBoard` để theo dõi real-time: loss, gradient norm, BLEU scores và hình ảnh sinh mẫu trong quá trình train.

---

## 📦 Cài đặt & Hướng dẫn sử dụng
**Môi trường yêu cầu:** Python 3.8+, PyTorch, Torchvision, OpenAI CLIP, NLTK, tqdm, matplotlib.
```bash
pip install torch torchvision openai-clip numpy matplotlib tqdm nltk
```
**Thực thi toàn bộ pipeline:** Mã nguồn được đóng gói trong `NLP_Image_Captioning_using_CLIP_and_LSTM.ipynb`. Chỉ cần chạy tuần tự các cell để thực hiện tiền xử lý, huấn luyện, lưu checkpoint và sinh caption mẫu trên tập ảnh bất kỳ.

---
📚 *Tài liệu tham khảo:* Radford et al. (2021) - *Learning Transferable Visual Models From Natural Language Supervision* (CLIP) | Hochreiter & Schmidhuber (1997) - *Long Short-Term Memory* | Flickr8k Dataset.
