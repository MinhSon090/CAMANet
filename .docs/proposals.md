# TÀI LIỆU CẢI TIẾN KIẾN TRÚC & PHƯƠNG PHÁP (PROPOSALS & EXTENSIONS)

> **Mô hình nền tảng:** CAMANet (*Class Activation Map Guided Attention Network for Radiology Report Generation* - IEEE J-BHI 2024)  
> **Mục tiêu:** Lưu trữ, phân tích và hướng dẫn hiện thực hóa các đề xuất nâng cấp mô hình.  
> **Nguyên tắc thiết kế:** 
> - **Tính độc lập (Decoupling):** Mỗi ý tưởng có thể bật/tắt qua cờ lệnh (argument flags), kiểm thử độc lập mà không làm gãy kiến trúc gốc.
> - **Khả năng mở rộng (Extensibility):** Cấu trúc tài liệu theo dạng mô-đun hóa; dễ dàng bổ sung các ý tưởng tiếp theo mà không làm xáo trộn ngữ cảnh.

---

## Danh mục đề xuất cải tiến (Index Matrix)

| ID | Tên đề xuất | Phạm vi can thiệp | Trạng thái | Độc lập | Tệp mã nguồn liên quan |
|---|---|---|---|---|---|
| **PROP-01** | Multi-Token Prediction (MTP) | Decoder & Loss Function | Proposed | Có | `modules/my_encoder_decoder.py`, `trainer.py`, `loss.py` |
| **PROP-02** | Knowledge Graph Grounded CoT | Data Pipeline & Decoder | Proposed | Có | `data_utils`, `tokenizers.py`, `trainer.py`, `cam_attn_con.py` |
| *PROP-XX* | *(Dành cho đề xuất tiếp theo)* | ... | Pending | ... | ... |

---

## PROP-01: Multi-Token Prediction (MTP)

### 1. Động lực & Mục tiêu
* **Bài toán:** Decoder chuẩn của CAMANet sinh từ theo cơ chế Next-Token Prediction (NTP) đơn lẻ. NTP dễ rơi vào bẫy "tham lam" (*greedy search*), ngắt quãng các cụm thuật ngữ y khoa cố định (*clinical collocations* như *"pleural effusion"*, *"within normal limits"*) và dễ bị chi phối bởi các từ xuất hiện phổ biến trong ca bình thường (*data bias*).
* **Mục tiêu:** Dự đoán đồng thời $K$ token tương lai ($w_t, w_{t+1}, \dots, w_{t+K}$) tại mỗi bước giải mã nhằm:
  1. Ép vector ẩn của Decoder lập kế hoạch ngữ nghĩa xa hơn (*lookahead semantic planning*).
  2. Làm giàu biểu diễn cho mô-đun VTAC: các từ quan trọng mang ngữ cảnh của cả cụm bệnh lý, giúp gióng hàng với bản đồ tổn thương $d^v$ chuẩn xác hơn.
  3. Cho phép tăng tốc độ suy luận (1.5x – 2.5x) thông qua *Speculative Decoding* khi inference.

---

### 2. Thiết kế kiến trúc & Điểm can thiệp trong CAMANet

```
                      ┌── Head 0 (Chính) ─────────► Dự đoán w_t      (Loss_0)
                      │
Hidden State h_t ────┼── MTP Module 1 (Head 1) ──► Dự đoán w_{t+1}  (Loss_1)
(từ Decoder Layer L)  │
                      └── MTP Module 2 (Head 2) ──► Dự đoán w_{t+2}  (Loss_2)
```

1. **Khối dự đoán tương lai (MTP Heads):**
   * Sau tầng Decoder cuối cùng, trạng thái ẩn $h_t \in \mathbb{R}^D$ được đưa qua $K$ nhánh độc lập:
     * **Head 0 (Gốc):** Tuyến tính chiếu ra từ vựng: $\hat{p}_t = \text{Softmax}(W_0 h_t)$.
     * **Head $k$ ($k \in [1, K]$):** Bao gồm một khối biến đổi nhỏ (1 Transformer block nhẹ hoặc LayerNorm + MLP) kết hợp với biểu diễn $h_t$ để dự đoán token $w_{t+k}$:
       $$h_t^{(k)} = \text{LayerNorm}(h_t^{(k-1)} + \text{MTPBlock}_k(h_t^{(k-1)}))$$
       $$\hat{p}_{t+k} = \text{Softmax}(W_k h_t^{(k)})$$
2. **Hàm mất mát mở rộng:**
   $$\mathcal{L}_{MTP} = \mathcal{L}_{ce}(w_t) + \sum_{k=1}^K \gamma^k \mathcal{L}_{ce}(w_{t+k})$$
   * Trong đó $\gamma \in (0, 1]$ là hệ số suy giảm trọng số theo khoảng cách tương lai (ví dụ: $\gamma = 0.8$).
   * Tổng loss của CAMANet trở thành: $\mathcal{L}_{total} = \mathcal{L}_{MTP} + \lambda \mathcal{L}_{bce} + \delta \mathcal{L}_{mse}$.

---

### 3. Kế hoạch triển khai độc lập (Step-by-Step)
* **Tệp can thiệp:**
  1. `modules/my_encoder_decoder.py`: Thêm class `MTPHead(nn.Module)` chứa các head dự đoán bổ sung; sửa `Decoder.forward()` để trả về danh sách logits tương ứng với các bước tương lai.
  2. `modules/loss.py`: Bổ sung hàm tính loss MTP với mask che padding phù hợp theo độ lệch vị trí $k$.
  3. `modules/trainer.py`: Tích hợp tính loss MTP trong vòng lặp huấn luyện; khi inference chỉ sử dụng Head 0 (hoặc bật speculative decoding).
* **Cờ cấu hình (CLI Args):**
  * `--use_mtp` (bool, mặc định `False` để tương thích ngược).
  * `--mtp_depth` (int, mặc định `2`, dự đoán 2 token kế tiếp).
  * `--mtp_decay` (float, mặc định `0.8`).

---

## PROP-02: Knowledge Graph (KG) Grounded Chain-of-Thought (CoT)

### 1. Động lực & Mục tiêu
* **Bài toán:** Các tập dữ liệu RRG (IU X-Ray, MIMIC-CXR) chỉ cung cấp cặp `(Ảnh, Báo cáo chẩn đoán)` mà hoàn toàn không có nhãn quá trình suy luận (*thinking traces*). Để mô hình tự sinh CoT không giám sát rất dễ dẫn đến ảo giác nghiêm trọng (*thought hallucination*).
* **Mục tiêu:** Xây dựng quy trình tổng hợp chuỗi suy luận có căn cứ thực tế (*Factually-Grounded CoT*) từ **Đồ thị tri thức y khoa (Medical Knowledge Graph)** kết hợp LLM chuyên gia, thu về ~10.000 mẫu chất lượng cao để "dạy" mô hình suy luận chuẩn xác trước khi viết báo cáo.

---

### 2. Thiết kế Pipeline & Điểm can thiệp trong CAMANet

```
[Báo cáo chuẩn & Nhãn] ──► [KG Subgraph Extraction] ──► [LLM Verbalizer] ──► [Strict Verification]
(IU / MIMIC)                (RadGraph / ChestX-KG)      (Đổi thành văn bản)     (Lọc 10k mẫu sạch)
                                                                                     │
                                                                                     ▼
[Ảnh X-ray] ──► [CAMANet Decoder] ──► <think> Chuỗi suy luận </think> ──► [Findings Báo cáo]
                      ▲                                │
                      └──────── VTAC gióng hàng ───────┘
```

1. **Pipeline 3 giai đoạn chuẩn bị dữ liệu (Offline):**
   * **Giai đoạn 1 (KG Subgraph):** Từ báo cáo ground-truth và nhãn CheXpert, ánh xạ các thực thể vào Knowledge Graph (như RadGraph/RadLex). Lấy ra các bộ ba quan hệ y khoa có cấu trúc:  
     $$\langle \text{Anatomy: Right Lung Base} \rangle \to \langle \text{Sign: Blunting costophrenic angle} \rangle \to \langle \text{Diagnosis: Pleural Effusion} \rangle$$
   * **Giai đoạn 2 (LLM Verbalization):** Đưa đồ thị con vào Medical LLM để chuyển thành đoạn suy luận theo chuẩn quy trình lâm sàng:
     * *Bước 1: Quét hệ thống giải phẫu (Systematic anatomical review).*
     * *Bước 2: Chỉ điểm bất thường & vị trí (Abnormality & location).*
     * *Bước 3: Chẩn đoán phân biệt & kết luận (Differential diagnosis).*
   * **Giai đoạn 3 (Lọc nghiêm ngặt):** So khớp lại với nhãn CheXbert và bản đồ tổn thương; loại bỏ mọi chuỗi suy luận phát sinh triệu chứng lạ. Giữ lại ~10.000 mẫu đạt chuẩn factual 100%.

2. **Tương tác trực tiếp với kiến trúc CAMANet:**
   * Dữ liệu huấn luyện có định dạng: `<think> [Lý luận lâm sàng từ KG] </think> [Findings]`.
   * **Đồng bộ hóa với mô-đun CAM & VTAC:**
     * Trong đoạn `<think>`, các từ giải phẫu và triệu chứng có thể neo trực tiếp vào bản đồ nhiệt $d^v$ của CAMANet.
     * Mô-đun **VTAC** ([cam_attn_con.py](file:///c:/Users/jajak/Documents/GitHub/CAMANet/modules/cam_attn_con.py)) sẽ gióng hàng cả chuỗi suy luận logic lẫn câu kết luận với vùng tổn thương trên ảnh.

---

### 3. Kế hoạch triển khai độc lập (Step-by-Step)
* **Tệp can thiệp:**
  1. `modules/tokenizers.py`: Bổ sung các special tokens `<think>`, `</think>` vào từ điển.
  2. `modules/datasets.py`: Cập nhật Dataset class hỗ trợ đọc trường `reasoning_trace` từ file JSON đã sinh (nếu không có, fallback về báo cáo thường).
  3. `modules/cam_attn_con.py`: Tùy chọn cho phép VTAC tính trọng số chú ý trên toàn bộ chuỗi `<think>` hoặc chỉ trên phần kết luận chẩn đoán.
* **Cờ cấu hình (CLI Args):**
  * `--use_cot` (bool, mặc định `False`).
  * `--cot_data_path` (str, đường dẫn tới tệp 10k reasoning traces).
  * `--cot_loss_weight` (float, trọng số loss cho các token trong block suy luận).

---

## Hướng dẫn bổ sung đề xuất mới (Extension Guidelines)

Khi bổ sung ý tưởng mới vào tài liệu này:
1. Thêm 1 dòng tóm tắt vào bảng **Danh mục đề xuất cải tiến (Index Matrix)** với mã `PROP-0X`.
2. Tạo một section riêng với 3 tiểu mục tiêu chuẩn:
   * **1. Động lực & Mục tiêu:** Làm rõ vấn đề cần giải quyết và giá trị mang lại.
   * **2. Thiết kế kiến trúc & Điểm can thiệp trong CAMANet:** Sơ đồ khối, biểu diễn toán học, cơ chế phối hợp với các mô-đun hiện có (VDM, VDMAE, VTAC).
   * **3. Kế hoạch triển khai độc lập:** Chỉ rõ các file cần sửa, cấu trúc class mới và cờ lệnh CLI để đảm bảo cắm/rút độc lập.
