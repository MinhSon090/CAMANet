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
| ~~PROP-02~~ | ~~Knowledge Graph Grounded CoT~~ | ~~Data Pipeline & Decoder~~ | Archived | ~~Có~~ | *Xem ghi chú bên dưới* |
| **PROP-03** | Latent Thinking Tokens | Decoder | Proposed | Có | `modules/my_encoder_decoder.py`, `cam_attn_con.py`, `trainer.py` |
| **PROP-04** | Cross-View Relational Attention | Visual Encoder | Proposed | Có | `modules/visual_extractor.py`, `models/model.py` |
| **PROP-05** | Focal Attention Alignment Loss | Loss Function (VTAC) | Proposed | Có | `modules/weighted_mesloss.py`, `trainer.py` |
| **PROP-06** | Dynamic Top-K Selection (VTAC) | VTAC Module | Proposed | Có | `modules/cam_attn_con.py` |
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
Decoder Layer L  ──► h_t (D=512)
                      │
                      ├──► Head 0 (Chính): LayerNorm ──► Linear(D, V) ──► Dự đoán w_t      (Loss_0)
                      │
                      ▼
               [Shared MTP Trunk] (1 Block Transformer nhẹ hoặc MLP dùng chung)
                      │
                      ├──► Head 1: LayerNorm_1 ──► Linear(D, V) ──► Dự đoán w_{t+1}  (Loss_1)
                      │
                      └──► Head 2: LayerNorm_2 ──► Linear(D, V) ──► Dự đoán w_{t+2}  (Loss_2)
```

1. **Khối dự đoán tương lai: Shared Trunk & Lightweight Heads:**
   * **Shared Trunk:** Thay vì mỗi MTP head mang một Transformer layer riêng (~3.1M params/head), toàn bộ $K$ heads dùng chung một khối biến đổi trung gian (1 Transformer block nhẹ hoặc LayerNorm + MLP).
   * **Lightweight Heads:** Mỗi head phụ $k$ chỉ gồm một `LayerNorm` độc lập kết hợp với ma trận chiếu từ điển `Linear(D, V)`. 
   * **Weight Tying (Chia sẻ trọng số):** Ma trận `Linear(D, V)` của các head phụ có thể dùng chung trọng số với Head 0 (hoặc share với ma trận word embedding). Thiết kế này **giảm hơn 60% tham số phụ**, ngăn ngừa nguy cơ overfitting trên các tập dữ liệu nhỏ (như IU X-Ray).

2. **Hàm mất mát mở rộng với Trọng số mềm tăng dần (Curriculum Soft Weighting):**
   $$\mathcal{L}_{MTP} = \mathcal{L}_{ce}(w_t) + \omega(e) \sum_{k=1}^K \gamma^k \mathcal{L}_{ce}(w_{t+k})$$
   * $\gamma \in (0, 1]$: Hệ số suy giảm trọng số theo khoảng cách tương lai (mặc định $\gamma = 0.8$).
   * **Trọng số mềm tăng dần $\omega(e)$ (Warm-up Schedule):** Ở các epoch đầu, mô hình chưa nắm vững ngữ pháp và từ vựng cơ bản, gradient từ các token tương lai xa dễ gây nhiễu (*noisy gradients*). Do đó, trọng số MTP $\omega(e)$ được tăng tuyến tính theo epoch $e$:
     $$\omega(e) = \min\left(1.0,\ \frac{e}{E_{warmup}}\right)$$
     *(Ví dụ: với $E_{warmup} = 5$, ở epoch 1 hệ số là $0.2$, và từ epoch 5 trở đi đạt mức tối đa $1.0$).*
   * **Tổng loss của CAMANet:**
     $$\mathcal{L}_{total} = \mathcal{L}_{MTP} + \lambda \mathcal{L}_{bce} + \delta \mathcal{L}_{mse}$$

---

### 3. Kế hoạch triển khai độc lập (Step-by-Step)
* **Tệp can thiệp:**
  1. `modules/my_encoder_decoder.py`: Thêm class `MTPModule(nn.Module)` tích hợp Shared Trunk và Lightweight Heads (hỗ trợ Weight Tying); sửa `Decoder.forward()` để trả về danh sách logits tương ứng với các bước tương lai.
  2. `modules/loss.py`: Bổ sung hàm tính loss MTP với mask che padding phù hợp theo độ lệch vị trí $k$.
  3. `modules/trainer.py`: Tính toán trọng số mềm $\omega(e)$ theo epoch hiện tại và tích hợp loss MTP vào vòng lặp huấn luyện; khi inference chỉ sử dụng Head 0 (hoặc bật speculative decoding).
* **Cờ cấu hình (CLI Args):**
  * `--use_mtp` (bool, mặc định `False` để tương thích ngược).
  * `--mtp_depth` (int, mặc định `2`, dự đoán 2 token kế tiếp).
  * `--mtp_decay` (float, mặc định `0.8`, hệ số $\gamma$).
  * `--mtp_warmup_epochs` (int, mặc định `5`, số epoch warm-up cho trọng số mềm $\omega$).
  * `--mtp_weight_tying` (bool, mặc định `True`, chia sẻ ma trận phân loại từ vựng với Head 0).

---

## ~~PROP-02: Knowledge Graph (KG) Grounded Chain-of-Thought (CoT)~~ [ARCHIVED]

> **⚠️ Trạng thái: Archived** — Đề xuất này bị lưu trữ do phụ thuộc nặng vào LLM bên ngoài (xem phân tích bên dưới). Thay thế bởi **PROP-03 (Latent Thinking Tokens)**.

> **Lý do lưu trữ:**
> 1. **Reproducibility:** Kết quả phụ thuộc hoàn toàn vào LLM cụ thể (GPT-4, Med-PaLM...), phiên bản, và prompt. Reviewer sẽ đặt câu hỏi về tính tái lập.
> 2. **Chi phí & Accessibility:** Sinh 10k reasoning traces bằng API thương mại tốn kém; academic lab khó reproduce.
> 3. **Data contamination risk:** LLM có thể đã thấy MIMIC reports trong pretraining → thông tin bị rò rỉ.
> 4. **Attribution:** Khó phân tách đóng góp — cải thiện nhờ CoT hay nhờ LLM knowledge?
>
> **Giá trị được kế thừa:** Ý tưởng *structured clinical reasoning* (quét anatomy → chỉ bất thường → chẩn đoán) được kế thừa ở PROP-03 dưới dạng implicit latent reasoning.

### 1. Động lực & Mục tiêu (Lưu trữ tham khảo)
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

## PROP-03: Latent Thinking Tokens

### 1. Động lực & Mục tiêu
* **Bài toán:** PROP-02 (KG-CoT) cố gắng dạy model suy luận bằng chuỗi text tường minh (`<think>...</think>`) nhưng phụ thuộc vào LLM bên ngoài để sinh dữ liệu huấn luyện. Đồng thời, dataset RRG không cung cấp nhãn quá trình suy luận, nên mọi cách tiếp cận explicit text CoT đều cần dữ liệu tổng hợp — vốn khó kiểm soát chất lượng.
* **Mục tiêu:** Trang bị cho Decoder khả năng **"suy luận ngầm" (implicit reasoning)** trước khi viết báo cáo, bằng cách chèn $M$ **learnable latent tokens** vào đầu chuỗi decoder. Các token này:
  1. Cross-attend vào visual features để "quét" ảnh và lập kế hoạch nội dung.
  2. Cung cấp ngữ cảnh suy luận cho các report tokens phía sau (qua causal self-attention).
  3. Được căn chỉnh với foreground map $d^v$ qua auxiliary loss — đảm bảo "suy nghĩ" hướng vào vùng bất thường.
* **Ưu điểm vượt trội so với PROP-02:**
  * ✅ **Không phụ thuộc LLM** — hoàn toàn tự chủ.
  * ✅ **Không cần dữ liệu mới** — dùng đúng dataset hiện tại.
  * ✅ **Reproducibility** — deterministic, dễ tái lập.
  * ✅ **Computational overhead cực nhẹ** — chỉ thêm $M$ tokens (8–16) vào sequence.
  * ✅ **Interpretability** — có thể visualize attention maps của thinking tokens → xem model "nghĩ gì".

---

### 2. Thiết kế kiến trúc & Điểm can thiệp trong CAMANet

```
Decoder Input (khi training):

[LTHINK_1] [LTHINK_2] ... [LTHINK_M] [BOS] w_1 w_2 ... w_T [EOS]
 ↑                                      ↑        ↑
 Learnable embeddings                   │        Report tokens (có CE loss)
 (không có CE loss trực tiếp)           │
                                        Boundary token
```

**Sơ đồ tương tác với CAMANet:**

```
                    Cross-Attention Maps
                         ↓
[LTHINK tokens] ──attend──► [Visual patches + fore_rep (VDMAE)]
       ↓                           ↑
  Implicit reasoning          CAM-guided features
       ↓
[Report tokens] ──attend──► [Visual patches + LTHINK hidden states]
       ↓
    VTAC alignment ◄──── fore_map (từ VDMAE)
```

1. **Learnable Query Embeddings:**
   * $M$ embedding vectors $\{q_1, q_2, \dots, q_M\} \in \mathbb{R}^{D}$, khởi tạo ngẫu nhiên (Xavier uniform) hoặc từ disease label embeddings.
   * Prepend vào đầu chuỗi target embedding trước khi đưa vào Decoder.

2. **Causal Mask mở rộng:**
   * Thinking tokens attend được lẫn nhau và vào visual features (qua cross-attention).
   * Report tokens attend được cả thinking tokens (qua causal self-attention) → nhận thông tin đã "suy luận".
   * Thinking tokens **không** attend vào report tokens (giữ tính causal).

3. **Auxiliary Thinking Alignment Loss:**
   * Cross-attention maps của thinking tokens lên visual patches: $\alpha_m \in \mathbb{R}^{N_v}$ ($N_v$ = số visual patches).
   * Trung bình tất cả thinking tokens:
     $$\bar{\alpha} = \frac{1}{M} \sum_{m=1}^{M} \alpha_m$$
   * Align với foreground map $d^v$ từ VDM/VDMAE:
     $$\mathcal{L}_{think} = \text{MSE}(\bar{\alpha},\ d^v)$$
   * Ý nghĩa: ép thinking tokens tập trung vào vùng bất thường trước khi model viết báo cáo.

4. **Tổng loss mở rộng:**
   $$\mathcal{L}_{total} = \mathcal{L}_{ce} + \lambda \mathcal{L}_{bce} + \delta \mathcal{L}_{mse} + \eta \mathcal{L}_{think}$$
   * $\eta$ (thinking alignment weight): mặc định $0.1$.

5. **Inference:** Thinking tokens vẫn được prepend vào sequence. Model vẫn sinh report bình thường — chỉ khác là report tokens giờ có thêm "ngữ cảnh suy luận" từ thinking tokens qua self-attention. Output text chỉ lấy phần sau thinking tokens.

---

### 3. Kế hoạch triển khai độc lập (Step-by-Step)
* **Tệp can thiệp:**
  1. `modules/my_encoder_decoder.py`:
     * Thêm `nn.Parameter` hoặc `nn.Embedding` cho $M$ thinking tokens.
     * Sửa `Transformer.decode()`: prepend thinking embeddings vào `tgt` trước khi chạy decoder layers.
     * Mở rộng causal mask để thinking tokens attend lẫn nhau nhưng không attend report tokens.
     * Trả về thêm `think_attns` (cross-attention maps của thinking tokens).
  2. `modules/cam_attn_con.py`: Bổ sung hàm tính `think_align_loss` — MSE giữa trung bình cross-attention maps của thinking tokens và `fore_map`.
  3. `modules/trainer.py`: Tích hợp `think_align_loss` vào vòng lặp huấn luyện khi `--use_think_tokens` được bật. Khi tính CE loss, bỏ qua vị trí thinking tokens (chỉ tính trên report tokens).
* **Cờ cấu hình (CLI Args):**
  * `--use_think_tokens` (bool, mặc định `False` để tương thích ngược).
  * `--num_think_tokens` (int, mặc định `8`).
  * `--think_align_w` (float, mặc định `0.1`).
  * `--think_init` (str, `random` | `label_embed`, mặc định `random`). Nếu `label_embed`: khởi tạo thinking tokens từ embeddings của 8 disease labels phổ biến nhất.

---

## PROP-04: Cross-View Relational Attention (IU X-Ray)

### 1. Động lực & Mục tiêu
* **Bài toán:** IU X-Ray cung cấp 2 views (frontal + lateral) cho mỗi bệnh nhân. Hiện tại, CAMANet xử lý bằng cách trích xuất features từ mỗi view **độc lập** rồi **concatenate đơn giản** (`torch.cat`) trước khi đưa vào encoder. Cách này bỏ qua hoàn toàn mối quan hệ bổ trợ giữa 2 views:
  * Lateral view xác nhận tổn thương thấy trên frontal (ví dụ: pleural effusion).
  * Một số cấu trúc chỉ thấy rõ trên 1 view (ví dụ: retrocardiac opacity chỉ rõ trên lateral).
* **Mục tiêu:** Thêm một **Cross-View Attention** layer cho phép 2 views trao đổi thông tin trước khi concat, để features từ mỗi view được enrich bởi ngữ cảnh từ view kia.

---

### 2. Thiết kế kiến trúc & Điểm can thiệp trong CAMANet

```
Frontal Features ──┬── Cross-Attn(Q=F, KV=L) ──► Frontal' ──┐
                   │                                          ├── Concat ──► Encoder
Lateral Features ──┴── Cross-Attn(Q=L, KV=F) ──► Lateral' ──┘
```

1. **Cross-View Attention Module:**
   * Một `nn.MultiheadAttention` layer (hoặc custom attention block) thực hiện:
     $$\text{Frontal}' = \text{Frontal} + \text{CrossAttn}(Q=\text{Frontal},\ K=V=\text{Lateral})$$
     $$\text{Lateral}' = \text{Lateral} + \text{CrossAttn}(Q=\text{Lateral},\ K=V=\text{Frontal})$$
   * Residual connection đảm bảo thông tin gốc không bị mất.
   * Chỉ cần 1 attention layer — rất nhẹ (thêm ~0.5M params với $D=512$).

2. **Vị trí can thiệp:** Sau `VisualExtractor`, trước khi concatenate features trong `R2GenModel.forward()`.

3. **Tương thích:** Chỉ kích hoạt khi dataset là IU X-Ray (multi-image). Với MIMIC-CXR (single-image), module bị bypass hoàn toàn.

---

### 3. Kế hoạch triển khai độc lập (Step-by-Step)
* **Tệp can thiệp:**
  1. `modules/visual_extractor.py` (hoặc file mới `modules/cross_view_attn.py`): Thêm class `CrossViewAttention(nn.Module)` chứa 2 cross-attention sub-layers + LayerNorm + residual.
  2. `models/model.py`: Trong `forward()`, sau khi lấy features từ 2 views, gọi `CrossViewAttention` trước `torch.cat`. Chỉ kích hoạt khi `args.cross_view_attn == True` và input có 2 images.
* **Cờ cấu hình (CLI Args):**
  * `--cross_view_attn` (bool, mặc định `False`).
  * `--cross_view_heads` (int, mặc định `8`, số attention heads).

---

## PROP-05: Focal Attention Alignment Loss

### 1. Động lực & Mục tiêu
* **Bài toán:** Mô-đun VTAC hiện dùng **MSE loss** (`modules/weighted_mesloss.py`) để căn chỉnh cross-attention map với foreground map $d^v$. Tuy nhiên, vùng bất thường thường chỉ chiếm **5–15% diện tích ảnh** → MSE bị dominate bởi vùng background (chiếm 85–95%), khiến model dễ học shortcut "attention phẳng" thay vì tập trung vào tổn thương.
* **Mục tiêu:** Thay thế (hoặc bổ sung) MSE bằng loss function có **cơ chế focal** — upweight các vị trí mà attention lệch nhiều so với target, đặc biệt tại vùng foreground.

---

### 2. Thiết kế kiến trúc & Điểm can thiệp trong CAMANet

Hai phương án (chọn 1 hoặc kết hợp):

1. **Phương án A — Foreground-Weighted MSE:**
   * Dùng chính foreground map $d^v$ làm trọng số không gian:
     $$\mathcal{L}_{fw\text{-}mse} = \frac{1}{N}\sum_{i=1}^{N} (1 + \alpha \cdot d^v_i) \cdot (a_i - d^v_i)^2$$
   * $\alpha$ (foreground emphasis): mặc định $5.0$. Vùng có $d^v_i$ cao (foreground) được supervise mạnh gấp $1 + \alpha$ lần vùng background.
   * **Ưu điểm:** Cực kỳ đơn giản, thêm 1 dòng code.

2. **Phương án B — Focal MSE:**
   * Lấy cảm hứng từ Focal Loss (Lin et al., RetinaNet):
     $$\mathcal{L}_{focal\text{-}mse} = \frac{1}{N}\sum_{i=1}^{N} (1 - e^{-|a_i - d^v_i|^2})^{\beta} \cdot |a_i - d^v_i|^2$$
   * $\beta$ (focusing parameter): mặc định $2.0$. Các vị trí có error lớn (attention lệch nhiều so với target) được upweight mạnh hơn.
   * **Ưu điểm:** Tự động adaptive theo mức độ lỗi, không cần biết trước foreground map.

---

### 3. Kế hoạch triển khai độc lập (Step-by-Step)
* **Tệp can thiệp:**
  1. `modules/weighted_mesloss.py`: Thêm class `FocalMSELoss(nn.Module)` với 2 mode (foreground-weighted / focal). Giữ nguyên class `Weighted_MSELoss` gốc để backward-compatible.
  2. `modules/trainer.py`: Khi `--focal_mse` bật, khởi tạo `FocalMSELoss` thay cho `Weighted_MSELoss`.
* **Cờ cấu hình (CLI Args):**
  * `--focal_mse` (bool, mặc định `False` — fallback về MSE gốc).
  * `--focal_mode` (str, `fw_mse` | `focal`, mặc định `fw_mse`).
  * `--focal_alpha` (float, mặc định `5.0`, chỉ dùng cho mode `fw_mse`).
  * `--focal_beta` (float, mặc định `2.0`, chỉ dùng cho mode `focal`).

---

## PROP-06: Dynamic Top-K Selection trong VTAC

### 1. Động lực & Mục tiêu
* **Bài toán:** VTAC ([cam_attn_con.py](file:///c:/Users/jajak/Documents/GitHub/CAMANet/modules/cam_attn_con.py)) chọn "disease-relevant words" bằng **fixed top-k%** (hyperparameter `--topk`, ví dụ 10–30%) dựa trên cosine similarity giữa word embedding và foreground representation. Tỷ lệ cố định này không tối ưu:
  * Báo cáo **bình thường** (ca sạch) có rất ít từ bệnh lý → $k$ nên nhỏ, nếu không sẽ chọn nhầm noise words.
  * Báo cáo **bất thường phức tạp** (nhiều bệnh) có nhiều từ quan trọng → $k$ nên lớn để bao phủ đủ.
  * Trong thực tế, cùng 1 giá trị top-k% phải phục vụ cả 2 loại → suboptimal.
* **Mục tiêu:** Thay fixed top-k bằng **learnable gating mechanism**, cho phép model tự quyết định mỗi word token có "đáng để align" hay không, adaptive theo từng sample.

---

### 2. Thiết kế kiến trúc & Điểm can thiệp trong CAMANet

```
Word Embedding h_t ──┐
                     ├── Gate Network ──► g_t ∈ [0, 1]
Fore Rep  ───────────┘

Aggregated Attention = Σ g_t · attn_t   (weighted sum thay vì hard top-k mean)
```

1. **Learnable Gate Network:**
   * Cho mỗi word token $t$, tính gate value:
     $$g_t = \sigma\Big(W_g \cdot [h_t \| \text{fore\_rep}] + b_g\Big) \in [0, 1]$$
   * $W_g \in \mathbb{R}^{1 \times 2D}$, $b_g \in \mathbb{R}$. Chỉ cần 1 linear layer — cực nhẹ.
   * $\sigma$ là sigmoid.

2. **Aggregation thay đổi:**
   * **Hiện tại (hard top-k):** Chọn top-k% tokens, lấy mean cross-attention maps của chúng.
   * **Đề xuất (soft gating):**
     $$\text{total\_attn} = \frac{\sum_{t=1}^{T} g_t \cdot \alpha_t}{\sum_{t=1}^{T} g_t + \epsilon}$$
     * $\alpha_t$ là cross-attention map của word token $t$. $\epsilon$ chống chia 0.
   * **Ưu điểm:** Differentiable end-to-end, gradient chạy qua gate → model tự học chọn.

3. **Sparsity Regularization (tùy chọn):**
   * Thêm L1 penalty trên gate values để khuyến khích sparsity (chỉ vài words có gate cao):
     $$\mathcal{L}_{sparse} = \frac{\mu}{T} \sum_{t=1}^{T} g_t$$
   * $\mu$ (sparsity weight): mặc định $0.01$.

---

### 3. Kế hoạch triển khai độc lập (Step-by-Step)
* **Tệp can thiệp:**
  1. `modules/cam_attn_con.py`:
     * Thêm `GateNetwork(nn.Module)` — 1 linear layer + sigmoid.
     * Trong `CamAttnCon.forward()`: khi `--dynamic_topk` bật, thay thế hard top-k selection bằng soft gating. Khi tắt, giữ nguyên logic top-k gốc.
  2. `modules/trainer.py`: Thêm `sparsity_loss` vào tổng loss (nếu `--gate_sparsity > 0`).
* **Cờ cấu hình (CLI Args):**
  * `--dynamic_topk` (bool, mặc định `False` — fallback về fixed top-k).
  * `--gate_sparsity` (float, mặc định `0.01`, trọng số L1 regularization trên gate values).

---

## Hướng dẫn bổ sung đề xuất mới (Extension Guidelines)

Khi bổ sung ý tưởng mới vào tài liệu này:
1. Thêm 1 dòng tóm tắt vào bảng **Danh mục đề xuất cải tiến (Index Matrix)** với mã `PROP-0X`.
2. Tạo một section riêng với 3 tiểu mục tiêu chuẩn:
   * **1. Động lực & Mục tiêu:** Làm rõ vấn đề cần giải quyết và giá trị mang lại.
   * **2. Thiết kế kiến trúc & Điểm can thiệp trong CAMANet:** Sơ đồ khối, biểu diễn toán học, cơ chế phối hợp với các mô-đun hiện có (VDM, VDMAE, VTAC).
   * **3. Kế hoạch triển khai độc lập:** Chỉ rõ các file cần sửa, cấu trúc class mới và cờ lệnh CLI để đảm bảo cắm/rút độc lập.
