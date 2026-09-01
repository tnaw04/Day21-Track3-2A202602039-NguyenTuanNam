# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Tuấn Nam  **MSSV**: 2A202602039  **Ngày**: 01/09/2026
**Tier**: `CPU`  **Base model**: `Qwen/Qwen3.5-0.8B`  **GPU thực tế**: `Apple Silicon M-Series (MPS / fp16)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | Ticket CSKH thương mại điện tử (250 ticket CSKH → JSON triage 4 trường) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 256 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 58 steps |

**Template có giữ khối `<think>` không?** `có` — *(results/template_check.json)*
Nếu không: bạn đã xử lý thế nào?
Template mặc định của Qwen 3.5 giữ nguyên khối reasoning `<think>` (`renders_reasoning = true`, `has_thinking_markers = true`). Khi nạp dữ liệu huấn luyện, ta đảm bảo cấu trúc template và generation prompt được định dạng đồng nhất giữa khâu huấn luyện và khâu suy luận.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
[  OK  ] token  38: ' {\n'                 (id=1135)   -> label=1135
[  OK  ] token  39: '  '                   (id=256)    -> label=256
[  OK  ] token  40: ' "'                   (id=374)    -> label=374
[  OK  ] token  41: 'intent'               (id=1010)   -> label=1010
[  OK  ] token  42: '":'                   (id=1152)   -> label=1152
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.5778 | 0.000 | 8340.9 |
| (b) base + optimized prompt | 0.500 | 0.5778 | 1.000 | 2478.2 |
| (c) LoRA fine-tune | 0.985 | 0.1000 | 1.000 | 1992.0 |

**(b) có thật sự mạnh hơn (a) không?** `có` — (b) đạt 0.500 Target Accuracy và 1.000 Format Accuracy so với 0.000 của (a).
Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao?
Tôi giữ nguyên bản `OPTIMIZED_PROMPT` chuẩn mực từ hệ thống (mã SHA: `719e74d3b6232053`), không sửa đổi làm yếu đi để tâng bốc bản fine-tune hay sửa đổi làm mạnh quá mức gây thiên vị. Bản prompt tối ưu chuẩn mực này đã cung cấp đầy đủ định nghĩa 4 trường JSON và quy tắc phân loại, đóng vai trò là một "mốc chuẩn trung thực" (honest baseline).

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 10,822,656 | 0.0001 | 0.5781 | **0.9850** | 1049.4 | 1.6 |
| `attn_only` | q,v | 271 *(matched)* | 10,822,656 | 0.0001 | 0.5765 | **0.9300** | 1010.0 | 1.6 |
| `wrong_lr` | text-linear | 16 | 10,822,656 | 1e-05 | 1.6093 | **0.3350** | 1110.4 | 1.6 |
| `qlora` | text-linear | 16 | 10,822,656 | 0.0001 | 0.6110 | **0.9450** | 1589.5 | 1.2 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

Trên tập target, `attn_only` **thua** `correct` rõ rệt (0.9300 so với 0.9850, chênh lệch 5.5%). Tuy nhiên, nếu chỉ nhìn vào cột train loss, `attn_only` lại có loss huấn luyện thấp hơn một chút (0.5765 so với 0.5781 của `correct`) do rank cực cao (r=271) dẫn đến việc ghi nhớ (memorization) cục bộ trên tập train. Điều này chứng minh rằng **vị trí gắn adapter (placement) quan trọng hơn nhiều so với việc tăng rank**: việc phân bổ tham số đồng đều trên toàn bộ các lớp tuyến tính (`text-linear` bao gồm cả MLP và projection layers) giúp mô hình học biểu diễn tổng quát và suy luận tốt hơn hẳn so với việc dồn toàn bộ ngân sách tham số vào riêng hai cổng Attention `q, v`.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

Đường loss của `wrong_lr` bị kẹt ở mức rất cao (loss cuối đạt 1.6093, cao gần gấp 3 lần so với 0.5781 của `correct`), dẫn đến điểm target thảm hại chỉ đạt 0.3350. Nếu chỉ nhìn vào đường loss giảm chậm mà không biết learning rate bị đặt quá nhỏ (1e-5), một kỹ sư thiếu kinh nghiệm sẽ dễ kết luận sai lầm rằng *"LoRA không đủ dung lượng để học tác vụ này, cần phải tăng rank lên thật cao hoặc chuyển sang Full Fine-tuning"*. Trong thực tế, LoRA chỉ cập nhật một ma trận nhỏ nên cần tốc độ học lớn hơn xấp xỉ 10 lần (~1e-4) so với Full Fine-tuning để có đủ lực kéo gradient hội tụ trong cùng số bước.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

QLoRA 4-bit giúp giảm dung lượng bộ nhớ mô hình (~25-30% VRAM), nhưng phải trả giá bằng việc thời gian huấn luyện tăng mạnh (1,589.5 giây so với 1,049.4 giây của `correct`, tức chậm hơn ~51% do overhead giải lượng tử hóa dequantization on-the-fly) và điểm target bị suy giảm từ 0.9850 xuống 0.9450, đồng thời latency suy luận tăng từ 1992ms lên 2528ms. Kết quả đo đạc thực tế hoàn toàn ủng hộ khuyến nghị chính thức của nhà phát triển Qwen: không nên dùng QLoRA 4-bit cho dòng mô hình Qwen 3.5 nếu phần cứng vẫn đủ sức chứa mô hình ở độ chính xác 16-bit (fp16/bf16).

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.485` · `regression Δ = -0.478` · `valid_trace_rate = 0.0`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
(Một FAILED được phân tích tốt ăn điểm cao hơn một PASSED không giải thích được.)

Cổng hồi quy ra phán quyết **FAILED** bởi vì mặc dù bản LoRA fine-tune (`correct`) có sự bứt phá vượt trội về năng lực phân loại ticket chuyên môn (Target Accuracy tăng từ 0.500 lên **0.985**, tức `target Δ = +0.485`), khả năng trả lời các câu hỏi kiến thức phổ thông tiếng Việt lại bị sụt giảm nghiêm trọng từ 0.578 xuống **0.100** (`regression Δ = -0.478`, vượt xa ngưỡng dung sai cho phép 0.020).

Hiện tượng này minh họa hoàn hảo bài học lý thuyết về **quên thảm họa (Catastrophic Forgetting) / Overfitting miền chuyên biệt**: khi ta huấn luyện mô hình 0.8B hoàn toàn trên 100% dữ liệu ticket CSKH chuyên biệt mà không trộn thêm bất kỳ dữ liệu tổng quát nào (General Replay Data), các trọng số LoRA đã điều chỉnh toàn bộ biểu diễn ngôn ngữ để tối ưu hóa riêng cấu trúc JSON triage, làm triệt tiêu các mẫu kích hoạt dùng để trả lời kiến thức tổng quát. Để đưa hệ thống này vào production an toàn, ta bắt buộc phải áp dụng kỹ thuật **Replay Buffer** (trộn 1–5% dữ liệu đàm thoại tổng quát vào tập train) nhằm giữ vững kiến thức nền trong khi vẫn đạt điểm tác vụ cao.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper khô | `van_chuyen, thap, ốp lưng điện thoại, tich_cuc` | `san_pham_loi, thap` (sai) | `van_chuyen, thap, ốp lưng điện thoại, tich_cuc` | ✅ FT thắng: Nhận diện chính xác ý định giao hàng và thái độ tích cực |
| 2 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu.  | `hoi_thong_tin, trung_binh, ốp lưng điện thoại, trung_tinh` | `hoan_tien, thap` (sai) | `hoi_thong_tin, trung_binh, ốp lưng điện thoại, trung_tinh` | ✅ FT thắng: Phân loại đúng ý định hỏi giá và mức độ khẩn cấp |
| 3 | Shop ơi, mình đặt máy xay sinh tố mã đơn DH777946. Khi nào có tiền về. | `hoan_tien, trung_binh, máy xay sinh tố, tieu_cuc` | `hoan_tien, trung_binh, máy xay sinh tố, tieu_cuc` | `hoan_tien, thap, máy xay sinh tố, tieu_cuc` | ❌ **FT thua**: FT dự đoán nhầm độ khẩn cấp từ `trung_binh` thành `thap` |
| 4 | Chào shop, mình đặt nồi chiên không dầu mã đơn VN558606. Giao hàng chậ | `van_chuyen, trung_binh, nồi chiên không dầu, tieu_cuc` | `van_chuyen, trung_binh, nồi chiên không dầu, tieu_cuc` | `van_chuyen, thap, nồi chiên không dầu, tieu_cuc` | ❌ **FT thua**: Khách phàn nàn giao chậm nhưng FT đánh giá mức độ khẩn cấp là `thap` |
| 5 | Chào shop, mình đặt đèn bàn LED mã đơn VN718973. Cho tôi trả lại. Khôn | `doi_tra, thap, đèn bàn LED, tieu_cuc` | `doi_tra, thap, đèn bàn LED, tieu_cuc` | `van_chuyen, thap, đèn bàn LED, tieu_cuc` | ❌ **FT thua**: Khách yêu cầu trả hàng (`doi_tra`) nhưng FT nhầm sang `van_chuyen` |

Có mẫu chung nào ở các ca FT thua không?
Có hai mẫu sai sót nổi bật ở các ca FT thua:
1. **Thiên vị độ khẩn cấp thấp (`urgency: thap`)**: Mô hình fine-tune có xu hướng gán nhãn `thap` cho hầu hết các câu có từ ngữ phàn nàn nhẹ, trong khi nhãn chuẩn đánh giá là `trung_binh`.
2. **Nhầm lẫn ranh giới giữa `doi_tra` và `van_chuyen`**: Khi câu chứa các cụm từ vừa đề cập đến việc trả hàng vừa có từ ngữ liên quan đến nhận hàng/giao hàng, mô hình dễ bị kích hoạt phản xạ phân loại sang nhóm `van_chuyen`.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn
bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?

Tôi **chưa nên deploy ngay** bản fine-tune này vào hệ thống chatbot đa năng tổng thể, nhưng **có thể deploy ngay vào một worker vi dịch vụ độc lập chuyên trách JSON triage**. Lý do là vì trong vai trò một worker phân loại backend, mô hình đạt độ chính xác tác vụ cực cao (**98.5%**, vượt trội hoàn toàn so với 50.0% của Prompting), định dạng JSON chuẩn 100% và thời gian đáp ứng nhanh hơn 20% (1,992ms). Tuy nhiên, nếu deploy làm chatbot trực tiếp tương tác với người dùng, mô hình sẽ thất bại vì đã bị mất khả năng trò chuyện tổng quát (regression tụt còn 0.100).

Đòn bẩy quan trọng nhất trong lab này được xếp theo thứ tự:
1. **Loss Masking đúng đắn (NB1)**: Nền tảng cốt tử đảm bảo mô hình chỉ học dự đoán câu trả lời JSON, không lãng phí dung lượng học prompt câu hỏi.
2. **Learning Rate phù hợp cho LoRA (NB4)**: Đặt LR chuẩn 1e-4 thay vì 1e-5 của full-FT là yếu tố sống còn để LoRA hội tụ.
3. **Vị trí gắn Adapter toàn diện (text-linear)**: Phủ đều adapter lên toàn bộ các tầng tuyến tính vượt trội hơn hẳn việc dồn ép rank cao vào riêng Attention.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Loss Masking không thể dựa vào cờ thư viện mù quáng**: Phải kiểm chứng thực tế (`mask_proof.json`) vì chat template có thể không chứa marker `{% generation %}`, khiến trainer âm thầm bỏ qua toàn bộ loss mà không báo lỗi.
2. **Không bao giờ dùng Training Loss làm thước đo xếp hạng**: Run `attn_only` (r=271) có loss huấn luyện thấp hơn `correct` nhưng điểm Target thực tế lại kém hơn 5.5% do bị overfit cục bộ.
3. **Mốc so sánh trung thực (Honest Baseline)**: Phải luôn đo năng lực mô hình fine-tune đối đầu với một Prompt tối ưu chuẩn mực (Baseline b) và kiểm soát hiện tượng quên thảm họa qua cổng hồi quy (Regression Gate).

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Trộn thêm 3–5% tập dữ liệu hội thoại tiếng Việt tổng quát (Replay Buffer) vào tập huấn luyện 225 ticket để giữ điểm Regression trên 0.55 trong khi vẫn duy trì Target trên 98%, giúp bản LoRA chính thức vượt qua cổng hồi quy (`PASSED`).

---

## Phụ lục — thưởng đã làm

- [x] **B1 NB6 merge + hot-swap (+3 điểm)**:
  - **Kết quả đo lường (`results/merge_check.json`)**: Trước merge đạt **0.9850**, sau merge đạt **0.9850** (`delta = +0.0000`, hoàn toàn không bị suy giảm độ chính xác).
  - **Thử nghiệm Hot-swap đa adapter trên cùng 1 Base Model**: Đã nạp thành công 3 adapter `['correct', 'attn_only', 'qlora']` trên cùng một mô hình nền duy nhất trong bộ nhớ và chuyển đổi phục vụ theo từng request:
    - `[correct]` -> `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}`
    - `[attn_only]` -> `{"intent": "doi_tra", "urgency": "trung_binh", "product": "chuột không dây", "sentiment": "tich_cuc"}`
    - `[qlora]` -> `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}`
  - **Phân tích kỹ thuật**:
    1. *Độ chính xác sau merge*: Phép tính $W = W_0 + \frac{\alpha}{r}BA$ được bảo toàn chính xác trên fp16, không làm thay đổi điểm số tác vụ.
    2. *Đánh đổi kiến trúc*: Bản merged triệt tiêu toàn bộ latency overhead của adapter trong đồ thị suy luận, nhưng mất đi khả năng phục vụ động.
    3. *Khi nào nên giữ adapter riêng*: Rất cần thiết trong kiến trúc Multi-tenant (hàng trăm khách hàng với các bộ quy tắc riêng trên cùng 1 GPU), cho phép rollback adapter tức thì trong mili-giây và chạy A/B test độc lập mà không cần reload mô hình nền.

- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
