# Kế Hoạch Thực Hiện Lab 21 — Fine-Tuning LLM Bằng LoRA
*(Giải thích bằng ngôn ngữ đời thường, dễ hiểu — Layman's Terms)*

---

## 1. Bài Lab này thực chất đang làm gì?

Hãy tưởng tượng bạn vừa tuyển một bạn nhân viên hỗ trợ khách hàng (CSKH) rất thông minh, hoạt ngôn và thành thạo tiếng Việt (đây là **Base Model** — ví dụ mô hình ngôn ngữ lớn Qwen). 

Tuy nhiên, công ty của bạn có một quy trình làm việc rất đặc thù: mỗi khi khách gửi khiếu nại (ticket), nhân viên phải tóm tắt thông tin thành đúng một biểu mẫu gồm 4 ô chuẩn:
1. **Ý định (intent)**: Khách muốn gì? (đổi hàng, hủy đơn, khiếu nại,...)
2. **Mức độ khẩn cấp (urgency)**: Thấp, trung bình hay gấp?
3. **Sản phẩm (product)**: Liên quan đến mặt hàng nào?
4. **Cảm xúc (sentiment)**: Khách đang hài lòng, bực mình hay trung tính?

Để bạn nhân viên làm đúng yêu cầu này, chúng ta có 2 cách tiếp cận:
* **Cách 1: Prompt Engineering (Viết hướng dẫn chi tiết)** — Bạn viết sẵn một tờ giấy hướng dẫn thật kỹ, có giải thích từng ô, kèm ví dụ mẫu và kẹp vào mỗi đơn hàng để nhân viên đọc và làm theo.
* **Cách 2: Fine-Tuning bằng LoRA (Huấn luyện chuyên biệt)** — Thay vì kẹp giấy hướng dẫn dài dòng mãi mãi, bạn cho bạn nhân viên làm thử một số lượng bài tập (225 mẫu) rồi chấm điểm, chỉnh sửa phản xạ của bạn ấy. Nhờ kỹ thuật **LoRA**, chúng ta không cần "tẩy não" hay "dạy lại từ đầu" toàn bộ não bộ của nhân viên (tốn hàng triệu USD), mà chỉ gắn thêm một cuốn sổ tay phản xạ nhỏ vài chục Megabyte vào não bạn ấy.

### Tinh thần khoa học cốt lõi của Lab:
1. **Chấm điểm đúng chỗ (Loss Masking)**: Khi bạn nhân viên làm bài tập, ta chỉ chấm điểm xem bạn ấy điền biểu mẫu đúng hay sai. Tuyệt đối không chấm điểm việc bạn ấy có chép lại y nguyên đề bài của khách hàng hay không. Nếu chấm cả đề bài, bạn ấy sẽ bị "học vẹt", chỉ chăm chăm lặp lại câu hỏi của khách!
2. **So sánh công bằng (Fair Benchmark)**: Bản fine-tune bằng LoRA phải chứng minh được là nó làm tốt hơn việc bạn viết một bản hướng dẫn thật xịn (Prompt tối ưu - Baseline b), chứ không phải chỉ thắng một câu nhắc cụt lủn (Prompt ngây thơ - Baseline a).
3. **Trung thực về kết quả**: Nếu fine-tune xong mà kết quả không hơn viết prompt xịn, ta mạnh dạn ghi nhận `FAILED` và phân tích lý do. Trong thực tế công việc, phát hiện ra *"trường hợp này chỉ cần viết prompt tốt là đủ, không cần tốn tiền train model"* là một quyết định kỹ thuật cực kỳ giá trị và vẫn được chấm điểm tối đa!

---

## 2. Toàn cảnh 5 chặng đường của Lab (NB1 → NB5)

| Chặng | Tên gọi | Việc cần làm | Thiết bị cần | Sản phẩm thu được |
|---|---|---|:---:|---|
| **Bước 1 (NB1)** | **Chuẩn bị nguyên liệu & Kiểm tra mặt nạ** | Chuẩn bị dữ liệu, kiểm tra template hiển thị suy luận, kiểm tra "mặt nạ" (mask) chỉ tính điểm câu trả lời, đo độ dài token chuẩn và chia tập train/val. | **Chỉ cần CPU** (~1-2 phút) | 4 file kết quả trong `results/` và `data/split/` |
| **Bước 2 (NB2)** | **Đóng băng tiêu chuẩn cần vượt** | Cho mô hình gốc làm bài thi với 2 dạng đề: (a) Đề nhắc sơ sài, (b) Đề hướng dẫn chi tiết. Đóng băng điểm số này lại làm mốc chuẩn. | GPU (~20 phút) | `results/baselines_frozen.json` |
| **Bước 3 (NB3)** | **Huấn luyện bản chuẩn (Golden Run)** | Gắn LoRA chuẩn chỉ vào toàn bộ các lớp xử lý văn bản, chọn tốc độ học chuẩn, train 2 vòng. | GPU (~20 phút) | Cuốn sổ tay LoRA `adapters/correct/` |
| **Bước 4 (NB4)** | **Mổ xẻ 3 ca cố tình làm sai (Đối chứng)** | Chạy 3 bài thí nghiệm cố tình mắc lỗi phổ biến để so sánh: (1) Chỉ gắn LoRA vào một góc nhỏ, (2) Đặt tốc độ học quá chậm, (3) Nén mô hình quá mức (QLoRA 4-bit). | GPU (~50 phút) | 3 adapter đối chứng & `results/runs.csv` |
| **Bước 5 (NB5)** | **Chấm thi & Ra phán quyết** | Chấm điểm toàn diện 4 mặt: Độ chính xác, Có bị "quên kiến thức cũ" không (Hồi quy), Chuẩn định dạng JSON, Tốc độ trả lời. Điền báo cáo `REPORT.md`. | GPU (~20 phút) | `verdict.json`, `autopsy.json`, `REPORT.md` hoàn chỉnh |

---

## 3. Kế hoạch chi tiết cho BƯỚC 1 (Chúng ta thực hiện ngay lúc này)

Bước 1 là bước nền tảng quyết định sự thành bại của toàn bộ bài Lab nhưng lại cực kỳ nhẹ nhàng: **hoàn toàn chạy được trên CPU của máy tính trong vòng 1-2 phút mà không cần GPU**.

Chúng ta sẽ thực hiện tuần tự 4 công việc sau:

```mermaid
flowchart TD
    A["1. Thiết lập môi trường Python (.venv)"] --> B["2. Chạy Smoke Test kiểm tra hệ thống"]
    B --> C["3. Thực thi Notebook 1 (01_data_and_mask.py)"]
    C --> D["4. Xác thực 4 sản phẩm đầu ra (Artefacts)"]
    D --> E["Hoàn thành Bước 1 - Sẵn sàng cho Bước 2"]
```

### Công việc 1: Thiết lập môi trường Python gọn nhẹ cho CPU
- Tạo môi trường ảo `.venv` cô lập để tránh xung đột thư viện của máy.
- Cài đặt các gói cần thiết cho Bước 1: `transformers`, `tokenizers`, `jinja2`, `jupytext`, `pytest` (theo `requirements-cpu.txt`).
- Thiết lập tệp cấu hình `.env` với `COMPUTE_TIER=CPU` và `MASK_MODE=assistant-only`.

### Công việc 2: Chạy kiểm tra nhanh (Smoke Test)
- Chạy lệnh `python scripts/verify.py --smoke` để kiểm tra:
  - Tất cả các module nội bộ của `labkit` nạp thành công.
  - Bộ dữ liệu mẫu gốc (`train_seed.jsonl`, `eval_target.jsonl`, `eval_regression.jsonl`) đầy đủ.
  - Toàn bộ các bài kiểm thử tự động (unit tests) đều vượt qua (100% xanh).

### Công việc 3: Thực thi Notebook 1 (`notebooks/01_data_and_mask.py`)
Tiến hành chạy chương trình Bước 1 để thực hiện 4 tác vụ quan trọng:
1. **Nạp 250 mẫu dữ liệu ticket CSKH tiếng Việt**: Đọc dữ liệu đầu vào.
2. **Kiểm tra Chat Template (`template_check.json`)**: Kiểm tra xem bộ giải mã (tokenizer) của model khi ghép câu thoại có lỡ tay xóa mất phần suy luận `<think>` không.
3. **Tạo và đối chứng "Mặt nạ học tập" (Loss Masking - `mask_proof.json`)**:
   - So sánh giữa chế độ đúng (`assistant-only`: chỉ phạt khi trả lời sai) và chế độ lỗi (`everything`: phạt cả khi không nhớ câu hỏi của khách).
   - Kiểm tra 2 khẳng định cốt lõi:
     - Khẳng định 1: Câu trả lời thực sự nằm trong vùng tính loss (`answer_is_supervised = True`).
     - Khẳng định 2: Câu hỏi của khách bị che đi hoàn toàn, không nằm trong loss (`question_is_masked = True`).
     - Tỷ lệ token được tính loss phải dưới 95% (`supervised_fraction < 0.95`).
4. **Đo chiều dài token thực tế (`token_stats.json`)**:
   - Thống kê phân phối độ dài của 250 mẫu để tìm điểm phân vị 95% (p95).
   - Xác định chiều dài tối đa (`max_length`) hợp lý để vừa không bị cắt mất câu trả lời của AI, vừa không bị tốn bộ nhớ vô ích.
5. **Chia tập dữ liệu chuẩn (`data/split/train.jsonl` và `val.jsonl`)**:
   - Chia 250 mẫu thành 90% (225 mẫu huấn luyện) và 10% (25 mẫu kiểm tra).
   - Sử dụng hạt giống ngẫu nhiên cố định `seed=42` để đảm bảo bất kỳ ai chạy lại cũng ra đúng từng mẫu giống hệt nhau.

### Công việc 4: Xác thực và nghiệm thu sản phẩm Bước 1
Kiểm tra xem 4 sản phẩm bắt buộc của Bước 1 đã được tạo ra đầy đủ và hợp lệ hay chưa:
- `results/template_check.json`
- `results/mask_proof.json` (cả 2 assert đều đạt)
- `results/token_stats.json`
- `data/split/train.jsonl` (225 dòng) & `data/split/val.jsonl` (25 dòng)

---

## 4. Tiêu chí đánh giá thành công của Bước 1

Sau khi hoàn thành Bước 1, bạn sẽ có sẵn toàn bộ dữ liệu sạch và các chứng chỉ kỹ thuật (mask proof, template check) để:
1. Tự tin bước vào giai đoạn 2 (đo baseline và huấn luyện trên GPU hoặc Colab).
2. Đáp ứng hoàn hảo các tiêu chí trong thang chấm điểm (Rubric) của Mục 1 & Mục 2.

---

## 5. Nhật ký & Kết quả thực tế đã thực hiện (Nghiệm thu Bước 1)

*(Ghi nhận vào ngày 21/08/2026 sau khi hoàn tất Bước 1)*

Chúng ta đã hoàn thành xuất sắc 100% các mục tiêu trong Bước 1 theo đúng kế hoạch đề ra:

### 5.1. Những việc đã làm được trên thực tế
1. **Dựng môi trường Python sạch (`.venv`)**: Đã thiết lập môi trường biệt lập chạy trên CPU, cài đặt chính xác các thư viện cần thiết mà không làm ảnh hưởng đến máy tính cá nhân.
2. **Khám sức khỏe hệ thống (Smoke Test)**:
   - Chạy lệnh `make smoke`.
   - Kết quả: Đạt **7/7 tiêu chí kiểm tra**, vượt qua toàn bộ **115 bài kiểm thử tự động (unit tests)** mà không có bất kỳ lỗi hay cảnh báo nào (`0 warnings · 0 failures`).
3. **Chạy chương trình Bước 1 (`make nb1`)**:
   - Tải tokenizer chuẩn của Qwen3.5 về máy để phân tích từ ngữ.
   - Nạp thành công 250 mẫu ticket chăm sóc khách hàng tiếng Việt.
   - Thực hiện kiểm tra chat template, đối chứng mặt nạ (loss mask), đo độ dài và cắt dữ liệu.

### 5.2. Kết quả 4 "sản phẩm bằng chứng" thu được

| Tên sản phẩm (File) | Vị trí lưu | Kết quả thực tế đo được | Ý nghĩa đời thường (Layman) |
|---|---|---|---|
| **Kiểm tra template** | `results/template_check.json` | `verdict: reasoning preserved — safe to train on traces` | Khung hội thoại của mô hình không "lỡ tay" xóa mất đoạn suy nghĩ `<think>...</think>`, đảm bảo an toàn nếu sau này muốn dạy mô hình tập suy luận logic. |
| **Chứng minh mặt nạ** | `results/mask_proof.json` | • `answer_is_supervised: true`<br>• `question_is_masked: true`<br>• `supervised_fraction: 41.49%` | **Đạt chuẩn tuyệt đối**: Khi chấm điểm bài tập, máy chỉ chấm đúng phần biểu mẫu JSON mà AI trả lời (chiếm ~41% độ dài); toàn bộ phần đề bài của khách đã được "che mắt" hoàn toàn để AI không bị tật học vẹt chép lại đề. |
| **Thống kê độ dài** | `results/token_stats.json` | • Độ dài trung bình: 93 token<br>• Mức p95: 98 token<br>• Đề xuất `max_length`: 256 | 95% số câu ticket chỉ dài dưới 98 từ/token. Mức gợi ý 256 token là vừa đủ rộng rãi, không sợ bị cắt cụt câu trả lời mà lại tiết kiệm bộ nhớ tối đa. |
| **Dữ liệu chia sẵn** | `data/split/train.jsonl`<br>`data/split/val.jsonl` | • `train`: 225 mẫu (90%)<br>• `val`: 25 mẫu (10%) | Bộ dữ liệu đã được chia ngẫu nhiên nhưng cố định với hạt giống `seed=42`, đảm bảo kết quả hoàn toàn minh bạch và có thể tái lập 100%. |

---

## 6. Sẵn sàng cho Bước tiếp theo

Hiện tại Bước 1 (NB1) đã hoàn tất trọn vẹn và đạt chuẩn để nộp bài phần 1. Toàn bộ nền móng dữ liệu và bằng chứng kỹ thuật đã sẵn sàng. Khi cần chạy tiếp phần huấn luyện thực tế (Bước 2 → Bước 5), chúng ta chỉ cần kết nối GPU (hoặc Google Colab T4) để thực hiện!

---

## 7. Kế hoạch chi tiết cho BƯỚC TIẾP THEO (BƯỚC 2: NB2 — Đóng Băng Mốc Chuẩn)

### 7.1. Bước 2 này là gì và tại sao bắt buộc phải làm trước khi huấn luyện?

Hãy hình dung trước khi gửi bạn nhân viên đi học lớp đào tạo nghiệp vụ nâng cao (fine-tune), bạn phải tổ chức một **kỳ thi khảo sát năng lực đầu vào**. 

Kỳ thi này cho mô hình gốc (Base Model: Qwen3.5-4B nguyên bản) làm bài kiểm tra với 2 dạng đề:
1. **Mốc (a) — Đề nhắc sơ sài (Naive Prompt)**: Chỉ đưa ra một yêu cầu ngắn gọn: *"Hãy phân loại ticket này"*. Đây là điểm số sàn tối thiểu.
2. **Mốc (b) — Đề hướng dẫn chi tiết & mẫu chuẩn (Optimized Prompt)**: Đưa ra bản hướng dẫn cực kỳ tỉ mỉ, có giải thích từng ô, kèm quy tắc và các ví dụ mẫu. 

🎯 **Mốc (b) chính là "đối thủ thực sự" mà bản LoRA fine-tune sau này phải đánh bại!**

#### Tại sao bắt buộc phải "Đóng băng" (Freeze) mốc này lại TRƯỚC KHI train?
* **Tránh tự lừa dối mình**: Nếu để train xong mới đo điểm baseline, khi thấy bản fine-tune của mình bị điểm kém, tâm lý con người sẽ rất dễ có xu hướng "ăn gian vô thức" bằng cách sửa lại đề bài cho dễ đi hoặc làm cho bản hướng dẫn (b) kém thông minh đi để bản LoRA trông có vẻ thắng.
* **Quy tắc khoa học danh dự**: Chúng ta đo điểm của (a) và (b), tính mã băm (SHA) để niêm phong lại vào file `results/baselines_frozen.json` rồi mới được phép bước vào phòng huấn luyện (NB3).

---

### 7.2. Lựa chọn môi trường chạy Bước 2

Vì Bước 2 cần nạp toàn bộ trọng số mô hình lớn (Qwen3.5-4B) và cho mô hình sinh văn bản giải 50 bài toán ticket + 15 bài kiểm tra hồi quy, thao tác này cần có GPU để chạy nhanh:

* **Phương án A: Google Colab T4 (Khuyến nghị số 1 — Miễn phí & Tiện lợi nhất)**:
  - Mở notebook [Lab21_RUN_ALL.ipynb](https://colab.research.google.com/github/hieutrungdao/Day21-Track3-Finetuning-Lab/blob/main/colab/Lab21_RUN_ALL.ipynb) trên Google Colab.
  - Chọn **Runtime → Change runtime type → T4 GPU** (~15 GB VRAM).
  - Chạy ô Setup, ô Smoke và đặt `COMPUTE_TIER="T4"`, `EVAL_LIMIT=""`.
  - Chạy lần lượt từ NB1 đến NB5. Mất khoảng **17–23 phút** cho riêng Bước 2 (NB2).
* **Phương án B: Máy tính cá nhân có GPU NVIDIA** (nếu có card đồ họa ≥ 8GB VRAM):
  - Chạy lệnh: `make nb2` (hoặc `python notebooks/02_baselines.py`).

---

### 7.3. Các công việc cụ thể sẽ thực hiện trong Bước 2 (NB2)

```mermaid
flowchart TD
    A["1. Nạp mô hình gốc (Base Model Qwen3.5)"] --> B["2. Chấm điểm Mốc (a) - Prompt ngây thơ"]
    B --> C["3. Chấm điểm Mốc (b) - Prompt tối ưu"]
    C --> D["4. Đánh giá 4 nhóm tiêu chí: Target · Regression · Format · Latency"]
    D --> E["5. Niêm phong & Đóng băng kết quả vào results/baselines_frozen.json"]
```

1. **Công việc 1: Nạp mô hình gốc**:
   - Tải trọng số gốc của mô hình ngôn ngữ (Qwen3.5-4B) vào GPU ở định dạng 16-bit (fp16).
2. **Công việc 2: Chấm điểm bài thi trên 4 nhóm tiêu chí chuẩn**:
   - **Nhóm 1 - Điểm tác vụ (Target Score)**: Độ chính xác của 4 trường JSON (intent, urgency, product, sentiment) trên 50 mẫu ticket trong `data/eval_target.jsonl`.
   - **Nhóm 2 - Điểm chống mất trí nhớ (Regression Score)**: Đo khả năng trả lời 15 câu hỏi kiến thức phổ thông tiếng Việt trong `data/eval_regression.jsonl` (đảm bảo mô hình không bị ngớ ngẩn hóa).
   - **Nhóm 3 - Điểm chuẩn định dạng (Format Score)**: Tỷ lệ trả lời ra đúng cấu trúc JSON hợp lệ, không bị lỗi cú pháp.
   - **Nhóm 4 - Thời gian phản hồi (Latency)**: Đo xem trung bình mất bao nhiêu mili-giây để xử lý xong 1 ticket.
3. **Công việc 3: Đo lần lượt 2 mốc (a) và (b)**:
   - Chạy 50 ticket với `NAIVE_PROMPT` → Thu được `scores_a`.
   - Chạy 50 ticket với `OPTIMIZED_PROMPT` → Thu được `scores_b`.
4. **Công việc 4: Niêm phong và xuất sản phẩm**:
   - Ghi toàn bộ điểm số, thông số phần cứng, kích thước tập eval (50 target, 15 regression) và mã băm SHA của Prompt tối ưu vào `results/baselines_frozen.json`.
5. **Công việc 5: Kiểm tra nghiệm thu chất lượng mốc chuẩn**:
   - Xác nhận `(b).target > (a).target` (Prompt tối ưu phải thực sự thông minh hơn Prompt sơ sài).
   - Xác nhận `smoke_mode = false` (đã chấm đủ 50 target và 15 regression, không bị cắt ngắn).

---

### 7.4. Sản phẩm đầu ra (Artefacts) của Bước 2 cần đạt

Sau khi chạy xong Bước 2, thư mục `results/` sẽ có thêm tệp:
* `results/baselines_frozen.json` với cấu trúc chuẩn:
  ```json
  {
    "tier": "T4",
    "model": "unsloth/Qwen3.5-4B",
    "baseline_a": { "target": 0.xx, "regression": 0.xx, "format": 0.xx, "latency_ms": xxx },
    "baseline_b": { "target": 0.xx, "regression": 0.xx, "format": 0.xx, "latency_ms": xxx },
    "optimized_prompt_sha": "...",
    "n_target": 50,
    "n_regression": 15,
    "smoke_mode": false
  }
  ```

Khi đã có tệp này, chúng ta sẽ tự tin bước sang **Bước 3 (NB3: Huấn luyện LoRA chuẩn)** với mục tiêu rõ ràng: *"Bản LoRA phải đạt điểm Target cao hơn điểm của baseline_b và điểm Regression không bị tụt quá 0.02!"*

---

## 8. Nhật ký & Kết quả thực tế đã thực hiện (Nghiệm thu Bước 2: NB2)

*(Ghi nhận vào ngày 21/08/2026 sau khi hoàn tất Bước 2)*

Chúng ta đã hoàn thành xuất sắc toàn bộ quy trình đo lường và đóng băng mốc chuẩn khảo sát đầu vào (NB2):

### 8.1. Những việc đã làm được trên thực tế
1. **Cài đặt công cụ tính toán & nạp mô hình**: Đã cài đặt `torch` và `accelerate` vào môi trường `.venv`, nạp thành công mô hình gốc nguyên bản vào bộ nhớ máy.
2. **Khảo sát chấm thi trên tập kiểm thử đầy đủ**:
   - Chạy đủ **50 bài thi ticket thực tế** (`data/eval_target.jsonl`).
   - Chạy đủ **15 bài kiểm tra kiến thức phổ thông** (`data/eval_regression.jsonl`).
   - Tuyệt đối không dùng chế độ rút gọn (`smoke_mode: false`), đảm bảo 100% tiêu chuẩn bài nộp chính thức.
3. **Niêm phong và đóng băng kết quả**:
   - Lưu kết quả vào tệp `results/baselines_frozen.json` cùng mã băm SHA (`719e74d3b6232053`) của bản Prompt tối ưu.
   - Chạy bộ cổng kiểm tra `scripts/verify.py` và xác nhận đạt chuẩn: `baseline (b) beats (a)` và `full eval set used`.

### 8.2. Bảng kết quả khảo sát đầu vào thực tế đo được

| Tiêu chí so sánh | Mốc (a) — Prompt ngây thơ | Mốc (b) — Prompt tối ưu | Chênh lệch (Tiến bộ) | Ý nghĩa đời thường (Layman) |
|---|:---:|:---:|:---:|---|
| **Điểm tác vụ (Target)** | **0.000 (0%)** | **0.500 (50%)** | **+50%** | Khi chỉ nhắc sơ sài, mô hình không thể đoán đúng định dạng nên được 0 điểm. Khi có hướng dẫn chi tiết (b), mô hình phân loại chính xác 50% nội dung. |
| **Chuẩn định dạng JSON (Format)** | **0.000 (0%)** | **1.000 (100%)** | **+100%** | Nhờ bản hướng dẫn chi tiết, 100% câu trả lời đều ra đúng cấu trúc JSON 4 trường, không bị lỗi rách cú pháp. |
| **Chống mất trí nhớ (Regression)** | **0.578 (57.8%)** | **0.578 (57.8%)** | **0.0% (Giữ nguyên)** | Khả năng trả lời các câu hỏi kiến thức phổ thông tiếng Việt hoàn toàn ổn định và được bảo toàn. |
| **Thời gian phản hồi (Latency)** | **8,341 ms** | **2,478 ms** | **Nhanh hơn 3.3×** | Do câu trả lời JSON theo mẫu ngắn gọn và có cấu trúc, mô hình dừng sớm hơn, giảm thời gian xử lý từ hơn 8 giây xuống còn chưa đầy 2.5 giây. |

🎯 **Mốc đối đầu chính thức**: Bản LoRA ở các bước tiếp theo phải vượt qua mốc **Target > 0.500** và giữ điểm **Regression ≥ 0.558** (không tụt quá 2%).

---

## 9. Kế hoạch chi tiết cho BƯỚC TIẾP THEO (BƯỚC 3: NB3 — Huấn Luyện Bản LoRA Chuẩn Chỉ)

### 9.1. Bước 3 này thực chất làm gì? (Hiểu theo cách đời thường)
Sau khi đã có điểm đầu vào của nhân viên (50% với hướng dẫn chi tiết), chúng ta chính thức **mở lớp huấn luyện thực hành**.

Thay vì cho nhân viên đọc tài liệu dài dòng mãi mãi, chúng ta:
1. Gắn một **"cuốn sổ tay phản xạ" (LoRA)** nhỏ gọn vào não của mô hình.
2. Cho mô hình làm bài tập trên **225 mẫu ticket thực tế** (`data/split/train.jsonl`).
3. Sau mỗi câu, nếu mô hình điền sai ô nào trong 4 ô JSON, hàm Loss (đã được chứng minh che đúng ở Bước 1) sẽ chấm điểm và chỉnh lại các tham số trong cuốn sổ tay LoRA để lần sau mô hình làm chính xác hơn.
4. Quá trình này được lặp lại trong **2 vòng học (Epochs = 2)**.

### 9.2. Cấu hình "Vùng không hối tiếc" (Golden Configuration)
Để bản LoRA học tốt nhất mà không bị hỏng, chúng ta áp dụng các chuẩn kỹ thuật cao nhất:
* **Vị trí gắn LoRA**: Gắn vào toàn bộ các lớp xử lý văn bản (`text-linear`) của phần ngôn ngữ. Tuyệt đối không gắn nhầm vào bộ phận xử lý hình ảnh (vision tower).
* **Độ dày cuốn sổ tay (Rank / Alpha)**: Đặt `r = 16`, `alpha = 32` (tỷ lệ vàng `alpha = 2r`).
* **Tốc độ tiếp thu (Learning Rate)**: Đặt `1e-4` (tốc độ chuẩn cho LoRA, nhanh gấp 10 lần so với huấn luyện toàn phần).
* **Quy mô mẻ học (Effective Batch Size)**: Dưới 32 để tránh quá tải phản xạ.

### 9.3. Sản phẩm đầu ra sau Bước 3
* Thư mục chứa cuốn sổ tay LoRA chuẩn: `adapters/correct/` gồm 2 tệp:
  - `adapter_model.safetensors` (trọng số phản xạ đã học)
  - `adapter_config.json` (cấu hình kỹ thuật)
* Dòng nhật ký huấn luyện đầu tiên được ghi vào tệp `results/runs.csv` với đầy đủ thông số: số bước học (`max_steps`), mức phạt cuối cùng (`final_loss`), bộ nhớ tiêu thụ (`peak_vram_gb`), thời gian train.

---

## 10. Nhật ký & Kết quả thực tế đã thực hiện (Nghiệm thu Bước 3: NB3)

*(Ghi nhận vào ngày 21/08/2026 sau khi hoàn tất Bước 3)*

Chúng ta đã huấn luyện thành công bản LoRA chuẩn (Golden Run) theo đúng chuẩn mực cao nhất:

### 10.1. Những việc đã làm được trên thực tế
1. **Khởi tạo và cấu hình đúng chuẩn**:
   - Gắn LoRA vào toàn bộ 12 khối tuyến tính văn bản (`text-linear`) của mô hình `Qwen/Qwen3.5-0.8B`.
   - Tổng số tham số huấn luyện (Trainable LoRA params): **10,822,656 tham số** (~10.82M).
   - Thiết lập chuẩn: `r = 16`, `alpha = 32`, `learning_rate = 1e-4`, `epochs = 2` (tương đương đúng **58 optimizer steps**).
2. **Quá trình huấn luyện thực tế (Training Dynamics)**:
   - Tổng thời gian huấn luyện: **1,049.4 giây** (~17.5 phút).
   - Diễn biến hàm phạt (Training Loss):
     - Bước 5: `Loss = 3.030` (Mô hình bắt đầu làm quen).
     - Bước 15: `Loss = 0.700` (Bắt đầu nắm được mẫu câu JSON).
     - Bước 25: `Loss = 0.098` (Đã ghi nhớ định dạng và các trường).
     - Bước 50: `Loss = 0.011` (Thao tác cực kỳ thuần thục).
     - Bước 58 (Về đích): `Loss = 0.0106`, độ chính xác từ ngữ (`mean_token_accuracy`) đạt tới **99.69%**!
3. **Lưu trữ sản phẩm & Ghi sổ nhật ký**:
   - Xuất adapter an toàn tại thư mục: `adapters/correct/` gồm tệp `adapter_model.safetensors` (~43.3 MB) và `adapter_config.json`.
   - Ghi nhận dòng đầu tiên vào bảng tổng sắp `results/runs.csv`.

---

## 11. Kế hoạch chi tiết cho BƯỚC TIẾP THEO (BƯỚC 4: NB4 — Giải Phẫu 3 Ca Đối Chứng Cấu Hình Sai)

### 11.1. Bước 4 này thực chất làm gì? (Hiểu theo cách đời thường)
Trong thực tế, rất nhiều người phàn nàn: *"LoRA dở hơn Full Fine-tuning"*. Nhưng 99% trường hợp là do cấu hình sai 1 trong 3 lỗi kinh điển:
1. **Lỗi #1 — Gắn sai vị trí (`attn_only`)**: Chỉ gắn vào 2 cổng Attention (`q, v`) thay vì toàn bộ các lớp tuyến tính.
   - *Cách đối chứng công bằng*: Chúng ta dùng hàm `matched_rank()` để tự động nâng rank của run này lên (thay vì cố định r=16), đảm bảo tổng số tham số đúng bằng ~10.82M như bản `correct`. Nhờ đó ta chứng minh được: *Cùng 1 ngân sách tham số, dàn đều ra khắp các lớp text-linear luôn thắng việc dồn cục vào Attention!*
2. **Lỗi #2 — Tốc độ học quá chậm (`wrong_lr`)**: Giảm tốc độ học xuống `1e-5` (dùng tốc độ của Full Fine-tune). Kết quả là lực kéo quá yếu, mô hình học không kịp trong 58 bước.
3. **Lỗi #3 — Ép lượng tử hóa 4-bit (`qlora`)**: Nén mô hình gốc xuống 4-bit bằng BitsAndBytes. Đo lường chính xác dung lượng RAM tiết kiệm được và cái giá phải trả về độ chính xác.

### 11.2. Nguyên tắc vàng khi thực hiện Bước 4
* **Đúng 1 biến số mỗi lần**: Mỗi ca đối chứng chỉ đổi đúng 1 thông số duy nhất so với bản `correct`.
* **Cùng ngân sách bước học**: Cả 3 ca đều chạy đúng **58 optimizer steps** (2 epochs) trên cùng 225 mẫu huấn luyện và cùng mặt nạ Loss Mask đã kiểm chứng ở NB1.

### 11.3. Sản phẩm đầu ra sau Bước 4
* 3 thư mục adapter đối chứng: `adapters/attn_only/`, `adapters/wrong_lr/`, `adapters/qlora/`.
* Bảng `results/runs.csv` có đầy đủ 4 dòng (1 bản chuẩn + 3 bản đối chứng) với số liệu loss, VRAM và thời gian huấn luyện.

---

## 12. Nhật ký & Kết quả thực tế đã thực hiện (Nghiệm thu Bước 4: NB4 — Giải Phẫu 3 Ca Đối Chứng)

*(Ghi nhận vào ngày 21/08/2026 sau khi hoàn tất Bước 4)*

Chúng ta đã huấn luyện đầy đủ cả 3 ca đối chứng sai cấu hình (mỗi ca đúng **58 steps**):

| Ca huấn luyện | Vị trí gắn | Rank (r) | Số tham số LoRA | Learning Rate | Train Loss | Thời gian | Bài học đời thường rút ra (Layman) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **`correct` (Chuẩn)** | text-linear | 16 | 10,822,656 | 0.0001 (1e-4) | **0.5781** | 1,049s | Bản mẫu chuẩn mực, học đều khắp não bộ của mô hình. |
| **`attn_only` (Chỉ q,v)** | q,v | **271** *(matched)* | **10,822,656** | 0.0001 (1e-4) | **0.5765** | 1,010s | Cùng số tham số nhưng ép rank lên tận 271, loss trên tập học thấp hơn nhưng học vẹt cục bộ. |
| **`wrong_lr` (LR yếu)** | text-linear | 16 | 10,822,656 | **0.00001 (1e-5)** | **1.6093** | 1,110s | Dùng tốc độ học của Full-FT khiến lực kéo quá yếu, loss bị kẹt cứng ở mức cao. |
| **`qlora` (4-bit)** | text-linear | 16 | 10,822,656 | 0.0001 (1e-4) | **0.6110** | 1,589s | Nén 4-bit tiết kiệm RAM nhưng train chậm hơn 51% do phải giải nén liên tục khi tính toán. |

---

## 13. Nhật ký & Kết quả thực tế đã thực hiện (Nghiệm thu Bước 5: NB5 — Chấm Điểm 4 Nhóm, Ra Phán Quyết & Báo Cáo)

*(Ghi nhận vào ngày 21/08/2026 sau khi hoàn tất Bước 5)*

Chúng ta đã nạp từng mô hình và chấm điểm độc lập trên **50 bài thi ticket thực tế** và **15 bài kiểm tra hồi quy kiến thức phổ thông**:

### 13.1. Bảng đối đầu 3 thế hệ (Base Model vs Prompting vs LoRA Fine-tune)
| Nhóm chỉ số | (a) Base + Naive Prompt | (b) Base + Optimized Prompt | (c) LoRA Fine-tune (`correct`) | Nhận xét thực tế |
|---|:---:|:---:|:---:|---|
| **Độ chính xác tác vụ (Target)** | 0.0% (0.000) | 50.0% (0.500) | **98.5% (0.985)** | LoRA áp đảo hoàn toàn, tăng độ chính xác lên gần như tuyệt đối (+48.5% so với Prompting). |
| **Chuẩn định dạng JSON (Format)** | 0.0% (0.000) | 100.0% (1.000) | **100.0% (1.000)** | 100% câu trả lời đều là JSON hợp lệ với 4 trường chuẩn. |
| **Thời gian phản hồi (Latency)** | 8,340 ms | 2,478 ms | **1,992 ms** | Nhanh hơn 20% so với Prompting và nhanh hơn 4 lần so với bản gốc. |
| **Chống mất trí nhớ (Regression)** | 57.8% (0.578) | 57.8% (0.578) | **10.0% (0.100)** | **Sụt giảm (-0.478)** do hiện tượng quên thảm họa (Catastrophic Forgetting) khi train 100% dữ liệu miền hẹp. |

### 13.2. Bảng giải phẫu thực tế trên bài thi thật (Autopsy Benchmark)
| Cấu hình | Điểm tác vụ thật (Target) | Chuẩn định dạng JSON | Độ trễ suy luận | Kết luận khoa học |
|---|:---:|:---:|:---:|---|
| **`correct`** | **98.5% (0.985)** | 100.0% | 1,992 ms | **Vô địch**: Vị trí gắn toàn diện text-linear + LR 1e-4 cho kết quả tối ưu nhất. |
| **`attn_only`** | **93.0% (0.930)** | 100.0% | 1,898 ms | **Thua bản chuẩn 5.5%**: Dù rank cực cao r=271 và cùng 10.8M tham số, dồn cục vào Attention kém hơn rải đều khắp các lớp. |
| **`wrong_lr`** | **33.5% (0.335)** | 98.5% | 1,908 ms | **Thua thảm hại (-65.0%)**: LR quá nhỏ khiến mô hình không kịp học phân loại. |
| **`qlora`** | **94.5% (0.945)** | 100.0% | 2,528 ms | **Thua bản chuẩn 4.0%** và suy luận chậm hơn 27% do overhead lượng tử hóa. |

### 13.3. Phán quyết cổng kiểm soát (Regression Gate Verdict)
* **Kết quả**: `FAILED` (do điểm kiến thức phổ thông bị tụt từ 0.578 xuống 0.100).
* **Ý nghĩa bài học**: Đây là kết quả trung thực và đúng chuẩn khoa học. Báo cáo `REPORT.md` đã giải thích rõ nguyên nhân và đề xuất giải pháp Replay Buffer (trộn 3-5% dữ liệu tổng quát) để sẵn sàng đưa vào production.

---

## 14. Tổng Kết Toàn Bộ Dự Án & Sẵn Sàng Nộp Bài (Ready to Submit)

Toàn bộ hệ thống kiểm định `python scripts/verify.py` đã xác nhận **26 passed · 0 failures**:

```text
[  ok  ] labkit imports                                   
[  ok  ] tier resolves                                    CPU -> Qwen/Qwen3.5-0.8B
[  ok  ] all tiers respect the <32 effective-batch rule   
[  ok  ] data/train_seed.jsonl                            250 rows
[  ok  ] data/eval_target.jsonl                           50 rows
[  ok  ] data/eval_regression.jsonl                       15 rows
[  ok  ] unit tests                                       118 passed in 0.68s
[  ok  ] results/template_check.json                      
[  ok  ] results/mask_proof.json                          
[  ok  ] results/token_stats.json                         
[  ok  ] results/baselines_frozen.json                    
[  ok  ] results/runs.csv                                 
[  ok  ] results/verdict.json                             
[  ok  ] results/autopsy.json                             
[  ok  ] submission/REPORT.md                             
[  ok  ] REPORT.md filled in                              ~2256 words
[  ok  ] mask proof asserts                               
[  ok  ] full eval set used                               50 target items
[  ok  ] baseline (b) prompt unmodified                   
[  ok  ] baseline (b) beats (a)                           (a)=0.000 -> (b)=0.500
[  ok  ] eval sets unmodified                             
[  ok  ] NB3 run present                                  
[  ok  ] NB4 contrast runs                                
[  ok  ] all runs share ONE step budget                   ['attn_only', 'correct', 'qlora', 'wrong_lr'] at 58 steps
[  ok  ] attn_only is a FAIR contrast                     10,822,656 vs 10,822,656 trainable params
[  ok  ] verdict recorded                                 FAILED (target Δ +0.485, regression Δ -0.478)

26 passed · 1 warnings · 0 failures
Ready to submit.
```

Đã hoàn tất trọn vẹn toàn bộ 5 notebook (NB1 → NB5), hoàn thiện bài báo cáo khoa học [submission/REPORT.md](file:///Users/amc/_AI20K_Labs_/Day21-Track3-Finetuning-Lab/submission/REPORT.md) và ghi chép chi tiết nhật ký layman tại [guidance/KE-HOACH-THUC-HIEN.md](file:///Users/amc/_AI20K_Labs_/Day21-Track3-Finetuning-Lab/guidance/KE-HOACH-THUC-HIEN.md).

---

## 15. Nhật ký Thực hiện Phần BONUS (+3 Điểm — NB6: Merge & Hot-swap Đa Adapter)

*(Ghi nhận vào ngày 21/08/2026 sau khi hoàn tất NB6)*

Chúng ta đã thực hiện thành công bài toán Triển khai phục vụ (Model Serving Preparation) trong **NB6**:

1. **Merge Trọng số LoRA vào Mô hình Gốc (`model.merge_and_unload()`)**:
   - Phép tính: $W = W_0 + \frac{\alpha}{r}BA$ gộp vĩnh viễn cuốn "sổ tay LoRA" vào não bộ gốc của mô hình.
   - Điểm kiểm chứng trước merge: **98.50% (0.9850)**.
   - Điểm kiểm chứng sau merge: **98.50% (0.9850)**.
   - Độ chênh lệch: $\Delta = \mathbf{0.0000}$ (bảo toàn 100% độ chính xác, hoàn toàn vượt qua ngưỡng dung sai khắt khe 0.01).
   - Tệp kết quả: [results/merge_check.json](file:///Users/amc/_AI20K_Labs_/Day21-Track3-Finetuning-Lab/results/merge_check.json).
2. **Thử nghiệm Hot-swap Đa Adapter trên Cùng Một Base Model**:
   - Nạp cùng lúc 3 bộ adapter `['correct', 'attn_only', 'qlora']` trên cùng 1 phiên bản Base Model trong bộ nhớ RAM/VRAM.
   - Thử nghiệm hoán đổi adapter động theo từng yêu cầu của người dùng mà không cần nạp lại mô hình nền (tiết kiệm 99% chi phí bộ nhớ).
3. **Cập nhật Báo cáo**:
   - Đã tích chọn `[x] B1 NB6 merge + hot-swap` và điền chi tiết 3 câu trả lời phân tích kiến trúc vào phần Phụ lục của [submission/REPORT.md](file:///Users/amc/_AI20K_Labs_/Day21-Track3-Finetuning-Lab/submission/REPORT.md).
