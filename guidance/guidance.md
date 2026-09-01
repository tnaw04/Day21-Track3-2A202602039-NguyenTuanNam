## 1. Thuật ngữ cần biết

Bản đồ Lab

### Đọc trước khi bắt đầu

240 phút Trung cấp

Bạn không được chấm vì ‘fine-tune đã thắng’. Bạn được chấm vì chứng minh được hai điều: (1) loss chỉ tính trên câu trả lời, không tính trên câu hỏi; (2) bản LoRA có (hoặc không) thắng chính model đó khi đã được prompt tử tế — và bạn phát hiện được nếu nó không thắng. FAILED phân tích đúng vẫn đủ điểm.

#### Bài này đang nói về điều gì?

Supervised Fine-Tuning (SFT): dạy model bắt chước cặp (câu hỏi, câu trả lời) bằng cách tối thiểu hoá loss trên token được gán nhãn.

Loss mask: token nào được tính vào loss, token nào bị bỏ (gán IGNORE_INDEX = -100). Che sai = model học viết lại câu hỏi.

Chat template: cách tokenizer ghép system/user/assistant thành một chuỗi. Một số template xoá khối <think> ngay lúc render.

LoRA: gắn ma trận thấp-hạng vào lớp linear, chỉ train vài triệu tham số thay vì cả tỷ trọng số gốc.

Ba baseline: (a) base + prompt ngây thơ, (b) base + prompt tối ưu, (c) bản fine-tune. Mốc phải vượt là (b), không phải (a).

Phép so sánh công bằng: cùng số step, cùng ngân sách tham số, mỗi lần chỉ đổi một biến.

1. Ticket CSKH tiếng Việt → JSON 4 trường (intent, urgency, product, sentiment) — thang đo khách quan, không cần LLM-as-judge.

2. Nhìn loss mask bằng mắt: giải mã token được supervise, khẳng định câu trả lời nằm trong loss và câu hỏi không nằm trong loss.

3. Đóng băng eval + đo (a) và (b) TRƯỚC khi train, để không vô thức chỉnh prompt cho fine-tune trông thắng.

4. Train cấu hình ‘không hối tiếc’: all-linear text decoder, LR ≈ 10× full-FT, batch hiệu dụng < 32.

5. Chạy 3 đối chứng cùng step: attn_only (rank khớp ngân sách), wrong_lr, qlora — mỗi run chỉ một biến.

6. Phán quyết bốn nhóm (target · regression · format · latency) so với (b). Viết report kể cả ca fine-tune thua.

#### Buổi Lab diễn ra như thế nào?

1. 0–25 phút Học viên
   ##### Chuẩn bị môi trường + smoke test
   Mở Colab T4 (khuyến nghị) hoặc máy cá nhân, clone repo, cài dependency, chạy make smoke / verify --smoke. Xác nhận GPU hiện, COMPUTE_TIER đúng, unit test xanh.

2. 25–50 phút Học viên
   ##### NB1 — Dữ liệu, chat template & loss mask
   Nạp 250 mẫu, kiểm tra template có giữ <think> không, so sánh mask assistant-only vs everything, ghi mask_proof.json + token_stats.json, split seed 42. Không cần GPU.

3. 50–85 phút Học viên
   ##### NB2 — Đóng băng eval và đo baseline (a) (b)
   Tải base model, chấm prompt ngây thơ và prompt tối ưu trên 4 nhóm, lưu baselines_frozen.json. Đọc số (b) và chấp nhận mốc này trước khi train. Cần GPU.

4. 85–130 phút Học viên
   ##### NB3 — Train cấu hình đúng
   Gắn LoRA vào toàn bộ linear của text decoder (không gắn vision tower), train ~2 epoch, lưu adapters/correct/ ngay, ghi dòng correct vào runs.csv.

5. 130–185 phút Học viên
   ##### NB4 — Giải phẫu 3 cấu hình sai
   Chạy attn_only (rank matched), wrong_lr, qlora với cùng số optimizer step. Nếu Colab đứt, resume từ adapter đã lưu — không train lại từ đầu.

6. 185–240 phút Học viên
   ##### NB5 phán quyết + viết REPORT + nộp
   Chấm fine-tune trên 4 nhóm, cổng hồi quy, chấm 3 adapter đối chứng bằng điểm target (không bằng train loss), chọn ≥5 ví dụ có cả ca thua, điền REPORT.md, chạy make verify, nộp bài.

#### Kết thúc bài, bạn có gì?

- Artefact NB1: results/mask_proof.json (hai assert xanh), template_check.json, token_stats.json, data/split/{train,val}.jsonl.

- Artefact NB2: results/baselines_frozen.json với (a), (b) và SHA của OPTIMIZED_PROMPT.

- Artefact NB3–NB4: adapters/correct/ + 3 adapter đối chứng, results/runs.csv đủ 4 dòng cùng max_steps.

- Artefact NB5: results/verdict.json, autopsy.json, qualitative.json.

- submission/REPORT.md điền đủ 7 mục, số liệu khớp results/, ≥2 ca fine-tune thua, kết luận ≥150 từ.

- make verify thoát mã 0 (không FAIL). WARN được phép nếu bạn giải thích trong report.

Chưa cần lo

Không có GPU vẫn làm được NB1 và toàn bộ unit test — đó là phần quyết định kết quả nhiều nhất. Phần train dùng Colab Free T4. Bản fine-tune thua baseline (b) vẫn đủ điểm nếu bạn phân tích thẳng; cái bị trừ là làm yếu prompt (b) hoặc sửa tập eval sau khi thấy kết quả.

Đọc bảng này **trước khi mở notebook**. Mỗi thuật ngữ sẽ xuất hiện lại đúng chỗ bạn phải kiểm tra bằng tay, không chỉ “tin library”.

| Thuật ngữ gốc | Bản chất khái niệm | Minh hoạ trực quan |
| --- | --- | --- |
| **SFT (Supervised Fine-Tuning)** | Không phải “nhồi kiến thức mới vào model”. SFT kéo phân phối token của model sát với cặp (câu hỏi, câu trả lời) bạn đưa. Model vẫn là cùng một mạng; bạn chỉ dịch chuyển hành vi trên miền hẹp. | Giống kèm một nhân viên CSKH mới: không dạy lại tiếng Việt, chỉ dạy “ticket kiểu này thì trả JSON kiểu kia”. |
| **Loss / cross-entropy** | Ở mỗi vị trí, model đoán token tiếp theo. Loss lớn = đoán sai. Huấn luyện = giảm loss trên những token **được phép tính**. | Nếu bạn chấm cả phần đề bài, học sinh sẽ học thuộc đề, không học cách trả lời. |
| **Loss mask (`labels = -100`)** | Token nào `labels = -100` thì **không** vào loss. Token còn lại mới là thứ model bị phạt khi đoán sai. Che sai thì mọi số train/eval phía sau đều vô nghĩa. | NB1 bắt bạn **giải mã** đoạn được supervise: phải thấy JSON trả lời, **không** thấy câu hỏi “Alo shop, mình đặt balo…”. |
| **`MASK_MODE=assistant-only`** | Loss chỉ trên lượt assistant (câu trả lời). Đây là mặc định SFT của lab. | Model học viết JSON triage, không học viết lại ticket. |
| **`MASK_MODE=everything`** | Bug kinh điển: loss tính **cả prompt**. Model học bắt chước câu hỏi. | Triệu chứng sau 3 giờ train: model in lại “Shop ơi, mình đặt…” thay vì JSON. Lab cố ý cho bạn nhìn thấy bug này ở NB1, trước khi tốn GPU. |
| **`MASK_MODE=response-only` / `masked-think`** | Với model có chế độ thinking: chỉ supervise phần sau `</think>`, hoặc loại khối suy luận khỏi loss. Tránh “reasoning-trace collapse”: accuracy tăng nhưng `<think>` rỗng. | Thưởng B3: train hai lần hai mode, so `valid_trace_rate`. Không bắt buộc cho bài core. |
| **Chat template** | Công thức ghép `system` / `user` / `assistant` thành một chuỗi tokenizer hiểu (kèm special token). Bạn không tự nối string. | Một số template **xoá** nội dung `<think>` lúc `apply_chat_template`. Trace trong dataset không bao giờ tới loss, và **không có exception**. NB1 ghi `template_check.json` đúng để bắt việc này. |
| **`max_length`** | Trần số token mỗi mẫu. Đặt theo **p95 đo được**, làm tròn lên luỹ thừa 2 — không đoán. Quá lớn = trả VRAM cho padding; quá nhỏ = cắt mất câu trả lời (phần bạn đang train). | NB1 in `token_stats.json`. Tier T4 đặt 1024. Nếu p95 khác, ghi lý do trong REPORT.md. |
| **LoRA** | Thay vì cập nhật ma trận gốc `W`, học hai ma trận nhỏ `B`, `A` sao cho `ΔW ≈ (α/r)·BA`. Base đóng băng. Adapter chỉ vài chục MB. | Một cửa hàng, nhiều “phong cách CSKH”: mỗi khách hàng một adapter, cùng một model gốc trong VRAM. |
| **Rank `r` và `alpha`** | `r` = bậc xấp xỉ (năng lực so với lượng thông tin trong data, không phải nút “chất lượng”). Lab đặt `alpha = 2r` (với r=16 thì alpha=32). | 250 mẫu triage JSON **không** đủ để r=64 tỏ ra hơn r=16. Thí nghiệm cũ “quét rank” dễ kết luận sai nếu gắn LoRA sai chỗ. |
| **`target_modules` / vị trí gắn** | LoRA gắn vào lớp linear nào. Deck: gắn **toàn bộ linear của text decoder**, không chỉ `q_proj, v_proj`. | So `q,v @ r=16` với `all-linear @ r=16` là so **ngân sách**, không so vị trí. Lab bắt `attn_only` dùng `matched_rank()` để số tham số lệch < 5%. |
| **`text-linear` ≠ `all-linear` của PEFT** | Qwen3.5 có **vision tower**. `all-linear` của PEFT gắn cả encoder ảnh bạn không train → adapter phình, merge sai. | `resolve_target_modules()` chỉ trả lớp text decoder. Nếu adapter “to bất thường”, đây là nguyên nhân. |
| **Learning rate LoRA** | Thang LoRA ≈ **10×** LR full fine-tune. Full-FT 4B ~ `1e-5` → LoRA `1e-4`. Dùng LR full-FT cho LoRA thì loss gần như phẳng. | Run `wrong_lr` ở NB4 chỉ đổi đúng một số này. Nếu không biết LR, bạn sẽ kết luận “LoRA không học được”. |
| **Batch hiệu dụng** | `per_device_batch × grad_accum`. LoRA chịu batch lớn kém hơn full-FT; trần lab là **< 32**. Tier T4: batch 1 × accum 16 = 16. | Tăng batch “cho nhanh” có thể làm LoRA kém đi — không phải lúc nào cũng là free lunch. |
| **bf16 / fp16 / QLoRA** | T4 **không có** bfloat16 (cần Ampere). Lab tự chọn bf16 → fp16 → fp32. QLoRA 4-bit **không** phải mặc định: nhà cung cấp khuyên không dùng QLoRA trên Qwen3.5; lab để bạn **đo** ở run `qlora`. | fp16 cần gradient scaling. Vài step đầu log `grad_norm: nan` là GradScaler đang dò thang — bình thường. `nan` suốt run thì run đã chết. |
| **Baseline (a) (b) (c)** | (a) base + prompt 1 câu. (b) base + prompt đã thiết kế (schema, enum, ví dụ). (c) LoRA fine-tune, thường chỉ cần prompt ngắn vì hành vi đã vào trọng số. | Nếu (b) đã cao sẵn, bài toán **có thể không cần** fine-tune. Đó là kết luận hợp lệ, được chấm đầy đủ. |
| **Đóng băng eval** | Sau NB2, không sửa `eval_target.jsonl`, `eval_regression.jsonl`, hay làm yếu `OPTIMIZED_PROMPT`. Checksum + SHA được `make verify` kiểm. | Sửa đề sau khi biết điểm = tự chấm lại cho mình. Muốn đổi corpus: khai báo `data/CUSTOM_DATASET.md` và chạy lại **cả** pipeline. |
| **Bốn nhóm điểm** | **target**: đúng 4 trường JSON so với nhãn. **regression**: 15 câu kiến thức phổ thông — fine-tune không được quên. **format**: parse được JSON + đủ 4 khoá. **latency**: ms/mẫu, greedy. | Perplexity / train loss **không** phải bằng chứng nộp bài. Có thể thêm, không được thay. |
| **Cổng hồi quy (`regression_gate`)** | PASSED khi **cả hai**: target **thắng** (b), và regression không tụt quá **0.02**. Thua (b) hoặc quên kiến thức phổ thông → FAILED. | FAILED phân tích đúng > PASSED không giải thích được. Đừng nới ngưỡng hay làm yếu (b). |
| **`matched_rank()`** | Tính rank cho `attn_only` sao cho số tham số train ≈ `correct` (lệch < 5%). Trên Qwen3.5-4B thường ra r lớn (cỡ ~90). | Không có bước này, NB4 đo “adapter nào nhiều tham số hơn”, không đo “gắn chỗ nào quan trọng hơn”. |
| **Adapter vs merge** | Adapter: `W` gốc + ΔW lúc suy luận (linh hoạt, hot-swap). Merge: cộng ΔW vào `W` một lần, đồ thị giống base, hết overhead, mất khả năng tháo ra dễ dàng. | NB6 (thưởng): merge xong điểm không được tụt quá 0.01; rồi gắn ≥2 adapter trên cùng một base. |

---

---

## 2. Mục tiêu & đầu ra

Bạn hoàn thành khi **đồng thời** có đủ các bằng chứng sau — không phải khi “loss đã giảm”:

1. **Mask đúng bằng mắt, không bằng niềm tin.** `results/mask_proof.json` có `answer_is_supervised: true`, `question_is_masked: true`, và `supervised_fraction` **< 0.95**. Bạn dán được 3–5 dòng đầu của đoạn được tính loss vào REPORT.md.

2. **Mốc đánh giá bị đóng băng trước khi train.** `results/baselines_frozen.json` có điểm (a) và (b); `(b).target > (a).target`; bạn **không** sửa tập eval sau đó.

3. **Có một bản LoRA “đúng cấu hình” trên đĩa.** Thư mục `adapters/correct/` chứa `adapter_model.safetensors` + `adapter_config.json`; `results/runs.csv` có dòng `correct` với loss, VRAM, `max_steps`.

4. **Ba đối chứng công bằng.** Cùng `max_steps` với `correct`; `attn_only` lệch ngân sách tham số < 5%; mỗi run chỉ một biến (vị trí / LR / 4-bit).

5. **Phán quyết bốn nhóm.** `results/verdict.json` ghi PASSED hoặc FAILED; `results/autopsy.json` xếp 4 run bằng **điểm target**, không bằng `final_loss`; `results/qualitative.json` có cả ca thắng lẫn ca thua.

6. **Report khớp số.** `submission/REPORT.md` đủ 7 mục, không còn placeholder `<điền>`, mọi con số lấy từ `results/`, kết luận ≥150 từ, ≥2 ví dụ fine-tune **thua**.

7. **`python scripts/verify.py` (hoặc `make verify`) thoát mã 0.** FAIL phải sửa; WARN phải giải thích trong report.

> **Câu hỏi lab bắt bạn trả lời (chỉ hai câu, mọi thứ khác phục vụ chúng):**
>
>
>
> 1. Phần được tính loss có đúng là câu trả lời không? *(NB1, chạy được trên CPU)*
>
> 2. Bản fine-tune có thắng base đã prompt tử tế không — và bạn có phát hiện được nếu nó không thắng? *(NB2 đóng băng, NB5 phán quyết)*

**Điểm không nằm ở chỗ fine-tune thắng.** Điểm nằm ở chỗ bạn **biết** nó có thắng hay không.

---

---

## 3. Chuẩn bị

### 3.1. Bạn cần gì

| Hạng mục | Yêu cầu tối thiểu | Ghi chú |
| --- | --- | --- |
| Tài khoản Google | Có, để dùng Colab | **Khuyến nghị cho mọi học viên**, kể cả khi laptop có GPU |
| GPU | Colab Free **T4** (khoảng 14,6 GB khả dụng, không phải 16) | Không GPU: vẫn làm NB1 + test tại chỗ; train trên Colab |
| Trình duyệt | Chrome / Edge | Colab free **chỉ cho một phiên GPU**. Tắt tab Colab GPU khác trước khi bắt đầu |
| Thời gian GPU liên tục | ~95–110 phút core (NB1→NB5) trên T4, lần đầu cộng ~1,5 phút tải ~9,32 GB trọng số | Đừng bắt đầu NB3/NB4 khi còn 20 phút pin / sắp hết giờ Colab |
| Repo | [Day21-Track3-Finetuning-Lab](https://github.com/hieutrungdao/Day21-Track3-Finetuning-Lab) | Lab đi kèm deck `day21-fine-tuning-llms-lora-qlora.tex` |
| Kiến thức nền | Đã nghe qua transformer, token, GPU VRAM | Không cần đã từng fine-tune |
| Bài nộp | URL **GitHub** hoặc **HuggingFace Hub** — bên trong phải có `results/` + `submission/REPORT.md` | Xem mục 6 |

**Không cần:** dataset riêng (corpus mặc định đủ nộp), vLLM/SGLang (Ngày 20), DPO (Ngày 22). HuggingFace token chỉ bắt buộc nếu bạn nộp qua Hub (mục 6.2).

### 3.2. Chọn tier phần cứng

Đổi tier = sửa `COMPUTE_TIER` trong `.env` (máy cá nhân) hoặc ô form trên Colab. Cả lab đọc một nguồn này.

| Bạn có | Điền `COMPUTE_TIER` | Model | Làm được gì |
| --- | --- | --- | --- |
| Không GPU / Mac MPS | `CPU` | Qwen3.5-0.8B | **NB1 + toàn bộ test**. Train thì sang Colab T4 |
| Laptop GPU 8–12 GB (RTX 3060/4060) | `LAPTOP` | Qwen3.5-2B | Tất cả; 4B có thể OOM ở NB4 |
| **Colab Free T4 / GPU 12–22 GB** | **`T4` (mặc định)** | Qwen3.5-4B (`unsloth/Qwen3.5-4B`) | Tất cả. Đây là đường nộp bài chuẩn |
| L4 / A100 / RTX 3090+ (≥22 GB) | `BIGGPU` | Qwen3.5-9B | Tất cả, nhanh hơn |

Chi tiết VRAM và lỗi phần cứng: `HARDWARE-GUIDE.md`.

**Hai cần gạt khi hết giờ** (để mặc định khi **nộp** bài; `results/` ghi lại nếu bạn rút gọn):

```bash
EVAL_LIMIT=8 make pipeline     # ít mẫu eval → phần SINH ngắn lại (không phải bài nộp)
EPOCHS=1     make pipeline     # nửa số step → phần HUẤN LUYỆN ngắn lại (nộp được nếu cả 4 run cùng EPOCHS)
```

`EPOCHS` áp **cả NB3 lẫn NB4**. Không chỉnh một nửa — ba đối chứng chỉ có nghĩa khi cùng số step với `correct`.

### 3.3. Cấu trúc repo bạn sẽ đụng tới

```
Day21-Track3-Finetuning-Lab/
├── notebooks/                 # bản gốc (jupytext py:percent) — chạy bằng python hoặc Jupyter
│   ├── 01_data_and_mask.py    # NB1 CPU
│   ├── 02_baselines.py        # NB2 GPU
│   ├── 03_train_correct.py    # NB3 GPU
│   ├── 04_misconfig_autopsy.py
│   ├── 05_evaluate_and_verdict.py
│   └── 06_merge_and_serve.py  # tuỳ chọn, +3 điểm
├── colab/                     # .ipynb sinh từ notebooks/ — dùng trên Colab
│   └── Lab21_RUN_ALL.ipynb    # 4 ô: setup → smoke → pipeline → verify
├── data/
│   ├── train_seed.jsonl       # 250 ticket CSKH → JSON
│   ├── eval_target.jsonl      # 50 mẫu target (đóng băng)
│   ├── eval_regression.jsonl  # 15 câu phổ thông
│   └── checksums.json         # verify phát hiện nếu bạn sửa eval
├── src/labkit/                # thư viện lab (đừng sửa trừ khi biết vì sao)
├── scripts/verify.py          # cổng trước khi nộp
├── submission/REPORT.md       # TEMPLATE — phải điền, không nộp nguyên mẫu
├── results/                   # lab GHI vào đây; grader đọc cái này
└── adapters/                  # lab GHI adapter vào đây
```

### 3.4. Bài toán bạn đang dạy model

**Đầu vào:** ticket CSKH tiếng Việt, ví dụ:

> Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.

**Đầu ra bắt buộc:** đúng một JSON, đúng 4 khoá, không markdown, không giải thích:

```json
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}
```

| Trường | Tập giá trị |
| --- | --- |
| `intent` | `doi_tra` \| `van_chuyen` \| `hoan_tien` \| `san_pham_loi` \| `hoi_thong_tin` |
| `urgency` | `cao` \| `trung_binh` \| `thap` |
| `sentiment` | `tieu_cuc` \| `trung_tinh` \| `tich_cuc` |
| `product` | tên sản phẩm **xuất hiện trong ticket** (so khớp sau khi bỏ dấu) |

Vì sao chọn bài này: mọi nhóm điểm đều **khách quan** — không cần LLM chấm, không có “điểm cho không”.

---

---

## 4. Thực hành

Làm **đúng thứ tự**. NB2 trước NB3 là chủ ý: đo baseline sau khi train sẽ khiến bạn (vô thức) chỉnh prompt cho tới khi fine-tune thắng.

Có **hai đường**. Chọn một, đi hết. Đừng trộn output Colab với output laptop rồi nộp.

| Đường | Khi nào chọn | Lệnh cốt lõi |
| --- | --- | --- |
| **A. Colab T4** | Mặc định. Windows, không GPU, hoặc muốn ít setup | Mở `Lab21_RUN_ALL.ipynb` → 4 ô |
| **B. Máy cá nhân** | Đã có NVIDIA GPU + CUDA đúng bản torch | `make setup` → `make pipeline` |

Windows PowerShell **không** có `make` mặc định. Nếu không cài Make: dùng Colab, hoặc gọi trực tiếp `python notebooks/01_data_and_mask.py` trong venv.

---

### 4.0. Đường A — Colab (khuyến nghị): từng click

#### Bước A1 — Mở đúng notebook, chọn đúng GPU

1. Mở
    [`colab/Lab21_RUN_ALL.ipynb`](https://colab.research.google.com/github/hieutrungdao/Day21-Track3-Finetuning-Lab/blob/main/colab/Lab21_RUN_ALL.ipynb)

2. Menu **Runtime → Change runtime type → T4 GPU → Save**.

3. Xác nhận góc trên có chữ GPU / T4.

**Cấm:** để runtime CPU rồi chạy ô 3. Sẽ rất chậm hoặc OOM giả (torch CPU).

**Cấm:** mở nhiều tab Colab GPU cùng lúc. Lỗi `Quá nhiều phiên đang hoạt động` → **Runtime → Manage sessions → Terminate** các phiên khác.

#### Bước A2 — Mỗi lần repo đổi: mở LẠI tab, đừng chỉ Reconnect

Colab đọc mã notebook từ GitHub **một lần**, lúc URL được mở. Reconnect / đổi runtime / máy ảo mới **không** nạp lại ô Setup. Ô Setup cũ vẫn in commit mới (vì nó `git pull`) nhưng cài theo **danh sách gói cũ**. Lỗi nổ ~10 phút sau, trong `get_peft_model()`.

**Quy tắc:** repo vừa push / bạn vừa được bảo “kéo bản mới” → đóng tab, mở lại URL, chạy ô 1 từ đầu.

#### Bước A3 — Ô 1: Setup (clone + pip)

Chạy ô **1. Setup**. Đợi pip xong (~1 phút).

**Kết quả mong đợi** (số commit và tên GPU có thể khác):

```
commit : <7 ký tự hex>
GPU    : Tesla T4
VRAM   : 14.6 GB
```

Nếu in `GPU: NONE` → dừng, quay lại A1. Đừng chạy tiếp.

Thư mục làm việc sau ô này là `/content/Day21-Track3-Finetuning-Lab`.

#### Bước A4 — Ô 2: Smoke

Chạy ô **2. Smoke**. Không cần GPU cho bước này.

**Kết quả mong đợi:** các dòng `[ ok ]`, unit tests passed, **0 failures**.

Nếu FAIL `labkit imports` hoặc `data/train_seed.jsonl missing`: ô 1 chưa chạy xong hoặc `chdir` sai. Chạy lại ô 1.

#### Bước A5 — Ô 3: Pipeline NB1 → NB5 — chỉnh form TRƯỚC khi chạy

Ô 3 có form. **Đọc kỹ trước khi bấm chạy:**

| Trường | Giá trị nộp bài | Giá trị chỉ để thử máy |
| --- | --- | --- |
| `COMPUTE_TIER` | `T4` | `T4` |
| `EVAL_LIMIT` | **để trống** `""` | `8` (notebook mặc định sẵn `8` — **đổi thành trống khi nộp**) |
| `STAGES` | `nb1 nb2 nb3 nb4 nb5` | có thể `nb1` rồi `nb2` … nếu resume |

> **Bẫy phổ biến:** form mặc định `EVAL_LIMIT = "8"`. Run đó **không nộp được** như bài đầy đủ (verify/report sẽ lộ smoke mode). Dùng `8` chỉ khi bạn đang kiểm tra pipeline; bài nộp phải để trống rồi chạy lại NB2 + NB5 (và không đụng adapter nếu đã train xong).

Chạy ô 3. **Đừng đóng laptop, đừng để tab sleep.** Ngân sách thật ~80–110 phút.

**Nếu đứt giữa NB4:** không chạy lại từ đầu. Adapter nào đã có `adapters/<tên>/adapter_model.safetensors` sẽ được bỏ qua. Đặt `STAGES` còn lại, ví dụ `nb4 nb5`. Muốn train lại một run: xoá thư mục adapter đó, hoặc `FORCE_RETRAIN=1` / `ONLY=qlora` khi chạy notebook 04.

#### Bước A6 — Ô 4: Gatekeeper

Chạy ô **4** sau khi ô 3 xong. Xem mục 5 để đọc output verify. Colab xoá runtime thì mất file — **copy artefact ra máy hoặc Google Drive trước khi đóng tab**.

Cách gọn: thanh Files bên trái Colab → vào `Day21-Track3-Finetuning-Lab/` → tải từng thư mục `results/`, `adapters/correct/`, `submission/` về máy (chuột phải → Download). Hoặc gắn Drive rồi copy:

```python
from google.colab import drive
drive.mount("/content/drive")
!cp -r /content/Day21-Track3-Finetuning-Lab/results /content/drive/MyDrive/lab21_results
!cp -r /content/Day21-Track3-Finetuning-Lab/adapters/correct /content/drive/MyDrive/lab21_correct
!cp /content/Day21-Track3-Finetuning-Lab/submission/REPORT.md /content/drive/MyDrive/lab21_REPORT.md
```

Viết `REPORT.md` trên máy hoặc trực tiếp trên Colab, rồi **push lên GitHub hoặc HuggingFace Hub** (mục 6). Đừng chỉ để file trên Drive — Drive không phải kênh nộp.

---

### 4.0b. Đường B — Máy cá nhân

Mở terminal tại chỗ bạn muốn để repo.

```bash
git clone https://github.com/hieutrungdao/Day21-Track3-Finetuning-Lab.git
cd Day21-Track3-Finetuning-Lab
cp .env.example .env
```

Mở `.env`, giữ `COMPUTE_TIER=T4` nếu GPU ≥12 GB; đổi `LAPTOP` / `BIGGPU` / `CPU` theo bảng 3.2. **Không commit file `.env`.**

**Không GPU:**

```bash
make setup-cpu
make smoke
make nb1
```

Xong NB1 thì sang Colab để train (đường A từ ô 1), hoặc dừng và làm phần CPU trước giờ lab.

**Có GPU NVIDIA** — cài **torch đúng CUDA của máy trước**, rồi mới `requirements.txt` (xem comment đầu file đó):

```bash
# ví dụ Linux CUDA 12.x — chỉnh index-url theo máy bạn
pip install torch --index-url https://download.pytorch.org/whl/cu124

make setup
make smoke
```

**Kết quả `make smoke` mong đợi:** giống A4 — import OK, data đủ dòng, pytest passed.

Kiểm tra GPU thật sự được PyTorch thấy:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

In `False` mà bạn tưởng có GPU → đang dùng torch bản CPU. Sửa trước khi train (sẽ “chạy” nhiều giờ trên CPU).

Chạy lần lượt hoặc một mạch:

```bash
make nb1          # ~25 giây–2 phút, CPU được
make nb2          # ~10–23 phút T4
make nb3          # ~25 phút T4
make nb4          # ~35 phút T4
make nb5          # ~10–12 phút T4
# hoặc: make pipeline
make verify
```

---

### 4.1. NB1 — Dữ liệu, chat template & mask *(CPU, ~25 giây–2 phút)*

**Mục đích:** nhìn thấy mask, không tin cờ thư viện. Đây là notebook **quyết định kết quả nhiều nhất**.

Mở `notebooks/01_data_and_mask.py` (hoặc `colab/Lab21_01_data_and_mask.ipynb`) và chạy từng ô, **hoặc** `make nb1`.

#### 4.1.1. Nạp corpus

**Kết quả mong đợi:**

```
250 mẫu huấn luyện
```

In ra mẫu đầu: có `instruction`, `input` (ticket), `output` (chuỗi JSON), `label` (object 4 trường).

**Tự kiểm:** mở `data/train_seed.jsonl` bằng editor, đọc 2–3 dòng. Đây là dữ liệu **tổng hợp có kiểm soát**, không phải log CSKH thật.

#### 4.1.2. Tokenizer — chỉ vài MB, không tải trọng số

In `eos_token` (thường là `<|im_end|>` hoặc tương đương Qwen).

#### 4.1.3. Kiểm tra bắt buộc #1 — template có nuốt `<think>` không?

Hàm `data.thinking_survives(tok)` nhét một khối `<think>` giả vào hội thoại, gọi `apply_chat_template`, xem nội dung còn không.

**Kết quả mong đợi:** file `results/template_check.json` được ghi. In `VERDICT:` và một đoạn “chuỗi đã render”.

**Bạn phải làm:** đọc `verdict` và đoạn render. Trả lời trong REPORT mục 1: template **có** hay **không** giữ `<think>`. Nếu không giữ: ghi bạn xử lý thế nào (ví dụ không nhét trace vào data, hoặc không dùng `masked-think` vì template đã xoá).

Không có đáp án “phải là có”. Có đáp án “bạn phải **biết**”.

#### 4.1.4. So sánh hai mode mask — DỪNG LẠI ĐỌC OUTPUT

Lab in hai lần, `assistant-only` và `everything`, mỗi lần có:

- số token supervise / tổng (`supervised x/y (z%)`)

- khối `--- LOSS TÍNH TRÊN ĐOẠN NÀY ---`

**Kết quả mong đợi khi đọc bằng mắt:**

| Mode | Đoạn được tính loss chứa | `supervised_fraction` (thứ tự) |
| --- | --- | --- |
| `assistant-only` | JSON trả lời (và có thể special token assistant) | **thấp**, rõ ràng < 95% |
| `everything` | **Cả câu hỏi** “Alo shop…” lẫn câu trả lời | gần **100%** |

Nếu với `assistant-only` bạn đã thấy nguyên ticket trong khối loss → mask hỏng, **đừng train**. Chạy lại, kiểm tra `MASK_MODE`.

#### 4.1.5. Kiểm tra bắt buộc #2 — `mask_proof.json`

Hai assert trong code:

- câu trả lời (40 ký tự đầu của assistant) **nằm trong** chuỗi supervise

- đoạn câu hỏi (40 ký tự đầu `input`) **không nằm trong** chuỗi supervise

**Kết quả mong đợi:** assert không nổ; `results/mask_proof.json` có dạng:

```json
{
  "mask_mode": "assistant-only",
  "n_supervised": <số>,
  "n_total": <số>,
  "supervised_fraction": 0.xx,
  "answer_is_supervised": true,
  "question_is_masked": true
}
```

Copy `supervised_preview` vào REPORT mục 2 (3–5 dòng đầu).

**Mất trắng tiêu chí 1.1** nếu `supervised_fraction ≥ 0.95` — bạn đang train cả prompt.

#### 4.1.6. `max_length` từ p95, không đoán

**Kết quả mong đợi:** `results/token_stats.json` có `p50`, `p95`, `suggested_max_length`.

Nếu in cảnh báo p95 ≠ `TIER.max_length`: **không tự ý sửa code cho “đẹp”**. Ghi lựa chọn vào REPORT (giữ 1024 vì VRAM T4, hay theo p95).

#### 4.1.7. Split cố định seed 42

**Kết quả mong đợi:**

```
train=225  val=25  -> data/split
```

Hai file `data/split/train.jsonl` và `val.jsonl`. Mọi notebook sau dùng split này. Đổi seed = **không so sánh được** với lần chạy khác.

#### Checkpoint NB1 — chỉ sang NB2 khi đủ

- `results/template_check.json`

- `results/mask_proof.json` — hai cờ `true`, fraction < 0.95

- `results/token_stats.json`

- `data/split/train.jsonl` và `val.jsonl`

---

### 4.2. NB2 — Đóng băng eval & đo (a) (b) *(GPU, ~10–23 phút T4)*

**Mục đích:** chốt mốc **trước** khi bạn có động cơ gian lận vô thức.

Chạy `make nb2` hoặc notebook 02. Lần đầu sẽ tải trọng số (~9 GB với 4B).

#### 4.2.1. Nạp hai tập eval — đừng sửa

- Target: `data/eval_target.jsonl` — **50** ticket (cùng schema train, **không trùng** train — lab đã assert khử nhiễm).

- Regression: `data/eval_regression.jsonl` — **15** câu kiểu “Thủ đô Việt Nam…”, chấm bằng keyword.

Nếu bạn set `EVAL_LIMIT`, console in `⚠ EVAL_LIMIT=… — SMOKE MODE`. File đóng băng sẽ có `"smoke_mode": true`.

#### 4.2.2. Hai prompt — hiểu khác biệt

Nằm trong `src/labkit/generate.py`:

| Baseline | Prompt | Vai trò |
| --- | --- | --- |
| **(a)** | `"Phân loại ticket sau."` | Sàn. Base không được dẫn schema. |
| **(b)** | `OPTIMIZED_PROMPT`: vai trò, schema 4 khoá, enum, 1 ví dụ | **Mốc phải vượt.** Fine-tune mà thua cái này thì chưa đáng ship. |

Bạn **được phép làm mạnh (b)** trước khi đóng băng (prompt engineering tử tế). Bạn **không được làm yếu (b)** để fine-tune trông thắng — `make verify` so SHA với bản shipped và WARN/FAIL theo ngữ cảnh; rubric coi đây là lỗi liêm chính.

#### 4.2.3. Chấm bốn nhóm

Cùng hàm `score_run` cho (a) và (b): sinh target → điểm field-accuracy + format; sinh regression → keyword recall; đo latency.

**Kết quả mong đợi trên console** (số **của bạn** sẽ khác — đừng copy số mẫu):

```
(a) base + naive prompt      target=0.xxx  regression=0.xxx  format=0.xxx  NNNNms
(b) base + optimized prompt  target=0.xxx  regression=0.xxx  format=0.xxx  NNNNms
```

**Đọc số trước khi đi tiếp:**

- **(b) > (a) ở target** → prompt tối ưu thật sự tốt hơn. Ổn.

- **(b) ≈ (a) hoặc (b) < (a)** → “optimized” chưa tối ưu. **Cải thiện prompt (b) ngay bây giờ**, chạy lại NB2, rồi mới train. Nếu không, “thắng” ở NB5 là ảo.

- **(b) đã rất cao** (ví dụ format ~1 và target cao) → có thể **không cần** fine-tune. Vẫn train hết lab để học thí nghiệm; phán quyết FAILED/PASS đều viết được.

#### 4.2.4. Ghi `baselines_frozen.json`

**Kết quả mong đợi:** file có `baseline_a`, `baseline_b`, `optimized_prompt_sha` (16 hex), `n_target`, `n_regression`.

Từ đây: **không** sửa hai file eval, **không** âm thầm làm yếu prompt. Muốn đổi dataset riêng: đọc README mục “Đổi dataset”, thêm `data/CUSTOM_DATASET.md`, chạy lại **cả** NB2→NB5.

#### Checkpoint NB2

- `results/baselines_frozen.json` có cả (a) và (b)

- Bạn đã **đọc và chấp nhận** số (b) — chưa thấy kết quả train

- `(b).target > (a).target` (nếu không: sửa prompt, chạy lại NB2)

---

### 4.3. NB3 — Train cấu hình đúng *(GPU, ~25 phút T4)*

**Mục đích:** một run trong “vùng không hối tiếc” (deck §10), không phải run đẹp nhất mọi thời đại.

| Nút | Giá trị lab | Vì sao |
| --- | --- | --- |
| Vị trí LoRA | Toàn bộ linear **text decoder** (`text-linear`) | Chỉ gắn q,v là Lỗi #1 |
| `r` / `alpha` | 16 / 32 | `alpha = 2r` |
| LR | `1e-4` (10× full-FT `1e-5`) | Lỗi #2 = dùng `1e-5` cho LoRA |
| Batch hiệu dụng | 16 trên T4 | Trần < 32 |
| `packing` | **Tắt** | Data đã tokenize + mask sẵn; packing phá căn chỉnh nhãn |
| `padding_free` | Không trên T4 | Cần FlashAttention (Ampere+); T4 là Turing; batch=1 cũng chẳng có padding để bỏ |
| Độ chính xác | Tự chọn: T4 → **fp16** + GradScaler | Không hardcode `bf16=True` |

Ô đầu in `device.banner()` — **đọc dòng precision**. Nếu bạn thấy bf16 trên T4 thì có gì đó sai.

#### 4.3.1. Nhìn kiến trúc thật

`layer_type_summary` in xen kẽ linear-attention / full-attention nếu model 2026 có. Đây không phải slide: đó là `model.config`.

#### 4.3.2. Đếm tham số — chỉ text, không vision

**Kết quả mong đợi:** in `placement=text-linear`, danh sách module, `trainable LoRA params ≈ X.XX M` (vài triệu, không phải cả tỷ).

Nếu số tham số lớn bất thường (cỡ full model): đang gắn nhầm vision tower. Dừng, kiểm tra `resolve_target_modules`.

#### 4.3.3. Dataset đã mask từ NB1

**Kết quả mong đợi:** `assert` `0 < supervised < total` xanh; in `mask_mode = assistant-only` và tỷ lệ token supervise.

Nếu assert nổ: **chưa có `data/split/`** hoặc mask hỏng → về NB1.

Ghi chú kỹ thuật (đọc, không cần sửa): cờ TRL `assistant_only_loss` dựa trên marker `{% generation %}` trong template. Qwen3.5 **không có** marker đó → cờ đó có thể supervise **0 token**, chỉ warning, loss curve vẫn “đẹp”, run vô giá trị. Lab **không** dựa cờ đó; lab dùng mask đã chứng minh ở NB1. (Tuỳ chọn: `python scripts/check_mask_agreement.py`.)

#### 4.3.4. Train rồi LƯU NGAY

In số epoch và số optimizer step. **Ghi số step lại** — NB4 phải trùng.

Sau `trainer.train()`:

**Kết quả mong đợi:**

- Thư mục `adapters/correct/` có `adapter_model.safetensors`, `adapter_config.json`

- `results/runs.csv` có **một dòng** `run=correct` với `final_loss`, `peak_vram_gb`, `max_steps`

Eval có thể OOM; **adapter đã trên đĩa thì không mất**. Đừng eval trước khi save — notebook đã save trước.

`grad_norm: nan` vài step đầu trên fp16: bình thường (GradScaler). Nan cả run: hỏng, xem `scripts/probe_precision.py` / SIMULATION-FINDINGS F-23.

#### Checkpoint NB3

- `adapters/correct/adapter_model.safetensors` tồn tại

- `runs.csv` có dòng `correct`

- Đã ghi loss cuối và peak VRAM vào nháp REPORT mục 4

---

### 4.4. NB4 — Giải phẫu cấu hình sai *(GPU, ~35 phút T4)*

Đây là phần **dễ mất điểm nhất** nếu so sánh không công bằng.

Ba run, **cùng `max_steps` với NB3**, mỗi run **một** biến:

| Run | Đổi đúng một thứ | Kỳ vọng định tính (kiểm bằng số của bạn, không chép) |
| --- | --- | --- |
| `attn_only` | Chỉ `q,v` nhưng **r được nâng** cho bằng số tham số `correct` | Đo *vị trí*, không đo *ngân sách*. Thắng/thua/hoà trên **target ở NB5** đều được điểm nếu lập luận khớp số |
| `wrong_lr` | LR = `1e-5` (thang full-FT) | Loss thường gần phẳng / giảm rất chậm |
| `qlora` | 4-bit thay 16-bit | VRAM thấp hơn; chất lượng ? — bạn đo, đối chiếu khuyến nghị “đừng QLoRA Qwen3.5” |

Trước khi train, notebook in bảng vị trí × rank × số tham số. **Đọc bảng đó.** Cùng r=16, `q,v` ít tham số hơn `text-linear` rất nhiều — đó là lý do phải `matched_rank()`.

**Resume khi Colab đứt:**

```text
skip attn_only: adapters/attn_only/ already trained
```

là đúng. Xoá thư mục adapter nếu muốn train lại đúng run đó.

```bash
# máy cá nhân
ONLY=qlora make nb4
FORCE_RETRAIN=1 make nb4    # train lại cả 3
```

**macOS:** `bitsandbytes` thường không cài được → bỏ qua `qlora`, hai đối chứng kia vẫn chạy; ghi trong report.

**Kết quả mong đợi:** `runs.csv` có đủ 4 tên `correct`, `attn_only`, `wrong_lr`, `qlora` (hoặc thiếu `qlora` nếu không có CUDA 4-bit — WARN). Cột `trainable_params` của `attn_only` lệch < 5% so với `correct`. Cột `max_steps` **trùng nhau**.

**Cấm:** xếp hạng 4 run bằng `final_loss` của NB4. Đó là **Lỗi #3** (chấm bằng chỉ số thay thế). Câu trả lời cho 4.1–4.3 trong REPORT lấy từ **bảng target ở NB5 §4** (`autopsy.json`). Nếu thứ tự loss ≠ thứ tự target, **nói thẳng** — đó có thể là kết quả đáng giá nhất lab.

#### Checkpoint NB4

- Ba thư mục `adapters/attn_only`, `wrong_lr`, `qlora` (trừ khi khai báo không chạy được qlora)

- `runs.csv` cùng `max_steps`

- Đã ghi r matched của `attn_only` (sẽ là số lớn hơn 16)

---

### 4.5. NB5 — Bốn nhóm, phán quyết, định tính *(GPU, ~10–12 phút T4)*

Chạy `make nb5`. Fine-tune được chấm với **`NAIVE_PROMPT`**, không phải prompt dài — đúng ý fine-tune: hành vi đã vào trọng số.

Nếu lỗi `eval slice mismatch`: `EVAL_LIMIT` của NB5 khác NB2. Để **cùng giá trị** (cả hai trống, hoặc cả hai `8`).

#### 4.5.1. Bảng ba baseline

In markdown: (a), (b), (c) × target / regression / format / latency.

Ghi nguyên bảng vào REPORT mục 3. **Số phải khớp** `verdict.json` / `baselines_frozen.json`.

#### 4.5.2. Cổng hồi quy

```
PASSED
 - beat the optimized-prompt baseline by +0.xxx with +0.xxx general-capability change.
```

hoặc

```
FAILED
 - target task did not beat the optimized-prompt baseline (...)
 - general capability regressed by 0.xxx (tolerance 0.020). ...
```

File: `results/verdict.json`.

**Nếu FAILED — đừng sửa eval.** Thứ tự chẩn đoán:

1. `format` thấp → template/mask (NB1), không phải “cần rank lớn hơn”

2. `regression` tụt → quên thảm hoạ → trộn 1–5% dữ liệu phổ thông (deck §14.3); với bài nộp core, **phân tích** hiện tượng vẫn được điểm

3. `target` không nhúc nhích → xem `wrong_lr` trước khi đụng rank

4. Cả ba ổn nhưng vẫn thua (b) → prompt engineering đã thắng. Kết luận hợp lệ: **không nên deploy** bản fine-tune này

Viết diễn giải ≥100 từ trong REPORT mục 5.

#### 4.5.3. Autopsy — chấm 3 cấu hình sai trên **cùng** thang target

Sinh `results/autopsy.json`. Đây là bảng điền cột **target (NB5 §4)** trong REPORT mục 4.

Trả lời 4.1, 4.2, 4.3 **mỗi câu ≥3 câu văn**, bám số:

- **4.1** Vị trí vs rank: `attn_only` cùng số param — thắng/thua/hoà trên target? Thứ tự có giống train loss không?

- **4.2** `wrong_lr`: đường loss khác gì; kết luận sai nếu không biết LR?

- **4.3** `qlora`: tiết kiệm bao nhiêu GB VRAM, trả giá bằng điểm nào; có ủng hộ khuyến nghị nhà cung cấp không?

#### 4.5.4. Định tính — bắt buộc có ca THUA

Notebook in 3 ca tệ nhất và 3 ca tốt nhất. File `results/qualitative.json`.

Trong REPORT mục 6: **≥5 dòng**, trong đó **≥2 fine-tune thua** so với nhãn hoặc so với (b). Chỉ chọn ca thắng = cherry-pick, **mất hết** tiêu chí 3.4.

Gợi ý nhận xét: nhầm `intent` gần nhau (`doi_tra` vs `hoan_tien`), `product` cắt thiếu từ, JSON bọc markdown, urgency bị phóng đại.

#### Checkpoint NB5

- `results/verdict.json`

- `results/autopsy.json`

- `results/qualitative.json`

- Đã chọn ≥2 ca thua cho report

---

### 4.6. Viết `submission/REPORT.md`

Mở đúng file mẫu `submission/REPORT.md`. **Không** xoá 7 mục. Thay mọi `<điền>`, `<paste>`, `<0.xx>`.

| Mục | Lấy số từ | Bẫy |
| --- | --- | --- |
| 1 Setup | `.env`, `token_stats.json`, `template_check.json` | `max_length` phải nói p95 |
| 2 Mask | `mask_proof.json` | Dán preview, không viết “mask đúng” suông |
| 3 Ba baseline | `baselines_frozen.json` + `verdict.json` comparison | Khai nếu đã sửa `OPTIMIZED_PROMPT` — mạnh lên hay yếu đi |
| 4 Autopsy | `runs.csv` + **`autopsy.json`** | Xếp hạng bằng target, không bằng train loss |
| 5 Phán quyết | `verdict.json` | FAILED được điểm nếu phân tích |
| 6 Định tính | `qualitative.json` + tự đối chiếu (b) | ≥2 ca thua |
| 7 Kết luận | lập luận nhân quả ≥150 từ | Không generic (“em học được LoRA rất mạnh”) |

`make verify` đếm placeholder; còn `<điền>` → FAIL. Dưới ~400 từ → WARN.

`submission/REFLECTION.md` là tuỳ chọn phản tư 5 câu — nên làm vì tiêu chí 4.4 “điều tôi học được” cần cụ thể.

---

### 4.7. Thưởng (sau khi verify core xanh)

Không bắt buộc. Chi tiết: `BONUS-CHALLENGE.md`. Đánh dấu phụ lục REPORT.

| Mã | Việc | Gợi ý lệnh / artefact |
| --- | --- | --- |
| B1 +3 | NB6 merge + điểm không tụt >0.01 + hot-swap ≥2 adapter | `make nb6` → `results/merge_check.json` |
| B2 +3 | Dataset miền riêng ≥200 mẫu + khử nhiễm | `data/CUSTOM_DATASET.md`; chạy lại hết pipeline |
| B3 +4 | Reasoning-trace collapse | `MASK_MODE=assistant-only` rồi `response-only`, so `valid_trace_rate` |
| B4 +3 | Quét rank r∈{8,16,64}, **cố định** `text-linear` | Trả lời khi nào rank mới là đòn bẩy |
| B5 +2 | Push adapter lên HF Hub công khai | Tự có nếu nộp kênh HuggingFace; nếu nộp GitHub thì vẫn push thêm adapter lên Hub + link trong REPORT |

---

---

## 5. Kiểm tra kết quả

### 5.1. Lệnh bắt buộc trước khi nộp

```bash
make verify
# hoặc: python scripts/verify.py
```

**Kết quả mong đợi khi sẵn sàng nộp:**

```
[  ok  ] labkit imports
[  ok  ] ...
[  ok  ] results/verdict.json
...
N passed · 0 warnings · 0 failures

Ready to submit.
```

Có WARN vẫn có thể nộp nếu bạn **giải thích** (ví dụ đổi prompt (b) cho mạnh hơn, custom dataset đã có `CUSTOM_DATASET.md`). Có **FAIL** thì **chưa nộp**.

Verify kiểm tra những gì grader quan tâm:

| Kiểm tra | Ý nghĩa |
| --- | --- |
| Hai assert mask xanh, fraction < 0.95 | Pipeline sai từ gốc thì mọi số sau vô nghĩa |
| `attn_only` lệch param < 5% | Đối chứng vị trí, không phải ngân sách |
| Cùng `max_steps` bốn run | Đối chứng cấu hình, không phải “train lâu hơn” |
| Checksum eval | Không sửa đề sau khi biết điểm |
| SHA prompt (b) | Không làm yếu (b) |
| `(b) > (a)` | Prompt “tối ưu” phải tối ưu thật |
| REPORT không còn placeholder | Không nộp template trống |

### 5.2. Checklist tự đọc file (không chỉ tin verify)

Mở từng file, đối chiếu REPORT:

```text
results/mask_proof.json
results/template_check.json
results/token_stats.json
results/baselines_frozen.json
results/runs.csv
results/verdict.json
results/autopsy.json
results/qualitative.json
adapters/correct/adapter_model.safetensors
submission/REPORT.md
```

### 5.3. Lỗi thường gặp

| Triệu chứng | Nguyên nhân | Việc làm |
| --- | --- | --- |
| Model viết lại câu hỏi lúc suy luận | `MASK_MODE=everything` hoặc mask proof đỏ | Về NB1, `assistant-only`, train lại |
| `supervised_fraction ≥ 0.95` | Loss cả prompt | Như trên; đừng nộp |
| Loss phẳng từ step 0 | LR thang full-FT | Đúng bài học `wrong_lr`; run `correct` phải là `1e-4` |
| `format` ~0 nhưng target khá | Hai scorer từng lệch — bản lab hiện tại đã cùng parser; nếu vẫn gặp: model bọc text trước JSON | In raw output; kiểm tra template/EOS |
| `regression` tụt mạnh | Quên thảm hoạ (catastrophic forgetting) | Phân tích trong verdict; tuỳ chọn thêm 1–5% replay |
| OOM ngay NB3 | Tier lớn hơn GPU | Hạ `LAPTOP` hoặc `max_length`; T4 đừng >1024 |
| OOM run thứ 2 NB4 | Model cũ chưa giải phóng | Notebook đã gọi `free_memory()`; nếu tự sửa code, nhớ `del` + empty_cache |
| `Your setup doesn't support bf16` | GPU pre-Ampere | Lab tự chuyển fp16; đừng hardcode bf16 |
| Rất chậm dù “có GPU” | torch CPU | `torch.cuda.is_available()` |
| Colab đứt NB4 | Idle / hết phiên free | Resume; đừng `FORCE_RETRAIN` nếu adapter đã đủ |
| `get_peft_model()` nổ sau 10 phút | Ô Setup Colab cũ, thiếu pin `torchao` | Mở **lại tab** notebook từ GitHub, chạy ô 1 mới |
| `make verify` FAIL checksum | Sửa `eval_*.jsonl` | Khôi phục git; hoặc khai `CUSTOM_DATASET.md` và chạy lại toàn bộ |
| `attn_only is a FAIR contrast` FAIL | Không dùng `matched_rank`, hoặc train lệch | Chạy lại NB4 từ notebook lab, đừng hardcode r=16 cho attn_only |
| Hai cột xếp hạng khác nhau | Train loss ≠ target | **Đúng bài.** Viết vào 4.1, xếp hạng theo target |
| Placeholder còn trong REPORT | Copy mẫu chưa điền | Search `<điền>` `<paste>` `<0.xx>` |
| Windows: `make` không chạy | Không có GNU Make | Dùng Colab, hoặc `python scripts/verify.py` + `python notebooks/0X_....py` |

### 5.4. Hết giờ trên lớp — ưu tiên điểm

Thứ tự: **NB1 → NB2 → NB3 → NB5**, rồi NB4 nếu còn GPU. NB1 và NB5 mang nhiều điểm nhất. NB6 bỏ nếu thiếu thời gian.

---

---

## 6. Nộp bài

`requiresSubmission: true`. Bạn nộp **một URL**: repo **GitHub** *hoặc* model repo **HuggingFace Hub**. Không nộp thư mục máy, không nộp Drive.

Grader **đọc `results/`** và đối chiếu REPORT. Thiếu `results/` thì không chấm được, kể cả khi adapter đã lên Hub.

Chạy `make verify` **trước** khi push. Dán URL vào form / LMS theo hướng dẫn giảng viên.

### 6.1. Artefact bắt buộc (cả hai kênh)

Bên trong repo GitHub hoặc model repo HuggingFace phải có:

- `submission/REPORT.md` đã điền (trên Hub có thể đặt thêm nội dung vào model card / `README.md`, nhưng **vẫn giữ file này** để khớp rubric)

- Toàn bộ JSON + `runs.csv` trong `results/` (mục 5.2; thiếu file → verify FAIL)

- Số trong report **khớp** file (tiêu chí 4.3)

Nên có thêm `adapters/correct/` (`adapter_model.safetensors` + `adapter_config.json`). Không bắt buộc push `attn_only` / `wrong_lr` / `qlora` — nặng, grader không cần. Không push `.venv/`, checkpoint optimizer, `gguf/`, file `.env` có token, trọng số **base model**.

Repo **public** (hoặc private nhưng đã mời tài khoản chấm). Ghi MSSV + họ tên ở đầu `REPORT.md`.

### 6.2. Chọn một kênh: GitHub hoặc HuggingFace

#### Kênh GitHub

1. Fork [repo lab](https://github.com/hieutrungdao/Day21-Track3-Finetuning-Lab) **hoặc** tạo repo mới tên `lab21-<MSSV>`.

2. Đưa artefact vào đúng path:

```
lab21-<MSSV>/
├── submission/REPORT.md
├── results/                 ← TẤT CẢ json + runs.csv
├── adapters/correct/        ← khuyến nghị
├── notebooks/               ← .py hoặc .ipynb đã clear output
└── LINKS.md                 ← tuỳ chọn: URL HuggingFace nếu bạn cũng đẩy adapter
```

3. Adapter LoRA thường vài chục MB — GitHub chấp nhận. Nếu >100 MB, bật [Git LFS](https://git-lfs.com/) cho `*.safetensors`, hoặc đẩy adapter sang HuggingFace rồi chỉ để link trong `REPORT.md` / `LINKS.md`.

4. Commit, push, kiểm tra trên web: mở được `results/verdict.json` và `submission/REPORT.md`.

5. Nộp URL dạng `https://github.com/<user>/lab21-<MSSV>`.

Trên Colab, sau khi điền report (thay email, user, MSSV; **không** commit token):

```python
%cd /content/Day21-Track3-Finetuning-Lab
!git config user.email "ban@example.com"
!git config user.name "Ten Ban"
!git remote add mine https://github.com/<user>/lab21-<MSSV>.git
!git add submission/REPORT.md results adapters/correct
!git status    # xem lại: không có .env, không có .venv
!git commit -m "Lab 21 submission"
!git push mine HEAD:main
```

#### Kênh HuggingFace Hub

1. Tạo tài khoản [huggingface.co](https://huggingface.co/) → Settings → Access Tokens → token **Write**.

2. Đăng nhập trên máy hoặc Colab:

```bash
pip install -U huggingface_hub
huggingface-cli login
```

3. Tạo model repo public, ví dụ `<user>/lab21-<MSSV>-qwen35-triage-vi`.

4. Đẩy adapter **và** artefact (từ thư mục lab, sau `make verify`):

```python
from huggingface_hub import HfApi, login
login()  # hoặc biến môi trường HF_TOKEN
api = HfApi()
repo_id = "<user>/lab21-<MSSV>-qwen35-triage-vi"
api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
api.upload_folder(folder_path="adapters/correct", repo_id=repo_id, repo_type="model")
api.upload_folder(folder_path="results", path_in_repo="results", repo_id=repo_id, repo_type="model")
api.upload_file(path_or_fileobj="submission/REPORT.md", path_in_repo="submission/REPORT.md",
                repo_id=repo_id, repo_type="model")
```

Hoặc, nếu adapter đang load bằng PEFT:

```python
model.push_to_hub("<user>/lab21-<MSSV>-qwen35-triage-vi")
```

Sau `push_to_hub` vẫn phải **upload thêm** `results/` và `submission/REPORT.md` — chỉ có adapter thì không chấm được.

5. Copy report (hoặc tóm tắt + đường dẫn `submission/REPORT.md`) vào model card. Ghi rõ base model (`unsloth/Qwen3.5-4B` hoặc đúng tier bạn chạy).

6. Nộp URL dạng `https://huggingface.co/<user>/lab21-<MSSV>-qwen35-triage-vi`.

Nộp HuggingFace đúng artefact **đồng thời được tính thưởng B5 (+2)** — ghi link trong phụ lục REPORT.

Có thể dùng **cả hai**: GitHub cho code + `results/` + report, Hub cho adapter; khi đó nộp URL GitHub và ghi URL Hub trong `REPORT.md` / `LINKS.md`.

### 6.3. Thang điểm (để bạn tự đối chiếu, không phải để “tối ưu cóc”)

100 điểm + tối đa 15 thưởng. Chi tiết `rubric.md`.

| Khối | Điểm | Bạn chứng minh bằng |
| --- | --- | --- |
| Pipeline đúng | 30 | mask proof, template, p95, adapter `correct` |
| Thí nghiệm công bằng | 25 | matched rank, cùng step, một biến, phân tích vị trí vs rank, xếp theo target |
| Đánh giá & phán quyết | 25 | (b) đo trước và hơn (a), 4 nhóm, diễn giải verdict, ≥2 ca thua |
| Report | 20 | đủ 7 mục, kết luận ≥150 từ, số khớp, phản tư cụ thể |

### 6.4. Liêm chính (đọc một lần)

Được: làm **mạnh** prompt (b); đổi corpus nếu khai báo và chạy lại hết; kết luận “không nên fine-tune”.

Không được: làm yếu (b); sửa eval sau khi thấy điểm; xếp hạng bằng perplexity/train loss rồi viết như thể đó là năng lực tác vụ; chỉ trưng ca thắng.

---

---

## Phụ lục nhanh — bản đồ thời gian T4 (đo trên lab, cận trên)

| Notebook | Thời gian | GPU |
| --- | --- | --- |
| NB1 | ~25 giây–2 phút | Không |
| NB2 | ~10–23 phút | Có (sinh văn bản 2 lần) |
| NB3 | ~25 phút | Có |
| NB4 | ~35 phút | Có (3 run) |
| NB5 | ~10–12 phút | Có (sinh thêm) |
| Core | **~80–110 phút** | Có |
| NB6 | ~10 phút | Có, tuỳ chọn |
| Viết report | ~30 phút | Không |

Sinh văn bản chiếm phần lớn: eval được sinh **ba lần** — (a), (b), rồi fine-tune. Đó là giá của thiết kế ba-baseline; lab cho rằng nó đáng.

#### Góp ý cho buổi Lab

Không bắt buộc và không ảnh hưởng việc nộp bài. Giảng viên chỉ xem phản hồi ẩn danh.

[Góp ý bài Lab](https://codelabs.vlearn.dev/feedback?subject=lab&lab=k3-fine-tune-llm-bang-lora-chung-minh-mask-dung-roi-moi-phan-quyet-thang-thua)

#### Nộp bài và đánh giá Lab

Dán link GitHub, Drive hoặc LMS của bài đã nộp. Điểm và nhận xét sẽ không hiển thị tại đây.

Đang tải trạng thái bài nộp…
