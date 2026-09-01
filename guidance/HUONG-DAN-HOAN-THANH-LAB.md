# Hướng dẫn hoàn thành Lab 21 — Fine-tuning LLM bằng LoRA

Tài liệu này là lộ trình thực hành ngắn gọn, bám theo code và rubric hiện tại của
project. Hãy đọc thêm [guidance.md](guidance.md) nếu cần giải thích khái niệm chi tiết,
[rubric.md](../rubric.md) để tự chấm điểm và [HARDWARE-GUIDE.md](../HARDWARE-GUIDE.md)
khi gặp lỗi GPU.

## 1. Hiểu đúng mục tiêu của lab

Lab không chấm việc fine-tune nhất định phải thắng. Lab chấm việc bạn chứng minh được:

1. Loss chỉ tính trên câu trả lời của assistant, không tính trên ticket của người dùng.
2. Bản LoRA có hoặc không thắng chính base model khi base model được dùng với một
   prompt tốt.
3. Các run đối chứng là công bằng: cùng số optimizer step, gần bằng ngân sách tham số
   và mỗi run chỉ đổi một biến.
4. Kết luận dựa trên điểm tác vụ, hồi quy, định dạng và độ trễ; không dựa riêng vào
   training loss.

Vì vậy, verdict `FAILED` vẫn là một kết quả hợp lệ và có thể đạt điểm đầy đủ nếu số đo
đúng và phần phân tích trung thực.

## 2. Khi nào được xem là hoàn thành

Trước khi nộp, tối thiểu phải có:

```text
results/
├── template_check.json
├── mask_proof.json
├── token_stats.json
├── baselines_frozen.json
├── runs.csv
├── verdict.json
├── autopsy.json
└── qualitative.json

adapters/correct/
├── adapter_config.json
└── adapter_model.safetensors

submission/REPORT.md
```

Ngoài ra:

- `runs.csv` phải có bốn run: `correct`, `attn_only`, `wrong_lr`, `qlora`.
- Bốn run phải có cùng `max_steps`.
- Số tham số của `attn_only` phải lệch dưới 5% so với `correct`.
- `REPORT.md` phải đủ 7 mục, không còn placeholder, có ít nhất 5 ví dụ định tính và
  ít nhất 2 ca fine-tune thua.
- Lệnh verify phải thoát với mã 0; mọi `WARN` phải được đọc và giải thích nếu liên quan.

## 3. Chọn cách chạy

### Cách khuyến nghị: Colab Free T4

Đây là đường ít lỗi môi trường nhất và chạy được cả QLoRA contrast. Ngân sách đo thật
cho NB1–NB5 trên T4 là khoảng **100–130 phút**, chưa tính khoảng 30 phút viết report.
Một số comment cũ trong `Makefile` hoặc tài liệu cũ còn ghi ~80 phút; không nên lập kế
hoạch theo con số đó.

Mở notebook:

[Lab21_RUN_ALL.ipynb trên Colab](https://colab.research.google.com/github/hieutrungdao/Day21-Track3-Finetuning-Lab/blob/main/colab/Lab21_RUN_ALL.ipynb)

Sau đó:

1. Chọn **Runtime → Change runtime type → T4 GPU**.
2. Chạy ô Setup và xác nhận output có `Tesla T4`, VRAM khoảng `14.6 GB`.
3. Chạy ô Smoke; phải có 0 failure.
4. Trước lần chạy chính thức, đặt:

   ```text
   COMPUTE_TIER = "T4"
   EVAL_LIMIT   = ""
   STAGES       = "nb1 nb2 nb3 nb4 nb5"
   ```

5. Chạy pipeline và giữ tab hoạt động.
6. Chạy ô verify sau khi pipeline xong.
7. Tải hoặc push artefact trước khi đóng runtime; file trong `/content` sẽ mất khi VM
   Colab bị thu hồi.

Lưu ý quan trọng: `EVAL_LIMIT` trong notebook đang mặc định là `"8"`. Giá trị này chỉ
dùng để thử pipeline. Một run có `EVAL_LIMIT=8` sẽ ghi `smoke_mode=true` và bị full
verify từ chối.

Mỗi khi repo vừa được cập nhật, hãy đóng và mở lại tab Colab từ URL trên. Reconnect hoặc
đổi runtime không làm Colab đọc lại mã notebook mới.

### Máy cá nhân có NVIDIA GPU

Chạy từ thư mục gốc project:

```bash
cp .env.example .env
# Sửa COMPUTE_TIER trong .env cho đúng phần cứng.

# Cài PyTorch đúng CUDA của máy trước, rồi:
make setup
make smoke
make pipeline
make verify
```

Kiểm tra PyTorch thật sự thấy GPU trước khi train:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

Nếu không có GPU, vẫn có thể hoàn thành NB1 và test tại chỗ:

```bash
cp .env.example .env
# Đặt COMPUTE_TIER=CPU trong .env.
make setup-cpu
make smoke
make nb1
```

Sau đó dùng Colab T4 để chạy phần GPU. Không nên trộn một phần `results/` từ laptop với
một phần từ Colab trong bài nộp chính thức; hãy chọn một bộ artefact nhất quán.

## 4. Lộ trình NB1 → NB5

| Bước | Thời gian T4 | Việc chính | Artefact bắt buộc |
|---|---:|---|---|
| NB1 | ~25 giây–2 phút | dữ liệu, template, mask, p95, split | `template_check.json`, `mask_proof.json`, `token_stats.json`, `data/split/` |
| NB2 | ~17–23 phút | đo và đóng băng baseline (a), (b) | `baselines_frozen.json` |
| NB3 | ~15–25 phút | train LoRA đúng cấu hình | `adapters/correct/`, dòng `correct` trong `runs.csv` |
| NB4 | ~45–60 phút | train ba đối chứng | ba adapter đối chứng, đủ bốn dòng `runs.csv` |
| NB5 | ~21 phút | đánh giá, verdict, autopsy, định tính | `verdict.json`, `autopsy.json`, `qualitative.json` |

Phải làm đúng thứ tự. Đặc biệt, NB2 phải hoàn tất trước NB3 để baseline (b) được đóng
băng trước khi bạn nhìn thấy kết quả fine-tune.

### 4.1. NB1 — chứng minh mask đúng

Chạy:

```bash
make nb1
```

Hoặc chạy stage `nb1` trên Colab. Sau khi chạy, tự kiểm tra:

- `results/template_check.json`: đọc `verdict` và ghi vào report template có giữ khối
  `<think>` hay không.
- `results/mask_proof.json`:
  - `answer_is_supervised` phải là `true`;
  - `question_is_masked` phải là `true`;
  - `supervised_fraction` phải nhỏ hơn `0.95`.
- Đọc `supervised_preview`: phải thấy JSON trả lời, không thấy nguyên ticket.
- `results/token_stats.json`: ghi lại `p95` và `suggested_max_length`.
- `data/split/train.jsonl` và `val.jsonl`: với corpus mặc định phải có 225/25 mẫu,
  split bằng seed 42.

Không sang NB2 nếu hai assert mask chưa xanh. `MASK_MODE=everything` chỉ để minh họa
bug, tuyệt đối không dùng cho run chính thức.

Với corpus mặc định, `assistant-only`, `masked-think` và `response-only` có thể cho mask
giống nhau vì output chỉ là JSON, không có reasoning trace thật. Vì vậy
`valid_trace_rate=0` trên corpus này không tự động chứng minh reasoning-trace collapse.

### 4.2. NB2 — đóng băng mốc phải vượt

Chạy:

```bash
make nb2
```

NB2 đo:

- (a) base model + prompt ngây thơ;
- (b) base model + prompt đã tối ưu.

Điểm cần kiểm tra ngay:

- `(b).target > (a).target`;
- format của (b) nên cao hơn rõ rệt;
- `results/baselines_frozen.json` có `optimized_prompt_sha`, `n_target=50`,
  `n_regression=15`, `smoke_mode=false` cho run nộp bài.

Từ thời điểm này không sửa `data/eval_target.jsonl`, `data/eval_regression.jsonl` hoặc
làm yếu `OPTIMIZED_PROMPT`. Nếu (b) không hơn (a), hãy cải thiện prompt ngay lúc này,
chạy lại NB2 và đóng băng lại trước khi train.

### 4.3. NB3 — train cấu hình đúng

Chạy:

```bash
make nb3
```

Cấu hình chính cần nhận ra trong output:

| Thành phần | Giá trị chính thức |
|---|---|
| placement | `text-linear`, chỉ text decoder |
| rank / alpha | `16 / 32` |
| learning rate | `1e-4` |
| effective batch T4 | `1 × 16 = 16` |
| precision T4 | `fp16`, không phải bf16 |
| epochs mặc định | `2` |
| mask | mask đã tokenize và chứng minh ở NB1 |

Kiểm tra số module và số tham số trainable không bao gồm vision tower. Với Qwen3.5-4B,
`text-linear` còn gồm các projection của linear-attention; đây là chủ ý.

Sau train phải có:

```text
adapters/correct/adapter_model.safetensors
adapters/correct/adapter_config.json
```

và một dòng `correct` trong `results/runs.csv`. Ghi lại `final_loss`, `max_steps`,
`peak_vram_gb`, precision và thời gian train để điền report.

Vài giá trị `grad_norm: nan` ở những step đầu của fp16 có thể là GradScaler đang dò
scale. Nếu `nan` kéo dài hết run hoặc final loss vô nghĩa, không dùng adapter đó.

### 4.4. NB4 — ba đối chứng công bằng

Chạy:

```bash
make nb4
```

Ba run phải chỉ đổi một biến:

| Run | Biến được đổi |
|---|---|
| `attn_only` | vị trí gắn chỉ `q,v`; rank được `matched_rank()` nâng lên để khớp ngân sách tham số |
| `wrong_lr` | LR từ `1e-4` xuống `1e-5` |
| `qlora` | base 16-bit thành base 4-bit |

Sau NB4, kiểm tra `runs.csv`:

1. Có đủ `correct`, `attn_only`, `wrong_lr`, `qlora`.
2. Bốn giá trị `max_steps` giống hệt nhau.
3. Độ lệch ngân sách:

   ```text
   abs(attn_only_params - correct_params) / correct_params < 0.05
   ```

4. `attn_only` thường có rank lớn hơn nhiều so với 16. Không hardcode rank từ tài liệu;
   dùng đúng con số notebook tính trên model của bạn.

Không xếp hạng bốn run bằng `final_loss`. Cột đó là training loss và chỉ dùng để phân
tích động học học. Xếp hạng chính thức lấy từ điểm `target` trong `autopsy.json` sau NB5.

Nên lưu hoặc chụp phần log loss của `correct` và `wrong_lr` khi chạy, vì `runs.csv` chỉ
lưu final loss trong khi câu 4.2 của report yêu cầu nhận xét đường loss. Nếu không còn
log, chỉ mô tả điều số liệu bạn thực sự có chứng minh được; không bịa diễn biến đường
cong.

Nếu Colab đứt giữa NB4, adapter đã train xong sẽ được tự động bỏ qua khi chạy lại:

```bash
# Chạy tiếp các stage còn lại:
python scripts/colab_run.py nb4 nb5

# Chỉ train lại qlora:
ONLY=qlora python notebooks/04_misconfig_autopsy.py

# Chỉ dùng khi thực sự muốn train lại cả ba run:
FORCE_RETRAIN=1 python notebooks/04_misconfig_autopsy.py
```

Không dùng `FORCE_RETRAIN=1` cho resume bình thường.

### 4.5. NB5 — đánh giá và phán quyết

Chạy:

```bash
make nb5
```

NB5 tạo bảng (a), (b), (c) trên bốn nhóm:

- `target`: độ chính xác từng trường của JSON;
- `regression`: khả năng trả lời 15 câu phổ thông;
- `format`: JSON parse được và có đủ 4 khóa;
- `latency`: mili-giây trên mỗi mẫu với greedy decoding.

Cổng chỉ `PASSED` khi đồng thời:

```text
fine_tune.target > baseline_b.target
fine_tune.regression - baseline_b.regression >= -0.02
```

Nếu `FAILED`, không sửa eval và không nới gate. Chẩn đoán theo thứ tự:

1. Format thấp: kiểm tra prompt shape, chat template, EOS và mask.
2. Regression tụt quá 0.02: có dấu hiệu catastrophic forgetting.
3. Target không tăng: xem learning rate và dữ liệu trước khi tăng rank.
4. Format/regression ổn nhưng target vẫn thua (b): prompt engineering đang hiệu quả
   hơn fine-tune; kết luận “không deploy adapter này” là hợp lệ.

`results/autopsy.json` mới là nguồn để xếp hạng bốn cấu hình. Hãy đặt thứ tự target này
cạnh thứ tự final loss trong `runs.csv`; nếu hai thứ tự khác nhau, đó là một kết quả
quan trọng cần nói rõ trong report.

## 5. Điền `submission/REPORT.md`

Không xóa 7 mục của template. Dùng bảng sau để lấy đúng nguồn số:

| Mục report | Nguồn |
|---|---|
| 1. Setup | `.env`/tier, `token_stats.json`, `template_check.json`, `runs.csv` |
| 2. Mask proof | `mask_proof.json`, nhất là `supervised_preview` |
| 3. Ba baseline | (a), (b) từ `baselines_frozen.json`; (c) từ `verdict.json.comparison` |
| 4. Autopsy | thông số train từ `runs.csv`; điểm tác vụ từ `autopsy.json` |
| 5. Phán quyết | `verdict.json.verdict` và `valid_trace_rate` |
| 6. Định tính | `qualitative.json` kết hợp `data/eval_target.jsonl` |
| 7. Kết luận | lập luận từ toàn bộ số đo; tối thiểu 150 từ |

### Công thức nên dùng trong mục 4

- Sai lệch tham số `attn_only`:

  ```text
  abs(attn_only.trainable_params - correct.trainable_params)
  / correct.trainable_params × 100%
  ```

- VRAM QLoRA tiết kiệm:

  ```text
  correct.peak_vram_gb - qlora.peak_vram_gb
  ```

- Phần trăm VRAM tiết kiệm:

  ```text
  (correct_vram - qlora_vram) / correct_vram × 100%
  ```

Luôn trả lời QLoRA đã đổi VRAM lấy điều gì bằng **điểm target/format của chính run**,
không chép kết quả đo tham khảo trong `docs/`.

### Chọn ví dụ định tính

`qualitative.json` được sắp theo `ft_score` từ thấp đến cao và có trường `i`. Dùng `i`
để tìm ticket cùng nhãn đúng trong dòng tương ứng của `data/eval_target.jsonl`. Chọn ít
nhất 5 ví dụ, trong đó tối thiểu 2 ví dụ mà fine-tune thua; nên có cả ca thắng và ca
thua để tránh cherry-pick.

Phiên bản hiện tại không lưu raw prediction của baseline (b) vào
`baselines_frozen.json`. Nếu cần điền chính xác cột `(b) prompt` của report, hãy giữ lại
output trong lúc chạy notebook hoặc sinh lại baseline (b) chỉ trên các chỉ số đã chọn
trước khi đóng runtime. Không tự đoán câu trả lời của baseline.

### Kiểm tra placeholder thủ công

Gatekeeper chỉ phát hiện một số mẫu placeholder. Hãy kiểm tra rộng hơn:

```bash
rg -n '<[^>]+>' submission/REPORT.md
```

Xóa hoặc thay toàn bộ placeholder như `<n>`, `<model id>`, `<CPU|...>`, `<điền>`,
`<paste>`. Kiểm tra mọi con số trong report khớp chính xác với `results/`.

Phần verdict nên ít nhất 100 từ; kết luận ít nhất 150 từ. “Điều tôi học được” phải nói
rõ một niềm tin hoặc quyết định đã thay đổi vì số đo nào, không viết nhận xét chung như
“LoRA rất mạnh”.

## 6. Verify trước khi nộp

Máy cá nhân có `.venv`:

```bash
make verify
```

Colab cài package trực tiếp vào Python hệ thống:

```bash
python scripts/verify.py
```

Đọc toàn bộ output. `Ready to submit` với `WARN` nghĩa là không có lỗi cơ học nhưng bạn
vẫn phải giải thích warning. `FAILED` verdict của mô hình là chấp nhận được; dòng
`[ FAIL ]` của gatekeeper thì phải sửa.

Ngoài verify, tự kiểm tra vì rubric yêu cầu một số thứ gatekeeper chưa bắt hết:

```bash
test -f results/qualitative.json
test -f adapters/correct/adapter_model.safetensors
test -f adapters/correct/adapter_config.json
rg -n '<[^>]+>' submission/REPORT.md
```

Checklist cuối:

- [ ] Mask proof có hai cờ `true`, fraction < 0.95.
- [ ] Run chính thức dùng đủ 50 target và 15 regression item, không phải smoke mode.
- [ ] Baseline (b) được đo trước train và mạnh hơn (a).
- [ ] `runs.csv` đủ 4 run, cùng `max_steps`.
- [ ] `attn_only` lệch ngân sách dưới 5%.
- [ ] Bảng autopsy xếp theo target, không theo final loss.
- [ ] Report có 5 ví dụ, gồm ít nhất 2 ca fine-tune thua.
- [ ] Verdict và kết luận đủ độ dài, diễn giải nguyên nhân.
- [ ] Không còn placeholder.
- [ ] Verify có 0 `[ FAIL ]`.

## 7. Nộp bài và tránh mất artefact

Theo [rubric.md](../rubric.md), mọi định dạng nộp đều phải có `results/` đầy đủ và
`submission/REPORT.md`. Nếu LMS yêu cầu URL, dùng GitHub hoặc HuggingFace Hub theo chỉ
dẫn của giảng viên. Chỉ làm bonus sau khi core verify đã xanh.

### Bẫy `.gitignore` khi nộp GitHub

Project cố ý ignore file sinh ra trong `results/` và `adapters/`. Vì vậy lệnh `git add`
bình thường có thể không đưa artefact vào commit. Kiểm tra bằng:

```bash
git check-ignore -v results/verdict.json adapters/correct/adapter_model.safetensors
```

Khi đang tạo commit nộp bài, force-add đúng phạm vi artefact:

```bash
git add submission/REPORT.md submission/REFLECTION.md
git add -f results adapters/correct
git status --short
```

Trước khi commit, xác nhận `git status` có các JSON, `runs.csv`,
`adapter_model.safetensors` và `adapter_config.json`. Không thêm `.env`, token, `.venv`,
cache, checkpoint optimizer hoặc trọng số base model.

Sau khi push, mở repo trên web và thử mở trực tiếp:

- `submission/REPORT.md`;
- `results/verdict.json`;
- `results/runs.csv`;
- `adapters/correct/adapter_config.json`.

Nếu adapter vượt giới hạn file của GitHub, dùng Git LFS hoặc đẩy adapter lên
HuggingFace Hub, nhưng vẫn phải giữ `results/` và report ở nơi grader truy cập được.

## 8. Bonus — tối đa +15 điểm

Chỉ bắt đầu bonus sau khi bộ core NB1–NB5 đã chạy xong, report core đã có số và verify
không còn `[ FAIL ]`. Nên sao lưu toàn bộ `results/`, `adapters/correct/` và
`submission/REPORT.md` trước khi thử bonus vì một số thí nghiệm sẽ ghi đè
`adapters/correct` hoặc `results/verdict.json`.

| Bonus | Điểm | Độ khó | Artefact/bằng chứng chính |
|---|---:|---|---|
| B1. Merge và hot-swap | +3 | vừa | `results/merge_check.json`, ≥2 adapter trên cùng base |
| B2. Dataset miền riêng | +3 | khó | ≥200 mẫu, `data/CUSTOM_DATASET.md`, chạy lại toàn pipeline |
| B3. Reasoning-trace collapse | +4 | rất khó | hai run mask mode, target/trace/regression của cả hai |
| B4. Rank sweep có kiểm soát | +3 | khó | r ∈ {8,16,64}, giữ placement/LR/step cố định |
| B5. HuggingFace Hub | +2 | dễ | adapter public và link trong report |
| B6. Optimizer mismatch | 0 | nghiên cứu | dự đoán trước, Adam vs Muon có kiểm soát |
| B7. MoE route-aware LoRA | 0 | nghiên cứu | full vs top-routed vs random experts |

B1–B5 cộng đúng tối đa 15 điểm. B6 và B7 không tính điểm nhưng vẫn có thể đưa vào phụ
lục như thí nghiệm mở rộng.

### 8.1. B1 — merge và phục vụ nhiều adapter (+3)

Điều kiện đầu vào: core đã có `adapters/correct` và ít nhất một adapter đối chứng như
`attn_only`.

Chạy full eval, không dùng smoke mode:

```bash
unset EVAL_LIMIT
make nb6
```

Trên Colab:

```bash
python notebooks/06_merge_and_serve.py
```

Kiểm tra:

- `results/merge_check.json` tồn tại;
- `after_merge - before_merge >= -0.01`;
- output phần hot-swap cho thấy ít nhất hai tên adapter trên cùng một base đã nạp;
- ghi lại output của cùng một ticket khi đổi adapter.

Trong phụ lục report, trả lời:

1. Merge có làm điểm target tụt không, tụt bao nhiêu?
2. Merge bỏ overhead adapter nhưng đánh đổi khả năng tháo/gắn và hot-swap ra sao?
3. Khi nào nên giữ adapter riêng: nhiều khách hàng/tác vụ, rollback nhanh, A/B test,
   hoặc cần cập nhật độc lập.

`adapters/merged/` chứa trọng số đã merge và có thể rất lớn. Không cần đẩy thư mục này
lên GitHub để chứng minh B1; `merge_check.json` và phân tích trong report mới là bằng
chứng gọn nhất.

### 8.2. B2 — dataset miền riêng (+3)

Yêu cầu tối thiểu:

- ít nhất 200 mẫu chất lượng cao trong miền bạn thật sự quan tâm;
- có train/eval tách biệt và khử trùng lặp;
- có `data/CUSTOM_DATASET.md`;
- chạy lại toàn bộ NB1→NB5, không dùng kết quả core cũ ghép vào.

`CUSTOM_DATASET.md` nên ghi rõ:

1. Nguồn dữ liệu và quyền sử dụng.
2. Cách thu thập/làm sạch/gán nhãn.
3. Kích thước train, validation, target eval và regression eval.
4. Cách khử nhiễm: exact match, normalized match, near-duplicate hoặc kiểm tra thủ công.
5. Vì sao dữ liệu mới về phân phối so với dữ liệu web phổ thông mà base có thể đã thấy.
6. Schema của từng record và các enum/tiêu chí chấm.

Đường ít phải sửa code nhất là giữ schema record mà lab đang dùng:

```json
{
  "instruction": "...",
  "input": "...",
  "output": "...",
  "label": {"intent": "...", "urgency": "...", "product": "...", "sentiment": "..."}
}
```

Nếu đổi sang một tác vụ hoàn toàn khác, phải cập nhật cả scorer, prompt và tập regression,
đồng thời giải thích thang đo mới. Không chỉ thay train data rồi giữ evaluator cũ.

Quy trình:

1. Sao lưu bộ core mặc định.
2. Tạo dataset và `CUSTOM_DATASET.md`.
3. Chốt eval trước khi train.
4. Chạy lại NB1→NB5 từ đầu.
5. Verify có thể báo `WARN` vì checksum eval thay đổi; file `CUSTOM_DATASET.md` là khai
   báo bắt buộc, nhưng bạn vẫn phải giải thích thay đổi trong report.

Không cập nhật eval sau khi nhìn thấy điểm model. Dataset 200 mẫu được kiểm tra kỹ có
giá trị hơn hàng nghìn mẫu crawl nhưng nhiễu.

`data/CUSTOM_DATASET.md` cũng đang nằm trong `.gitignore`; khi nộp B2 qua GitHub, nhớ
thêm rõ ràng bằng `git add -f data/CUSTOM_DATASET.md` và kiểm tra lại `git status`.

### 8.3. B3 — reasoning-trace collapse (+4)

Đây là bonus khó nhất. Với corpus mặc định, đổi `MASK_MODE` **chưa tạo ra thí nghiệm
có nghĩa** vì:

- toàn bộ output train hiện là JSON trần, không có `<think>...</think>` thật;
- `assistant-only`, `masked-think`, `response-only` vì thế có thể sinh mask giống nhau;
- `generate_batch()` mặc định đang gọi `enable_thinking=False`, nên
  `valid_trace_rate` có thể luôn bằng 0 dù model có khả năng thinking.

Muốn làm B3 đúng, cần:

1. Một tập train có reasoning trace thật nằm trong nội dung assistant, ví dụ:

   ```text
   <think>lập luận có nội dung và hợp lệ</think>
   {"intent": "...", ...}
   ```

2. Chạy NB1 để xác nhận template giữ trace và hai mask mode thật sự supervise số token
   khác nhau.
3. Hai run có cùng model, data, seed, placement, rank, LR và step; chỉ đổi:
   `assistant-only` so với `response-only`.
4. Một bản evaluator bonus bật thinking nhất quán cho cả hai run, chẳng hạn truyền
   `enable_thinking=True` vào generation. Không thay evaluator giữa hai run.
5. Lưu riêng adapter và verdict của từng mode.

Không nên chạy hai lệnh bonus trực tiếp trên bộ core duy nhất vì NB3/NB5 sẽ ghi đè
`adapters/correct` và `results/verdict.json`. Cách an toàn nhất là dùng hai clone/worktree
riêng. Nếu dùng cùng một Colab runtime, sau mỗi run phải copy artefact sang tên riêng,
ví dụ:

```bash
MASK_MODE=assistant-only python notebooks/03_train_correct.py
python notebooks/05_evaluate_and_verdict.py
cp -R adapters/correct adapters/b3_assistant_only
cp results/verdict.json results/verdict_b3_assistant_only.json

MASK_MODE=response-only python notebooks/03_train_correct.py
python notebooks/05_evaluate_and_verdict.py
cp -R adapters/correct adapters/b3_response_only
cp results/verdict.json results/verdict_b3_response_only.json
```

Trên máy cá nhân có `.venv`, có thể thay hai lệnh `python notebooks/...` bằng
`MASK_MODE=... make nb3` và `make nb5`.

Sau thí nghiệm, khôi phục `adapters/correct` và `verdict.json` của core trước khi nộp,
hoặc để toàn bộ B3 trong một repo/nhánh bonus riêng và link từ report.

Bảng bắt buộc trong phụ lục:

| `MASK_MODE` | target | valid trace rate | regression | max steps |
|---|---:|---:|---:|---:|
| assistant-only | | | | |
| response-only | | | | |

Phân tích liệu target có tăng trong khi valid trace rate giảm hay không, và liệu chỉ
nhìn target có phát hiện vấn đề không. Không kết luận “collapse” nếu hai mask thực tế
giống nhau hoặc evaluator vẫn tắt thinking.

### 8.4. B4 — quét rank có kiểm soát (+3)

Train ba run với:

```text
placement = text-linear
r         = 8, 16, 64
alpha     = 2r
LR        = 1e-4
16-bit base
same seed, data, mask, effective batch, epochs và max_steps
```

Project chưa có một lệnh `make` riêng cho B4. Cách làm an toàn là sao chép logic NB3
sang một notebook/script bonus, đặt các key và output directory riêng như
`rank_r8`, `rank_r16`, `rank_r64`. Không sửa `SPECS["correct"]` rồi ghi đè adapter core.

Sau train, chấm cả ba adapter trên cùng target eval bằng `NAIVE_PROMPT` như NB5, rồi
lưu một bảng, ví dụ `results/rank_sweep.json`:

| run | r | trainable params | max steps | final loss | target | format | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| rank_r8 | 8 | | | | | | |
| rank_r16 | 16 | | | | | | |
| rank_r64 | 64 | | | | | | |

Trong report, so ba effect size bằng điểm target:

- rank: biên độ tốt nhất − tệ nhất trong r ∈ {8,16,64};
- placement: `correct.target - attn_only.target`;
- learning rate: `correct.target - wrong_lr.target`.

Từ đó xếp hạng rank, placement và LR theo ảnh hưởng thực đo. Trả lời liệu 250 mẫu có
đủ thông tin để rank 64 dùng hết năng lực hay chỉ tăng tham số/ghi nhớ.

### 8.5. B5 — đẩy adapter lên HuggingFace Hub (+2)

Yêu cầu: model repo công khai, adapter tải được và link nằm trong phụ lục report.

```bash
pip install -U huggingface_hub
huggingface-cli login
```

Ví dụ upload cả adapter và bằng chứng:

```python
from huggingface_hub import HfApi

api = HfApi()
repo_id = "<user>/lab21-<MSSV>-qwen35-triage-vi"
api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
api.upload_folder(
    folder_path="adapters/correct",
    repo_id=repo_id,
    repo_type="model",
)
api.upload_folder(
    folder_path="results",
    path_in_repo="results",
    repo_id=repo_id,
    repo_type="model",
)
api.upload_file(
    path_or_fileobj="submission/REPORT.md",
    path_in_repo="submission/REPORT.md",
    repo_id=repo_id,
    repo_type="model",
)
```

Model card nên ghi base model, tier/GPU, task, schema output, cách load bằng PEFT, điểm
(a)/(b)/(c), verdict và giới hạn đã quan sát. Không upload token hoặc base weights.

Sau upload, mở repo ở chế độ chưa đăng nhập hoặc cửa sổ ẩn danh để chắc chắn grader
truy cập được. Ghi URL vào checkbox B5 trong `submission/REPORT.md`.

### 8.6. B6 — optimizer mismatch, không tính điểm

Thí nghiệm một optimizer họ Muon trên model đã pretrain bằng Adam. Đây không phải việc
chỉ đổi tên optimizer rồi giữ nguyên LR.

Quy trình nghiên cứu đúng:

1. Viết dự đoán trước khi chạy.
2. Tạo run bonus riêng, không ghi đè `correct`.
3. Giữ model, data, mask, placement, rank, effective batch và step giống run đối chứng.
4. Chọn/tune LR phù hợp cho Muon; không bê `1e-4` của Adam sang mà không biện minh.
5. So target, regression, format, latency và độ lớn cập nhật; không chỉ so train loss.

Nếu cần cài thư viện ngoài, pin phiên bản vào file requirements bonus để thí nghiệm tái
lập được. Báo cáo cả trường hợp dự đoán sai và giải thích dựa trên số.

### 8.7. B7 — MoE route-aware LoRA, không tính điểm

Bonus này cần base MoE và GPU lớn, ví dụ một biến thể Qwen3.5 MoE; T4 mặc định không
phù hợp. Thiết kế gồm ba nhánh cùng ngân sách:

1. LoRA trên toàn bộ expert phù hợp.
2. Profile routing trên calibration set, chọn top 25% expert được gọi nhiều nhất ở mỗi
   lớp rồi chỉ gắn LoRA vào các expert đó.
3. Chọn ngẫu nhiên 25% expert làm control.

Giữ data, step, rank/LR và evaluator giống nhau. Báo cáo số tham số trainable, target,
regression, format, VRAM và routing coverage. Không train router vì rủi ro mất ổn định;
mục tiêu là đo giá trị của tín hiệu routing, không thay đổi chính sách định tuyến.

### 8.8. Cách ghi bonus vào report

Trong phần “Phụ lục — thưởng đã làm”, chỉ đánh dấu mục thực sự có artefact. Với mỗi
bonus nên có:

- cấu hình và biến duy nhất được thay đổi;
- bảng số liệu;
- đường dẫn artefact hoặc notebook/script;
- câu trả lời cho câu hỏi của bonus;
- giới hạn và điều chưa kiểm chứng.

Không để kết quả bonus ghi đè hoặc làm mâu thuẫn số core trong 7 mục chính.

## 9. Xử lý tình huống thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Colab báo không có GPU | Chọn lại T4; không chạy NB2–NB5 trên CPU |
| T4 hiển thị bf16 | Dừng và kiểm tra môi trường; T4 phải dùng fp16 |
| NB2/NB5 lâu không in kết quả | Chờ các dòng progress/ETA; generation là phần tốn thời gian nhất |
| `eval slice mismatch` | NB2 và NB5 phải cùng `EVAL_LIMIT`; run nộp bài để cả hai unset |
| Verify báo smoke mode | Xóa/unset `EVAL_LIMIT`, chạy lại NB2 và NB5 đầy đủ |
| Bốn run khác `max_steps` | Đặt một `EPOCHS` chung và train lại NB3/NB4 tương ứng |
| Đã thử `EPOCHS=1`, sau đó đổi về 2 | Phải train lại cả `correct` và ba contrast; NB4 cần `FORCE_RETRAIN=1` hoặc xóa đúng adapter cũ |
| Colab đứt giữa NB4 | Chạy lại NB4; adapter hoàn tất được tự skip |
| OOM ở run sau | Không chạy nhiều phiên GPU; để notebook giải phóng model giữa các run |
| `format` gần 0 | Kiểm tra raw output, prompt shape, mask, template và EOS |
| `regression` tụt | Phân tích catastrophic forgetting; không nới ngưỡng |
| Fine-tune thua baseline (b) | Báo `FAILED` trung thực và kết luận có/không nên deploy dựa trên 4 nhóm |
| GitHub không thấy `results/` | File đang bị ignore; dùng `git add -f` đúng thư mục artefact |

## 10. Thứ tự ưu tiên khi thiếu thời gian

Ưu tiên học và lấy bằng chứng theo thứ tự:

1. NB1 — mask proof.
2. NB2 — baseline đóng băng.
3. NB3 — adapter `correct`.
4. NB5 — verdict cho run chính.
5. NB4 — ba contrast công bằng.
6. NB6 và bonus — chỉ làm sau khi core hoàn chỉnh.

Tuy nhiên, bài nộp đầy đủ theo rubric vẫn cần NB4. Nếu chỉ dùng `EVAL_LIMIT=8` để thử,
hãy chạy lại NB2 và NB5 với full eval. Nếu từng đổi `EPOCHS`, hãy bảo đảm cả NB3 và NB4
được train lại cùng ngân sách trước khi nộp.
