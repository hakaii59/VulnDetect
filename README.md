# Vulnerachek-AI

Hệ thống phát hiện lỗ hổng bảo mật trong source code C/C++/Java, tích hợp
vào CI/CD pipeline để gate Pull Request.

## Kiến trúc

Pipeline 3 lớp xử lý tuần tự trên diff của PR:

1. **Regex pattern matching** — lọc sơ bộ, tốc độ cao (Stage 2)
2. **Tree-sitter AST parsing** — kiểm tra ngữ cảnh cấu trúc code, loại bớt
   false positive từ lớp 1 (Stage 3)
3. **GraphCodeBERT classifier** (fine-tuned) — phân loại cuối, giảm false
   positive hơn nữa (Stage 4)

Kết quả cuối dùng để comment + fail/pass check trên GitHub Actions (Stage 6).

## Tiến độ

- [x] **Stage 1 — Data pipeline**: hợp nhất MegaVul/Big-Vul/CVEfixes/PrimeVul
      thành schema thống nhất, split theo project/commit, tính trọng số cân
      bằng class. Xem chi tiết bên dưới.
- [ ] Stage 2 — Regex layer
- [ ] Stage 3 — Tree-sitter AST layer
- [ ] Stage 4 — Fine-tune GraphCodeBERT
- [ ] Stage 5 — Export ONNX + inference
- [ ] Stage 6 — Tích hợp GitHub Actions
- [ ] Stage 7 — Benchmark & docs đầy đủ

---

## Stage 1 — Data Pipeline

### Cài đặt

```bash
pip install -r requirements.txt
pip install -e .
```

### 1. Tải dataset

4 dataset (MegaVul, Big-Vul, CVEfixes, PrimeVul) được host bên ngoài
(Google Drive/Zenodo/Hugging Face) và quá lớn để tải tự động. Xem hướng dẫn
chi tiết từng dataset (link chính thức, cách convert, tên field) tại
**[docs/DATASETS.md](docs/DATASETS.md)**.

Đặt file đã tải theo đường dẫn khai báo trong `configs/dataset.yaml`
(mặc định `data/raw/<dataset>/...`). Không cần đủ cả 4 mới chạy được — set
`enabled: false` cho dataset nào chưa có.

Kiểm tra nhanh tên cột/field trong file đã tải có khớp với loader không
(không parse toàn bộ, chỉ đọc header/entry đầu):

```bash
python scripts/build_dataset.py --verify
```

### 2. Build dataset thống nhất

```bash
python scripts/build_dataset.py --config configs/dataset.yaml
```

Output: `data/processed/{train,val,test}.jsonl`, mỗi dòng có schema:

```json
{
  "code": "...",
  "label": 1,
  "language": "c",
  "cwe_type": "CWE-119",
  "source": "bigvul",
  "project": "openssl/openssl",
  "commit_id": "...",
  "func_name": "...",
  "id": "79334fc7601a63cb",
  "weight": 1.4
}
```

(`weight` chỉ có ở `train.jsonl`, dùng cho `WeightedRandomSampler` ở Stage 4.)

### Các quyết định kỹ thuật chính

**Vì sao split theo project/commit thay vì random theo dòng?**
Phiên bản trước và sau khi vá của cùng một hàm chỉ khác nhau 1-2 dòng; các
hàm tiện ích cũng thường lặp lại gần như nguyên văn giữa nhiều commit/file
trong cùng project. Nếu split ngẫu nhiên theo dòng, một cặp gần-trùng-nhau
dễ bị chia sang cả train lẫn test — model học thuộc lòng thay vì học đặc
trưng lỗ hổng thực sự, khiến F1 trên test bị thổi phồng ảo. `split.py` gom
mọi record theo `project` (fallback về `commit_id`, rồi về hash nội dung) và
giữ nguyên cả nhóm trong một split duy nhất (xem docstring trong
`src/vulnerachek/data/split.py`).

**Vì sao dedup trước khi split?**
4 dataset đều crawl từ các commit vá CVE trên GitHub nên vùng crawl chồng
lấn nhiều — cùng một commit có thể xuất hiện ở cả Big-Vul lẫn CVEfixes. Nếu
không dedup, cùng một đoạn code có thể rơi vào train của dataset này và test
của dataset kia. `merge.py` dedup theo hash nội dung (đã chuẩn hóa
whitespace) trước khi đưa vào `split.py`, đồng thời loại các record có nội
dung giống hệt nhau nhưng nhãn mâu thuẫn giữa các nguồn.

**Vì sao dùng weighted sampling thay vì oversampling/undersampling?**
Corpus mất cân bằng ở 2 chiều cùng lúc: (a) hàm có lỗ hổng luôn là thiểu số
so với hàm an toàn, và (b) 4 dataset không đóng góp đều cho C/C++/Java
(Big-Vul và PrimeVul chỉ có C/C++). Nhân bản record thiểu số (oversampling)
khiến model dễ học thuộc văn bản bị lặp lại nhiều lần; bỏ bớt record đa số
(undersampling) làm mất dữ liệu vốn đã hiếm. `balance.py` tính trọng số
nghịch đảo theo từng cặp `(label, language)` để dùng với
`WeightedRandomSampler` ở Stage 4 — thay đổi tần suất *được lấy mẫu* mỗi
epoch mà không đụng vào tập dữ liệu gốc. Trọng số chỉ tính cho `train`;
`val`/`test` giữ nguyên phân phối tự nhiên để số liệu precision/recall/F1
phản ánh đúng hiệu năng trên dữ liệu thực tế (vốn cũng mất cân bằng).

### Test

```bash
pytest tests/ -v
```

38 unit test bao phủ schema validation, từng loader (dùng fixture nhỏ tự
tạo, không cần dataset thật), merge/dedup, split (đảm bảo không leak theo
project), và balance (đảm bảo trọng số cân bằng đúng theo bucket).

### Cấu trúc code

```
src/vulnerachek/data/
  schema.py          # VulnRecord thống nhất + chuẩn hóa CWE/ngôn ngữ
  loaders/
    base.py          # interface chung, column_map để không phải sửa code
                      # khi dataset đổi tên cột
    megavul.py        # JSON
    bigvul.py         # CSV, tách 1 dòng vul=1 thành cặp before(label 1)/after(label 0)
    cvefixes.py        # SQLite, JOIN nhiều bảng + fallback query
    primevul.py         # JSONL
  merge.py            # dedup theo content hash, loại conflict nhãn
  split.py             # split theo group (project/commit), không leak
  balance.py            # sample weight cho WeightedRandomSampler
scripts/build_dataset.py  # CLI orchestrator
configs/dataset.yaml        # đường dẫn raw data + column_map override
docs/DATASETS.md              # hướng dẫn tải từng dataset
tests/                          # unit test + fixture nhỏ (không cần dataset thật)
```

---

*Dự án tham khảo ý tưởng ban đầu từ repo demo `GraphCodeBERT-VulnDetector`
(đã clone về cùng thư mục cha, không nằm trong git history của dự án này).*
