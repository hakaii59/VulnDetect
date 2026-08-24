# Tải 4 dataset cho Stage 1

Cả 4 dataset đều lớn (hàng trăm MB tới vài GB) và được host bên ngoài GitHub
(Google Drive / Zenodo / Hugging Face). Không thể tải tự động trong pipeline
này — làm theo hướng dẫn dưới rồi đặt file vào đúng đường dẫn khai báo trong
`configs/dataset.yaml`.

Sau khi tải xong bất kỳ dataset nào, chạy để kiểm tra tên cột/field có khớp
với loader không, trước khi chạy full pipeline:

```bash
python scripts/build_dataset.py --verify
```

Nếu tên cột lệch (dataset đổi version), sửa `column_map` tương ứng trong
`configs/dataset.yaml` — không cần sửa code Python.

---

## 1. Big-Vul (`bigvul`)

- Repo: https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset
- README của repo có link Google Drive tới bản CSV đã làm sạch
  (`MSR_data_cleaned.csv`, ~500MB). Tải file này.
- Đặt vào: `data/raw/bigvul/MSR_data_cleaned.csv`
- Cột quan trọng: `func_before`, `func_after`, `vul`, `CWE ID`, `project`,
  `commit_id`, `lang`.
- Ngôn ngữ: chỉ C/C++.

## 2. MegaVul (`megavul`)

- Repo: https://github.com/Icyrockton/MegaVul
- Vào trang **Releases**: https://github.com/Icyrockton/MegaVul/releases —
  tải `megavul_simple.json` (nhẹ hơn, đủ dùng cho Stage 1) hoặc
  `megavul.json` (đầy đủ, có sẵn field `language`). Có thể cần tải riêng bản
  cho C/C++ và Java tùy theo version release.
- Cũng có mirror trên Hugging Face: https://huggingface.co/datasets/hitoshura25/megavul
- Đặt (các) file `.json` vào thư mục: `data/raw/megavul/`
  (loader tự động đọc mọi `*.json` trong thư mục này).
- Field quan trọng: `func`, `is_vul`, `cwe_ids`, `commit_hash`, `repo_name`,
  `language`.
- Ngôn ngữ: C, C++, Java.

## 3. CVEfixes (`cvefixes`)

- Repo (code thu thập): https://github.com/secureIT-project/CVEfixes
- Dataset thật nằm trên Zenodo (repo GitHub chỉ chứa code crawl, không chứa
  data do giới hạn dung lượng GitHub):
  https://zenodo.org/records/4476564 (DOI: 10.5281/zenodo.4476563)
  — bản mới nhất (v1.0.8, tính đến 2024-07-23) có thể nằm ở record khác,
  kiểm tra link "Versions" trên trang Zenodo trên để lấy bản mới nhất.
- Zenodo cung cấp **SQL dump nén**, không phải file `.db` sẵn dùng. Làm theo
  `INSTALL.md` của repo GitHub để convert sang SQLite:
  https://github.com/secureIT-project/CVEfixes/blob/main/INSTALL.md
- Đặt file kết quả vào: `data/raw/cvefixes/CVEfixes.db`
- Schema là quan hệ nhiều bảng (`method_change` -> `file_change` ->
  `commits` -> `repository`/`fixes` -> `cve` -> `cwe_classification`).
  Loader đã viết sẵn 1 câu JOIN mặc định; nếu version bạn tải có schema khác
  và câu query mặc định lỗi, loader tự fallback về query tối giản (mất
  project/CWE) và log warning — sửa `column_map.query` trong
  `configs/dataset.yaml` bằng câu SQL đúng để lấy lại đầy đủ field.
- Ngôn ngữ: đa dạng (C/C++/Java/Python/...); pipeline này chỉ giữ lại
  C/C++/Java, các ngôn ngữ khác bị loader bỏ qua.

## 4. PrimeVul (`primevul`)

- Repo: https://github.com/DLVulDet/PrimeVul — link tải (Google Drive) nằm
  trong README của repo.
- Mirror Hugging Face: https://huggingface.co/datasets/starsofchance/PrimeVul
- Tải về sẽ có `train.jsonl` / `valid.jsonl` / `test.jsonl` (PrimeVul đã tự
  chia split riêng, nhưng pipeline này **gộp và tự chia lại** theo
  project/commit cùng 3 dataset kia — xem docstring trong
  `src/vulnerachek/data/loaders/primevul.py` để biết lý do: nếu giữ nguyên
  split gốc, cùng 1 project có thể vừa nằm trong `PrimeVul-train` vừa nằm
  trong dataset khác ở `test`, gây rò rỉ dữ liệu).
- Đặt cả 3 file `.jsonl` vào: `data/raw/primevul/`
- Field quan trọng: `func`, `target`, `cwe`, `project`, `commit_id`.
- Ngôn ngữ: chỉ C/C++.

---

## Bắt đầu với ít hơn 4 dataset

Không cần tải đủ cả 4 mới chạy được. Trong `configs/dataset.yaml`, đặt
`enabled: false` cho dataset nào chưa tải xong — pipeline vẫn chạy bình
thường với phần còn lại.
