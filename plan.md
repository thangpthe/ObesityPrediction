# Project Plan — Obesity Risk Classification

> **Mục tiêu:** Tái cấu trúc repository thành một project Machine Learning có giám sát hoàn chỉnh, có thể chạy end-to-end: kiểm toán dữ liệu → tiền xử lý → huấn luyện/đánh giá → REST API bằng FastAPI → giao diện web đơn giản → tài liệu và kiểm thử.
>
> **Định vị:** Đây là công cụ học thuật minh họa phân loại mức độ/nguy cơ béo phì dựa trên dữ liệu đầu vào, **không phải thiết bị y tế và không đưa ra chẩn đoán lâm sàng**.

## 1. Tổng quan và phạm vi

### 1.1 Sản phẩm cần xây dựng

Ứng dụng web cho phép người dùng nhập các thông tin nhân khẩu học và lối sống (theo đúng schema cuối cùng đã được xác minh), gửi yêu cầu tới FastAPI và xem:
- Nhãn dự đoán theo các lớp thực sự có trong dataset.
- Xác suất dự đoán theo từng lớp (chỉ hiển thị khi mô hình hỗ trợ xác suất và đã được kiểm tra calibration).
- Tóm tắt dữ liệu đã nhập, thông báo lỗi rõ ràng và giải thích giới hạn của mô hình.
- API docs tự động tại `/docs`.

### 1.2 Phạm vi kỹ thuật đề xuất

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.11+ |
| Data processing | pandas, NumPy |
| Machine Learning | scikit-learn |
| Model persistence | joblib |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | HTML, CSS, JavaScript thuần (responsive, form đơn giản) |
| Visualization / EDA | matplotlib, seaborn |
| Testing | pytest, HTTPX |
| Code quality | Ruff, type hints |
| Version control | Git, GitHub |

Không cần xây dựng frontend phức tạp. Ưu tiên một form dễ dùng, có kiểm tra dữ liệu đầu vào, kết quả rõ ràng và giao tiếp qua REST API.

## 2. Đánh giá hiện trạng repository

Repository hiện có một số thành phần có thể tái sử dụng, nhưng chưa nên xem là sản phẩm hoàn chỉnh.

### 2.1 Thành phần đã tìm thấy

- Có các script tiền xử lý và huấn luyện: `scripts/pre_processing.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/metrics.py`.
- Có các script cũ ở root: `training.py`, `testing.py`, `pre-processing.py`.
- Có Flask app cũ tại `app.py` và template HTML tại `templates/index.html`.
- Có dataset trong `dataset/`: `obesity_prediction.csv`, `obesity_prediction_dataset_final.csv`, `data_bmi.csv`, `data_weight_height.csv`.
- Có model pickle và file test đã commit vào repository.
- Có thư mục `venv/` được đưa lên GitHub.

### 2.2 Vấn đề cần xử lý trước khi phát triển tiếp

| Mức độ | Phát hiện | Hướng xử lý |
|---|---|---|
| Blocker | `scripts/config.py` trỏ tới `data/raw/obesity_prediction.csv` và `data/raw/obesity_prediction_dataset_final.csv`, nhưng dữ liệu hiện có nằm trong `dataset/`. | Chuẩn hóa lại cấu trúc và đường dẫn; không để nhiều script tự định nghĩa path. |
| Blocker | Chưa xác minh được ý nghĩa và cách tạo nhãn `Obesity` (đặc biệt nhãn 0/1). | Tìm metadata, nguồn dataset, tài liệu gốc và quy tắc gán nhãn trước khi coi đây là bài toán chẩn đoán. |
| High | Feature `BMI` có thể được tính trực tiếp từ `Height` và `Weight`; nhãn `Obesity` cũng có thể được tạo từ BMI. | Kiểm tra quy tắc tạo nhãn. Nếu target được suy ra từ BMI, đưa BMI vào feature sẽ gây target leakage/đánh giá quá lạc quan. Cần so sánh thí nghiệm có và không có BMI. |
| High | Script preprocessing cân bằng lớp trước khi chia train/validation/test (nếu dùng output đã cân bằng để split). | Chia dữ liệu trước; chỉ resampling trên train fold. Giữ validation/test theo phân phối tự nhiên. |
| High | `CategoricalNB` đang bị comment và không phù hợp trực tiếp với các biến liên tục nếu chưa discretize. | Chỉ dùng khi có preprocessing phù hợp; ưu tiên baseline Logistic Regression, Decision Tree, Random Forest và HistGradientBoosting/Gradient Boosting. |
| High | Script cũ dùng metric nhị phân mặc định và `roc_curve` với nhãn dự đoán; không phù hợp nếu dữ liệu có nhiều lớp. | Xác minh số lớp; dùng classification_report, confusion matrix, macro-F1, balanced accuracy và ROC-AUC OVR khi phù hợp. |
| High | Model pickle và test data đang được lưu trong repository. | Chỉ giữ artifact cần thiết; không commit dữ liệu nhạy cảm hoặc test set. Thêm quy tắc .gitignore và cân nhắc Git LFS/Model Registry nếu artifact lớn. |
| Medium | `venv/` đã được commit. | Gỡ khỏi version control, thêm `.venv/`, `venv/`, cache và file môi trường vào `.gitignore`. |
| Medium | App hiện tại dùng Flask, nhận form và dự đoán trực tiếp; chưa có schema API, validation tập trung và test. | Thay backend bằng FastAPI, Pydantic request/response schemas và test endpoint. |
| Medium | Chưa thấy README/requirements chuẩn ở root. | Bổ sung README, requirements hoặc pyproject.toml, hướng dẫn setup/run/train/test. |

**Lưu ý quan trọng về dữ liệu:** Các file CSV đang tồn tại trong repo chưa đủ để kết luận dataset hợp lệ cho mục tiêu “chẩn đoán bệnh béo phì”. Không được tự đặt tên các nhãn 0/1 thành “béo phì/không béo phì” khi chưa tìm được mapping chính thức. Nếu không tìm được nguồn, data dictionary hoặc quy tắc gán nhãn đáng tin cậy, cần tìm lại raw dataset và ghi rõ nguồn, license, đơn vị đo, định nghĩa target.

## 3. Kiểm toán dữ liệu — việc phải hoàn thành đầu tiên

### 3.1 Xác minh nguồn và schema

Kiểm tra từng file:
- Nguồn tải, tên dataset gốc, phiên bản, license và điều kiện sử dụng.
- Số dòng/cột, kiểu dữ liệu, đơn vị đo, missing values, duplicate rows.
- Phạm vi hợp lệ của Age, Height, Weight, BMI, CH2O, FAF.
- Ý nghĩa của Gender, family_history_with_overweight, CH2O, FAF và từng giá trị mã hóa.
- Tập giá trị của target `Obesity`, số lượng mẫu theo lớp và mapping nhãn.
- Quan hệ giữa các file: file nào là raw, file nào đã ghép, lọc, cân bằng hoặc biến đổi.

Tạo báo cáo `reports/data_audit.md` và notebook EDA nếu cần. Không ghi đè raw data.

### 3.2 Kiểm tra tính nhất quán BMI

Nếu Height tính bằng mét và Weight tính bằng kg:

`BMI = Weight / (Height ** 2)`

So sánh BMI được cung cấp với BMI tính lại, ghi nhận sai số và xử lý đơn vị. Không tự động tính BMI nếu chưa xác nhận đơn vị Height/Weight.

### 3.3 Kiểm tra target leakage

Đây là điểm rủi ro lớn nhất của project:
1. Tìm hiểu target `Obesity` được bác sĩ/nguồn dữ liệu gán hay được sinh bằng ngưỡng BMI.
2. Nếu target được tạo từ BMI, mô hình có BMI trong feature chỉ đang học lại công thức/quy tắc tạo nhãn; kết quả accuracy cao không chứng minh khả năng dự báo hữu ích.
3. Chạy tối thiểu hai cấu hình: (A) có BMI và (B) không có BMI. Báo cáo rõ mục tiêu và sự khác biệt.
4. Nếu nhãn được suy ra từ BMI, cân nhắc đổi tên project thành “BMI-based obesity category classification” và mô tả đúng bản chất, không gọi là chẩn đoán bệnh.
5. Nếu mục tiêu là dự đoán nguy cơ trước khi có BMI, cần dataset có nhãn độc lập, thời điểm đo rõ ràng và các biến đầu vào có sẵn trước thời điểm dự đoán.

### 3.4 Chính sách chia dữ liệu và mất cân bằng

- Chia train/validation/test theo stratified split nếu phù hợp với cấu trúc dữ liệu.
- Khuyến nghị 70/15/15 hoặc 60/20/20; cố định random seed và lưu thông tin split.
- Nếu có nhiều bản ghi của cùng một người, phải split theo subject/group để tránh cùng người xuất hiện ở cả train và test.
- Nếu có yếu tố thời gian, cân nhắc temporal split.
- Chỉ fit imputer, scaler, encoder, feature selector và resampler trên train fold thông qua Pipeline/ColumnTransformer và imbalanced-learn Pipeline khi cần.
- Không undersample toàn bộ dataset trước khi split. Không cân bằng validation/test.
- Nếu dataset đã cân bằng từ trước, cần truy tìm cách cân bằng và raw data ban đầu.

### 3.5 Khi nào phải tìm lại raw data?

Yêu cầu tìm lại raw dataset nếu gặp một trong các trường hợp:
- Không xác định được nguồn, license hoặc target mapping.
- Dataset hiện tại đã bị undersample/oversample nhưng không còn file trước cân bằng.
- Không rõ các file đã được merge như thế nào hoặc không thể tái lập quy trình.
- BMI/target có dấu hiệu được tạo bằng quy tắc không rõ ràng.
- Dữ liệu thiếu cột/đơn vị hoặc schema không nhất quán.

**Raw data cần tìm/khôi phục:** file nguồn chưa cân bằng, chưa loại mẫu, chưa mã hóa nhãn thủ công; kèm data dictionary hoặc trang nguồn chính thức. Không cần tải lại nếu các file raw hiện có chứng minh được nguồn gốc và quy trình tạo dữ liệu.

## 4. Thiết kế kiến trúc mục tiêu

Luồng hoạt động:

`User → Web UI → FastAPI → Input Validation → ML Pipeline → Prediction → JSON Response → Web UI`

Luồng phát triển mô hình:

`Raw Data → Data Audit → Preprocessing Pipeline → Train/Validation → Model Selection → Final Test → Model Artifact + Metrics`

Nguyên tắc:
- Training code không nằm trong API request lifecycle.
- API chỉ load model đã train một lần khi startup.
- Tiền xử lý lúc inference phải giống hệt lúc training; đóng gói preprocessing và estimator trong cùng sklearn Pipeline.
- Không hard-code mapping category ở nhiều nơi; dùng một schema/contract thống nhất.
- Có version cho model artifact và metadata: model version, feature list, target mapping, training date, dataset version và metrics.
- Lỗi input trả về HTTP 422; lỗi model/service trả về HTTP 5xx có thông tin an toàn, không lộ stack trace.

## 5. Cấu trúc thư mục đề xuất

```text
ObesityPrediction/
├── app/
│   ├── main.py                  # FastAPI application
│   ├── api/
│   │   └── routes/
│   │       ├── health.py
│   │       └── predict.py
│   ├── schemas/
│   │   └── prediction.py        # Pydantic request/response
│   ├── services/
│   │   └── prediction_service.py
│   ├── ml/
│   │   ├── model_loader.py
│   │   └── preprocessing.py     # shared feature/schema helpers
│   └── core/
│       └── config.py
├── data/
│   ├── raw/                     # raw files; do not modify
│   └── processed/               # generated data; normally gitignored
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   └── training/
│       ├── audit_data.py
│       ├── build_dataset.py
│       ├── train.py
│       ├── evaluate.py
│       └── report.py
├── models/                      # selected model + metadata
├── reports/
│   ├── data_audit.md
│   └── model_evaluation.md
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── tests/
│   ├── test_health.py
│   ├── test_predict.py
│   └── test_preprocessing.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── .env.example
├── pyproject.toml
├── README.md
├── PLAN.md
└── Dockerfile                   # optional, after local flow works
```

Có thể giữ tên file `plan.md` theo yêu cầu hiện tại; cấu trúc trên dùng `PLAN.md` để minh họa vị trí tài liệu, không bắt buộc đổi chữ hoa/thường.

## 6. Kế hoạch xây dựng theo từng giai đoạn

### Phase 0 — Repository cleanup & reproducibility

**Công việc**
- [ ] Tạo branch phát triển riêng trước khi thay đổi cấu trúc.
- [ ] Xác định default branch, kiểm tra trạng thái Git và giữ lại lịch sử cần thiết.
- [ ] Xóa venv khỏi tracking; bổ sung .gitignore.
- [ ] Chuẩn hóa tên file, đường dẫn và cấu hình.
- [ ] Tạo pyproject.toml hoặc requirements.txt có version dependencies.
- [ ] Viết README với setup và cách chạy hiện tại.

**Acceptance criteria**
- Máy mới có thể tạo môi trường và cài dependencies theo hướng dẫn.
- Không cần dùng thư mục venv được commit.
- Các đường dẫn dữ liệu/model được quản lý tập trung.

### Phase 1 — Data audit & EDA

**Công việc**
- [ ] Xác minh nguồn, license, schema, đơn vị và target mapping.
- [ ] Thống kê missing, duplicate, outlier và class distribution.
- [ ] Kiểm tra logic BMI và nguy cơ leakage.
- [ ] Kiểm tra quan hệ/độ trùng lặp giữa các CSV.
- [ ] Lưu báo cáo audit, biểu đồ EDA và quyết định giữ/bỏ feature.
- [ ] Quyết định có cần tìm lại raw dataset hay không.

**Acceptance criteria**
- Có data dictionary và mô tả target rõ ràng.
- Có thể tái lập việc tạo processed dataset từ raw data.
- Có kết luận bằng văn bản về BMI leakage và class imbalance.
- Nếu không xác minh được target, dừng modeling và tìm lại dữ liệu.

### Phase 2 — ML pipeline & model selection

**Bài toán**
- Classification đa lớp hay nhị phân phải theo đúng target đã xác minh; không ép nhãn về 0/1 nếu nguồn có nhiều mức độ béo phì.

**Baseline đề xuất**
1. DummyClassifier — baseline tối thiểu.
2. LogisticRegression — baseline dễ giải thích.
3. DecisionTreeClassifier — mô hình dễ trực quan hóa.
4. RandomForestClassifier — mô hình ensemble mạnh, dễ làm mốc so sánh.
5. HistGradientBoostingClassifier hoặc GradientBoosting — mô hình bổ sung nếu dữ liệu phù hợp.

**Preprocessing**
- Numeric: imputation và scaling cho mô hình cần scaling.
- Categorical: imputation và OneHotEncoder(handle_unknown="ignore").
- Kết hợp bằng ColumnTransformer và Pipeline.
- Với class imbalance, thử class_weight hoặc resampling chỉ trên train fold; so sánh với baseline không cân bằng.
- Hyperparameter tuning bằng RandomizedSearchCV/GridSearchCV trên training data với stratified CV phù hợp.

**Evaluation**
- Accuracy chỉ là chỉ số phụ.
- Báo cáo precision, recall, F1 theo từng lớp; macro-F1, weighted-F1, balanced accuracy.
- Confusion matrix; ROC-AUC OVR/PR-AUC nếu phù hợp và có xác suất/score.
- Báo cáo cross-validation mean ± std.
- Đánh giá final test đúng một lần sau khi chốt model.
- Lưu metric, confusion matrix và model comparison table vào `reports/model_evaluation.md`.

**Acceptance criteria**
- Có baseline và bảng so sánh mô hình.
- Không có preprocessing leakage.
- Có kết quả final test có thể tái lập.
- Lựa chọn model dựa trên metric phù hợp với mục tiêu, không chỉ chọn accuracy cao nhất.
- Model artifact chứa toàn bộ preprocessing cần thiết.

### Phase 3 — FastAPI backend

**Endpoints tối thiểu**
- `GET /health`: trạng thái ứng dụng và model đã load hay chưa.
- `GET /metadata`: model version, feature contract và danh sách nhãn đã xác minh.
- `POST /api/v1/predict`: nhận dữ liệu hợp lệ, trả prediction và probabilities (nếu hỗ trợ).

**Request/response**
- Pydantic schema với kiểu dữ liệu, required fields, giới hạn hợp lý và validation cross-field.
- Đơn vị Height/Weight phải ghi rõ trong API contract.
- Không tự tính BMI ở frontend và backend theo hai cách khác nhau. Chốt một nguồn tính BMI sau data audit.
- Response có request ID, predicted label, class probabilities (nếu có), model version và disclaimer.
- Không lưu thông tin định danh hoặc dữ liệu sức khỏe cá nhân trong log.

**Acceptance criteria**
- Swagger docs hoạt động tại `/docs`.
- Test các trường hợp hợp lệ, thiếu trường, sai kiểu, giá trị ngoài phạm vi và model chưa sẵn sàng.
- API không train model khi nhận request.

### Phase 4 — Simple web UI

**Màn hình chính**
- Tiêu đề và mô tả ngắn về mục đích học thuật.
- Form nhập đúng các feature đã chốt.
- Gợi ý đơn vị và giải thích ngắn cho trường khó hiểu.
- Nút “Phân tích” có trạng thái loading và chống gửi nhiều lần.
- Khu vực kết quả hiển thị nhãn, xác suất (nếu được hỗ trợ), thông tin đầu vào và cảnh báo giới hạn.
- Responsive cho desktop/mobile, tiếng Việt, accessibility cơ bản.
- Có nút nhập lại/xóa form.

Frontend gọi FastAPI qua `fetch()`; không nhúng model hoặc quy tắc phân loại vào JavaScript.

**Acceptance criteria**
- Người dùng không cần biết API để sử dụng.
- Lỗi validation/network hiển thị dễ hiểu.
- Kết quả không dùng từ ngữ khẳng định chẩn đoán.

### Phase 5 — Testing, CI & packaging

- [ ] Unit tests cho data validation, preprocessing và prediction service.
- [ ] API integration tests bằng TestClient/HTTPX.
- [ ] Test model artifact có thể load trong môi trường sạch.
- [ ] CI chạy lint và pytest khi push/pull request.
- [ ] Thêm Dockerfile/docker-compose nếu môi trường local ổn định.
- [ ] Kiểm tra CORS, cấu hình secrets qua environment variables.
- [ ] Không bật debug mode khi deploy.
- [ ] Ghi lại cách train lại model và cách thay model artifact.

**Acceptance criteria**
- Test suite chạy tự động và pass.
- Có hướng dẫn chạy local từ đầu đến cuối.
- Có thể khởi động backend và frontend theo README.

### Phase 6 — Documentation & CV presentation

- [ ] README chuyên nghiệp: problem statement, dataset source/license, architecture, setup, API usage, screenshots, evaluation và limitations.
- [ ] Sơ đồ architecture và ML workflow.
- [ ] Ảnh chụp UI và Swagger docs.
- [ ] Bảng metric thực tế, không tự tạo số liệu.
- [ ] Nêu rõ phần việc đã thực hiện, công nghệ và quyết định kỹ thuật.
- [ ] Demo video ngắn 1–2 phút (form → API → result).
- [ ] Gắn release/tag khi có phiên bản demo ổn định.

## 7. API contract dự kiến

Ví dụ minh họa dưới đây chỉ là cấu trúc, các trường cuối cùng phải được xác nhận từ data audit:

```json
{
  "gender": 1,
  "age": 25,
  "height_m": 1.70,
  "weight_kg": 68.0,
  "family_history_with_overweight": 1,
  "ch2o": 2,
  "faf": 1
}
```

Response dự kiến:

```json
{
  "model_version": "v1",
  "predicted_class": "LABEL_FROM_VERIFIED_DATASET",
  "probabilities": {
    "LABEL_A": 0.72,
    "LABEL_B": 0.28
  },
  "disclaimer": "Kết quả chỉ phục vụ mục đích học tập, không thay thế đánh giá của chuyên gia y tế."
}
```

Không coi các giá trị ví dụ hoặc tên nhãn trong JSON là schema chính thức. Chỉ đưa BMI vào request nếu sau audit quyết định BMI là feature hợp lệ; nếu BMI được suy ra từ height/weight thì tránh yêu cầu người dùng nhập lặp lại.

## 8. Rủi ro và quyết định cần chốt

| Rủi ro | Tác động | Biện pháp |
|---|---|---|
| Target không có nguồn xác thực | Không thể diễn giải kết quả như chẩn đoán | Tìm raw data và metadata; đổi phạm vi thành bài toán phân loại nhãn dataset nếu cần |
| Target được tạo từ BMI | Leakage, metric cao giả tạo | Loại BMI hoặc đổi bài toán; làm thí nghiệm đối chứng |
| Dữ liệu đã cân bằng trước khi chia | Test không phản ánh phân phối thực | Khôi phục raw data, split trước resampling |
| Nhãn nhị phân hóa từ nhãn đa lớp | Mất ý nghĩa và sai mapping | Giữ nguyên target gốc, xác minh label encoder |
| Dataset nhỏ/thiên lệch | Khả năng tổng quát hóa thấp | Cross-validation, báo cáo limitation, tìm dữ liệu độc lập |
| Dùng mô hình học thuật như chẩn đoán | Gây hiểu lầm và rủi ro sức khỏe | Disclaimer rõ ràng, không đưa khuyến nghị điều trị |

## 9. Definition of Done — tiêu chí project đủ tốt để đưa vào CV

Project chỉ được xem là hoàn thành khi:
- [ ] Có nguồn dữ liệu, license, data dictionary và target definition rõ ràng.
- [ ] Có quy trình từ raw data đến model artifact tái lập được.
- [ ] Có xử lý class imbalance và target leakage đúng phương pháp.
- [ ] Có so sánh ít nhất 3 mô hình và báo cáo metric phù hợp.
- [ ] Có FastAPI REST API với schema, validation và Swagger docs.
- [ ] Có giao diện web responsive gọi API thật.
- [ ] Có unit/integration tests và CI cơ bản.
- [ ] Có README, architecture diagram, ảnh demo và hướng dẫn chạy.
- [ ] Không tuyên bố model có khả năng chẩn đoán y khoa hoặc hiệu năng chưa được đo.

### Mô tả CV tham khảo (chỉ sử dụng sau khi hoàn thành)

**Obesity Classification Web Application | Python, scikit-learn, FastAPI**

- Built an end-to-end supervised machine learning workflow for obesity-category classification, including data validation, preprocessing, model comparison and evaluation.
- Developed RESTful prediction APIs with FastAPI and Pydantic, and integrated a lightweight responsive web interface.
- Evaluated classification performance using per-class precision/recall/F1, macro-F1, balanced accuracy and confusion matrix; documented data limitations and leakage checks.

Chỉ điền số liệu hiệu năng sau khi chạy đánh giá trên test set độc lập.

## 10. Thứ tự thực hiện đề xuất

1. **Ưu tiên số 1:** Xác minh dataset, target mapping, BMI leakage và raw data. Chưa làm UI/API trước khi chốt được data contract.
2. Chuẩn hóa repository và môi trường chạy.
3. Xây pipeline huấn luyện/đánh giá tái lập được.
4. Chọn và lưu model cùng metadata.
5. Phát triển FastAPI và test API.
6. Làm giao diện web đơn giản.
7. Hoàn thiện CI, README, ảnh demo và nội dung CV.

**Điểm bắt đầu ngay:** mở `dataset/obesity_prediction.csv`, `dataset/obesity_prediction_dataset_final.csv`, `dataset/data_bmi.csv`, `dataset/data_weight_height.csv` để truy nguồn và so sánh schema; xác minh cách sinh cột `Obesity`. Nếu không có tài liệu gốc chứng minh target mapping, hãy tìm lại raw dataset trước khi huấn luyện mô hình mới.
