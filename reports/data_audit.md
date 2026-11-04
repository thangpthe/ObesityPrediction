# Data Audit Report — ObesityPrediction

> **Ngày kiểm tra:** 2026-11-04  
> **Phạm vi:** Chỉ đọc các file hiện có trong repository; không sửa/xóa dữ liệu gốc, không huấn luyện mô hình.  
> **Kết luận:** <strong>BLOCKED FOR MODEL TRAINING</strong> cho đến khi xác minh nguồn dữ liệu và định nghĩa nhãn.

## 1. Executive summary

Repository hiện có ba CSV đọc được, mỗi file 17.698 bản ghi, và một file raw được tham chiếu trong cấu hình nhưng rỗng.

| File | Kích thước | Số dòng dữ liệu | Số cột | Dòng trùng hoàn toàn |
|---|---:|---:|---:|---:|
| `dataset/obesity_prediction.csv` | 0 byte | — | — | — |
| `dataset/obesity_prediction_dataset_final.csv` | 509.992 bytes | 17.698 | 9 | 109 |
| `dataset/data_bmi.csv` | 334.258 bytes | 17.698 | 7 | 1.112 |
| `dataset/data_weight_height.csv` | 405.869 bytes | 17.698 | 8 | 561 |

Tất cả ba file CSV đọc được đều không có ô dữ liệu thiếu. Cả ba có phân phối nhãn cân bằng tuyệt đối: `Obesity=0: 8.849`, `Obesity=1: 8.849`.

Điều này phù hợp với dấu hiệu dữ liệu đã qua bước cân bằng lớp, không chứng minh đây là phân phối tự nhiên của quần thể.

## 2. Schema của dataset final

File `dataset/obesity_prediction_dataset_final.csv` có schema:

| Cột | Kiểu/miền quan sát | Ghi chú |
|---|---|---|
| Gender | integer, 0–1 | Mapping giới tính chưa có data dictionary |
| Age | integer, 14–61 | Tuổi |
| Height | float, 1,45–1,98 | Có vẻ là mét; cần xác nhận |
| Weight | float, 39–173 | Có vẻ là kg; cần xác nhận |
| BMI | float, 12,87–55,15 | Đối chiếu công thức ở mục 4 |
| family_history_with_overweight | integer, 0–1 | Mapping yes/no chưa được xác minh |
| CH2O | integer, 1–3 | Mã hóa mức uống nước; chưa có codebook |
| FAF | integer, 0–3 | Mã hóa tần suất vận động; chưa có codebook |
| Obesity | integer, 0–1 | **Chưa xác minh ý nghĩa/mapping nhãn** |

### Thống kê số học của final dataset

| Feature | Min | Max | Mean |
|---|---:|---:|---:|
| Age | 14 | 61 | 24,082 |
| Height | 1,45 | 1,98 | 1,708 |
| Weight | 39 | 173 | 88,348 |
| BMI | 12,87 | 55,15 | 30,087 |
| CH2O | 1 | 3 | 2,017 |
| FAF | 0 | 3 | 1,047 |

Tỷ lệ `family_history_with_overweight=1` là khoảng 82,89%; `Gender=1` là khoảng 53,61%. Không diễn giải các mã 0/1 khi chưa có codebook.

## 3. Kiểm tra nhãn và phân phối BMI

| Nhãn quan sát | Số mẫu | BMI min | BMI max | BMI trung bình |
|---|---:|---:|---:|---:|
| 0 | 8.849 | 12,87 | 36,71 | 23,439 |
| 1 | 8.849 | 16,51 | 55,15 | 36,734 |

Nếu thử quy tắc đơn giản `BMI >= 30 => Obesity=1`, có **675/17.698 bản ghi (3,81%) không khớp**:
- 408 bản ghi có nhãn 1 nhưng BMI dưới 30.
- 267 bản ghi có nhãn 0 nhưng BMI từ 30 trở lên.

Do đó, nhãn nhị phân hiện tại không phải phép biến đổi xác định đơn giản từ ngưỡng BMI 30. Đây không phải bằng chứng rằng BMI không gây leakage: nhãn có thể được tạo bằng ngưỡng khác, nhiều điều kiện, hoặc được kế thừa từ nguồn khác.

Có 52 nhóm feature vector giống nhau nhưng mang nhiều hơn một giá trị target. Điều này cần truy vết nguồn, quy tắc hợp nhất và cách gán nhãn trước khi học máy.

**Không xác nhận được mapping `0 = không béo phì`, `1 = béo phì` từ dữ liệu hoặc metadata hiện có.** App Flask cũ đang hard-code cách hiểu này; code ứng dụng không phải bằng chứng nguồn gốc nhãn.

## 4. Kiểm tra tính nhất quán BMI

Giả sử Height tính bằng mét và Weight tính bằng kg:

`BMI = Weight / Height²`

Trên 17.698 dòng của final dataset:

| Thống kê sai số tuyệt đối | Giá trị |
|---|---:|
| Median | 0,00253 |
| Mean | 0,00971 |
| P90 | 0,00489 |
| P95 | 0,06409 |
| P99 | 0,17020 |
| Max | 0,27309 |
| Sai số > 0,02 | 1.268 dòng |
| Sai số > 0,1 | 618 dòng |

Phần lớn sai số nhỏ, nhưng nhóm sai số lớn hơn đáng kể không thể giải thích đơn thuần bằng làm tròn hai chữ số. Cần kiểm tra nguồn, thao tác làm tròn, cách ghép bản ghi và đơn vị trước khi tính lại BMI hoặc xóa mẫu.

## 5. Kiểm tra trùng lặp và nguồn dữ liệu

- Final dataset: 109 dòng trùng toàn bộ 9 cột.
- `data_bmi.csv`: 1.112 dòng trùng toàn bộ 7 cột.
- `data_weight_height.csv`: 561 dòng trùng toàn bộ 8 cột.

Hai file thành phần có ít feature hơn, nên duplicate count của chúng có thể cao hơn khi các bản ghi khác nhau ở feature bị lược bỏ. Không thể chỉ dựa vào số duplicate này để kết luận các bản ghi là cùng một cá nhân.

### Dấu vết trong code

1. `scripts/config.py` trỏ đến `data/raw/obesity_prediction.csv` và `data/raw/obesity_prediction_dataset_final.csv`, nhưng các file trong repo lại ở `dataset/`.
2. `scripts/pre_processing.py` mô tả quy trình load dữ liệu, bỏ missing, tính BMI nếu chưa có, rồi cân bằng lớp trước khi ghi file final.
3. Root `pre-processing.py` tham chiếu `dataset/obesity_with_bmi.csv`, file này không tồn tại trong danh sách file đã kiểm tra.
4. `dataset/obesity_prediction.csv` tồn tại nhưng rỗng.
5. Chưa có README, data dictionary, thông tin nguồn/license hay script chứng minh cách tích hợp và tạo nhãn.

Một báo cáo dự án thứ cấp trên Studocu mô tả việc tích hợp hai dataset, loại 3.878 bản ghi trùng để còn 18.891 dòng, sau đó cân bằng hai lớp còn 17.698 dòng. Con số 17.698 trùng với số dòng trong repository. Đây là một **đầu mối truy vết**, không thay thế raw files hay tài liệu gốc.

## 6. Nguồn dữ liệu tham khảo cần đối chiếu

Một dataset công khai có schema nhân khẩu học/lối sống tương tự là UCI **Estimation of Obesity Levels Based on Eating Habits and Physical Condition**:
- UCI: https://archive.ics.uci.edu/dataset/544/estimation%2Bof%2Bobesity%2Blevels%2Bbased%2Bon%2Beating%2Bhabits%2Band%2Bphysical%2Bcondition
- DOI: https://doi.org/10.24432/C5H31Z
- License: CC BY 4.0.
- Quy mô: 2.111 dòng, 16 features, target gốc có 7 mức: Insufficient Weight, Normal Weight, Overweight Level I/II, Obesity Type I/II/III.
- Nguồn UCI ghi nhận 77% bản ghi của dataset UCI được tạo tổng hợp bằng Weka/SMOTE, 23% thu thập từ người dùng.

**Không được tự kết luận các CSV 17.698 dòng trong repo chính là dataset UCI.** Quy mô, target và schema đã khác; cần chứng minh quan hệ lineage bằng raw data, code hoặc metadata.

## 7. Những file cần tìm lại

Ưu tiên xin lại/khôi phục theo thứ tự:

1. **Raw nguồn A:** file `obesity_prediction.csv` không rỗng, đúng file được tham chiếu trong `scripts/config.py`.
2. **Raw nguồn B:** file `obesity_with_bmi.csv` được tham chiếu trong `pre-processing.py`.
3. **Bản tích hợp trước cân bằng:** có thể là file từng chứa 18.891 dòng theo đầu mối báo cáo thứ cấp.
4. **Data dictionary:** giải thích từng cột, đơn vị, mapping mã hóa và cách tạo `Obesity`.
5. **Tài liệu quy trình:** quy tắc loại duplicate, merge key, tạo BMI, chuyển nhãn nhiều lớp thành nhị phân và seed của bước resampling.

Nếu không tìm được raw A/B và data dictionary, nên thay bằng một dataset có nguồn chính thức và target được định nghĩa rõ ràng. Nếu dùng UCI 544, hãy xây bài toán phân loại 7 mức nguyên bản trước; không tự chuyển thành nhãn nhị phân nếu không có lý do và quy tắc được ghi rõ.

## 8. Khuyến nghị cho Phase 2

- Không dùng file final hiện tại làm bằng chứng hiệu năng mô hình trên phân phối thực tế.
- Không resample trước khi chia train/validation/test. Chỉ resample trên train fold.
- Giữ một test set tự nhiên, không cân bằng nhân tạo.
- Xác minh target mapping và thời điểm gán nhãn.
- Chạy đối chứng có/không có BMI, Height, Weight. Nếu target được tạo từ BMI/anthropometry, không đưa các biến này vào mô hình nếu mục tiêu là dự đoán độc lập với phép đo đó.
- Không gọi ứng dụng là công cụ chẩn đoán y khoa. Đây là project học thuật phân loại nhãn dữ liệu cho đến khi có validation lâm sàng phù hợp.

## 9. Kết luận

**Phase 1 chưa thể đóng hoàn toàn về mặt provenance.** Kiểm toán kỹ thuật của các file hiện có đã thực hiện được, nhưng nguồn raw, target mapping và lịch sử tích hợp/cân bằng vẫn chưa được xác minh. Khuyến nghị dừng huấn luyện model mới cho đến khi khôi phục được các file và tài liệu ở mục 7.


## 10. New source received after the initial audit

A separate uploaded archive has now been identified as the official UCI 544 obesity-level dataset. It is documented in [uci_dataset_assessment.md](uci_dataset_assessment.md) and its source attribution/integrity details are in [data/raw/README.md](../data/raw/README.md).

This UCI source is **not** the same as the legacy 17,698-row binary dataset. It contains 2,111 rows, 17 columns and the original seven-class `NObeyesdad` target; the UCI publication reports 77% synthetic records generated with Weka/SMOTE. It is therefore a suitable documented educational candidate, but not a pure raw cohort or an independent clinical test set.

The source archive and extracted CSV have not been committed by this documentation update. Place them under `data/raw/` as instructed in the UCI assessment and verify the recorded SHA-256 checksums. The audit script now includes the expected extracted CSV path in its inventory.
