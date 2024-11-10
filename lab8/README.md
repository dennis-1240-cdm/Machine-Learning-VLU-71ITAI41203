# Bài tập về nhà (Từ tập dũ liệu Wine)
# Công nghệ sử dụng
- Numpy
- Pandas
- Sikit-learn

# Thuật Toán
- KNN: Thuật toán sử dụng cho việc phân loại dữ liệu dựa trên việc tìm ra k điểm gần nhất trong tập huấn luyện của một điểm trong tập kiểm tra.
- Confusion Matrix (Ma trận nhầm lẫn): Ma trận này sẽ cho biết kết quả dự đoán đúng và sai của mô hình
- Các chỉ số đánh giá mô hình:
  + Accuracy (Độ chính xác): Được tính bằng công thức Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
  + Recall (Độ nhạy cảm): Được tính bằng công thức Recall = TP / (TP + FN)
    
  + Specificity (Độ đặc hiểu): Được tính bằng công thức Specificity = TN / (TN + FP)
    
  + Precision (Giá trị dự đoán dương): Được tính bằng công thức Precision = TP / (TP + FP)
    
  + F1 score: Được tính bằng công thức F1 score = (2*(Precision * Recall)) / (Precision + Recall)

- Trong đó:
  + True Negative(TN): Số lượng mẫu thực tế là âm (Negative) và mô hình cũng dự đoán là âm.
  + False Positive(FP): Số lượng mẫu thực tế là âm nhưng mô hình lại dự đoán là dương (Positive). Đây còn gọi là dương giả.
  + False Negative(FN): Số lượng mẫu thực tế là dương nhưng mô hình lại dự đoán là âm. Đây còn gọi là âm giả.
  + True Positive(TP): Số lượng mẫu thực tế là dương và mô hình cũng dự đoán là dương.
# Kết quả
<img width="851" alt="Screenshot 2024-11-10 at 18 15 25" src="https://github.com/user-attachments/assets/1a7a3f4f-7150-4579-9b68-eea99f38fc67">
