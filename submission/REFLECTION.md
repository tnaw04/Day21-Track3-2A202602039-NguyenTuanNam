# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Việc chat template có thể âm thầm bỏ qua toàn bộ loss nếu không có cờ hoặc cấu hình đúng, và run `attn_only` dù có training loss thấp hơn nhưng target accuracy thực tế lại kém hơn `correct` tới 5.5% do overfit cục bộ khi tăng rank quá cao trên ít module.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Mất nhiều thời gian nhất ở việc kiểm tra tính đúng đắn của loss mask (`mask_proof.json`) và hiểu rõ cơ chế đối chứng tham số giữa `correct` và `attn_only`. Ban đầu tôi nghĩ thời gian chủ yếu nằm ở khâu train, nhưng thực tế khâu chuẩn bị dữ liệu và verify mask mới là khâu cần cẩn trọng nhất.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước đây tôi tin rằng training loss càng thấp thì mô hình càng giỏi và chỉ cần tăng rank LoRA thật cao ở Attention là đủ. Sau lab này, tôi hiểu rằng vị trí gắn adapter trên toàn bộ text-linear quan trọng hơn nhiều so với việc ép rank cao ở Attention, và training loss thấp không đồng nghĩa với target accuracy cao.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để giải thích lý thuyết LoRA/QLoRA, rà soát logic code và hỗ trợ debug device mapping. AI ban đầu có xu hướng gợi ý hardcode `device_map="auto"` và `bf16=True`, gây lỗi trên môi trường CPU/T4 không hỗ trợ bf16, đòi hỏi phải can thiệp điều chỉnh lại cấu hình device linh hoạt.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên là xây dựng một bộ đánh giá tiêu chuẩn và đo một Honest Baseline thật vững chắc (Prompting tối ưu) cùng bộ kiểm tra hồi quy (Regression Gate) trước khi bắt tay vào train, đồng thời giải mã ngược loss mask để chắc chắn mô hình chỉ tính loss trên output mong muốn.
