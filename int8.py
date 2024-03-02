import numpy as np
import cv2
import tensorflow as tf

# Đường dẫn đến file TensorFlow Lite
tflite_model_path = 'yolov5s-int8.tflite'

# Tạo một TensorFlow Lite Interpreter
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Lấy thông tin về đầu vào và đầu ra của mô hình
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Kiểm tra đầu vào của mô hình (thường là tensor có kích thước [1, height, width, 3])
input_shape = input_details[0]['shape']

# Mở video file để đọc
cap = cv2.VideoCapture('Vehicle_Detection.mp4')

while True:
    # Đọc một frame từ video
    ret, frame = cap.read()

    # Chỉnh kích thước frame nếu cần thiết
    frame = cv2.resize(frame, (input_shape[2], input_shape[1]))

    # Chuyển đổi frame thành định dạng phù hợp để đưa vào mô hình
    input_data = np.expand_dims(frame, axis=0)

    # Đặt dữ liệu đầu vào cho mô hình
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Thực hiện inference
    interpreter.invoke()

    # Lấy kết quả từ đầu ra của mô hình
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Xử lý kết quả
    bboxes = output_data[0]

    # Loop qua mỗi bounding box và hiển thị nó trên frame
    for bbox in bboxes:
        # Lấy tọa độ của bounding box
        x_min, y_min, x_max, y_max = map(int, bbox[:4])

        # Hiển thị bounding box trên frame
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow('FRAME', frame)

    # Thoát khỏi vòng lặp nếu nhấn phím 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
