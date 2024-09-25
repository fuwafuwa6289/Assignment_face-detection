import cv2
import os

# สร้างโฟลเดอร์สำหรับบันทึกภาพถ้ายังไม่มี
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# เปิดกล้องหลัก
cam = cv2.VideoCapture(0)

# โหลด Haarcascade สำหรับการตรวจจับใบหน้า
haar_cascade = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml')

frame_count = 0

while True:
    ret, frame = cam.read()

    if not ret:
        print("ไม่สามารถอ่านข้อมูลจากกล้องได้")
        break

    # แปลงภาพเป็น grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้าในภาพ grayscale
    faces_rect = haar_cascade.detectMultiScale(gray_frame, 1.1, 9)

    # วาดสี่เหลี่ยมรอบๆ ใบหน้าที่ตรวจพบ
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # แสดงภาพที่มีการตรวจจับใบหน้า
    cv2.imshow('Camera - Face Detection', frame)

    # บันทึกภาพหนึ่งภาพต่อหนึ่งวินาที (แต่ยังคงแสดงวิดีโอแบบเรียลไทม์)
    if frame_count % 20 == 0:  # สมมติว่ากล้องมีเฟรมเรตประมาณ 20 FPS, บันทึกภาพทุกๆ 1 วินาที
        frame_filename = os.path.join(output_folder, f"frame_{frame_count//20}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"บันทึก {frame_filename}")

    frame_count += 1

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) == ord('q'):
        break

# ปล่อยการเชื่อมต่อกล้อง
cam.release()
cv2.destroyAllWindows()
