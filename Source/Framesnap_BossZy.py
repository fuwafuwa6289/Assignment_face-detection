import cv2
import os
import time

# รับจำนวนโฟลเดอร์ที่จะสร้างจากผู้ใช้
num_folders = int(input("Please enter the number of folders to create: "))

# สร้างโฟลเดอร์ตามจำนวนที่ผู้ใช้ต้องการ และให้ตั้งชื่อเอง
folder_names = []
for i in range(1, num_folders + 1):
    folder_name = input(f"Please enter the name for folder {i}: ")
    folder_names.append(folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"สร้างโฟลเดอร์ '{folder_name}' สำเร็จ")
    else:
        print(f"ใช้โฟลเดอร์ที่มีอยู่แล้ว: {folder_name}")

# เปิดกล้อง (กล้องหลักจะใช้ index = 0)
cap = cv2.VideoCapture(0)

# ตรวจสอบว่าเปิดกล้องสำเร็จหรือไม่
if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

print("กล้องพร้อมใช้งาน กด 'F' เพื่อเริ่มถ่ายภาพ หรือ 'Q' เพื่อออกจากโปรแกรม")

image_count = 0
folder_index = 0
capture_images = False
start_time = 0

while True:
    # อ่านภาพจากกล้อง
    ret, frame = cap.read()

    if not ret:
        print("ไม่สามารถรับภาพจากกล้องได้")
        break

    # แสดงภาพบนหน้าจอ
    cv2.imshow("Webcam", frame)

    # อ่านค่าจากคีย์บอร์ด
    key = cv2.waitKey(1) & 0xFF

    # กด 'F' เพื่อเริ่มถ่ายภาพในโฟลเดอร์ถัดไป
    if key == ord('f') and not capture_images and folder_index < num_folders:
        capture_images = True
        image_count = 0
        start_time = time.time()  # เริ่มต้นจับเวลา
        print(f"เริ่มถ่ายภาพในโฟลเดอร์ '{folder_names[folder_index]}'")

    # ถ่ายภาพ 10 ภาพ ดีเลย์ 3 วินาทีต่อภาพในโฟลเดอร์ปัจจุบัน
    if capture_images and image_count < 10:
        elapsed_time = time.time() - start_time
        if elapsed_time >= 3:  # ถ้าผ่านไป 3 วินาที
            image_count += 1
            image_path = os.path.join(folder_names[folder_index], f"image_{image_count}.png")
            cv2.imwrite(image_path, frame)
            print(f"บันทึกภาพ {image_count} ที่ {image_path} สำเร็จ")
            start_time = time.time()  # รีเซ็ตเวลาใหม่

        if image_count >= 10:
            print(f"ถ่ายภาพครบ 10 ภาพในโฟลเดอร์ '{folder_names[folder_index]}' แล้ว")
            capture_images = False
            folder_index += 1  # เปลี่ยนไปยังโฟลเดอร์ถัดไป

            if folder_index >= num_folders:
                print("ถ่ายครบทุกโฟลเดอร์แล้ว")
                break  # ออกจากลูปเมื่อครบทุกโฟลเดอร์

    # กด 'Q' เพื่อออกจากโปรแกรม
    if key == ord('q'):
        print("ปิดโปรแกรม")
        break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
