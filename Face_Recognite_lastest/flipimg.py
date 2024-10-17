import cv2
import os

# กำหนด path ที่เก็บภาพต้นฉบับ และ path สำหรับเก็บภาพที่ถูก flip
input_folder = 'datasets/Nine'  # โฟลเดอร์ที่มีภาพ 200 รูป
output_folder = 'output/Nine'  # โฟลเดอร์สำหรับเก็บภาพ flip

# ตรวจสอบว่ามีโฟลเดอร์ output อยู่หรือไม่ ถ้าไม่มีให้สร้าง
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# เริ่มนับจากภาพ 201
start_number = 201

# ดึงไฟล์ทั้งหมดในโฟลเดอร์ input
files = sorted(os.listdir(input_folder))

# ทำการ Flip ภาพทั้งหมดและบันทึกผลลัพธ์
for i, file_name in enumerate(files):
    # ตรวจสอบว่าเป็นไฟล์รูปภาพหรือไม่
    if file_name.endswith(('.png', '.jpg', '.jpeg')):
        # อ่านภาพจาก input_folder
        img = cv2.imread(os.path.join(input_folder, file_name))
        
        # ทำการ flip ภาพ (flip along y-axis: horizontal flip)
        flipped_img = cv2.flip(img, 1)
        
        # สร้างชื่อไฟล์ใหม่สำหรับภาพ flip
        output_file_name = f'{start_number + i}.jpg'
        output_path = os.path.join(output_folder, output_file_name)
        
        # บันทึกภาพที่ถูก flip
        cv2.imwrite(output_path, flipped_img)

print("Flip and save completed.")
