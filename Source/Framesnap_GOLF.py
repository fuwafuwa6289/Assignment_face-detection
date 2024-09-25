import cv2
import os
from tkinter import Tk, filedialog

# ฟังก์ชันสำหรับเลือกโฟลเดอร์
def select_folder():
    root = Tk()
    root.withdraw()  # ซ่อนหน้าต่างหลักของ Tkinter
    folder_selected = filedialog.askdirectory()  # เปิดหน้าต่างเลือกโฟลเดอร์
    return folder_selected

# ฟังก์ชันเปิดกล้องและบันทึกภาพ
def capture_images(folder_name):
    # เปิดกล้อง
    cap = cv2.VideoCapture(0)
    frame_count = 0  # ตัวนับเฟรม
    
    # ตรวจสอบว่ากล้องเปิดได้หรือไม่
    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return
    
    print(f"เริ่มการถ่ายภาพ, รูปภาพจะถูกบันทึกในโฟลเดอร์: {folder_name}")
    
    while True:
        # อ่านภาพจากกล้อง
        ret, frame = cap.read()
        
        if not ret:
            print("ไม่สามารถรับภาพจากกล้องได้")
            break
        
        # แสดงภาพที่ได้จากกล้อง
        cv2.imshow('Camera', frame)

        # รอการกดปุ่ม 'F' เพื่อเริ่มถ่ายภาพ
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('f'):  # ถ้ากดปุ่ม 'F' เริ่มบันทึกภาพ
            frame_count += 1
            file_name = f"{str(frame_count).zfill(3)}_{os.path.basename(folder_name)}.jpg"
            file_path = os.path.join(folder_name, file_name)
            cv2.imwrite(file_path, frame)
            print(f"บันทึกรูปภาพ: {file_name}")

        # ถ้ากดปุ่ม 'q' ให้หยุดการถ่ายภาพ
        if key == ord('q'):
            break
    
    # ปิดกล้องและหน้าต่าง
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # เลือกโฟลเดอร์เพื่อเก็บรูปภาพ
    selected_folder = select_folder()
    
    # ตรวจสอบว่ามีการเลือกโฟลเดอร์หรือไม่
    if selected_folder:
        capture_images(selected_folder)
    else:
        print("ไม่มีการเลือกโฟลเดอร์")
