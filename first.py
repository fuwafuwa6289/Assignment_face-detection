import cv2 as cv
import os

# ฟังก์ชันเพื่อรับข้อมูลสมาชิก
def get_members_info():
    members_info = []
    num_members = int(input("Enter the number of members: "))
    
    for _ in range(num_members):
        name = input("Enter the member's name: ")
        position = input("Enter job position: ")
        members_info.append((name, position))
        
    return members_info

# รับข้อมูลสมาชิก
members_info = get_members_info()

cam = cv.VideoCapture(0)

cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

image_counter = 1
frame_count = 0
frame_save = 60

# จำนวนรูปต่อคน
images_per_member = 10

# สร้างโฟลเดอร์ output หากยังไม่มี
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

member_index = 0

cv.namedWindow("Output", cv.WINDOW_NORMAL)
cv.setWindowProperty("Output", cv.WND_PROP_TOPMOST, 1)

while True:
    check, frame = cam.read()
    if check:
        # Flip the image horizontally
        frame = cv.flip(frame, 1)
        
        # แปลงภาพเป็นขาวดำเพื่อใช้ในการตรวจจับใบหน้า
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # ตรวจจับใบหน้า
        detect = cascade.detectMultiScale(gray_img, 1.3, 8)
        1
        # วาดกรอบสี่เหลี่ยมรอบใบหน้า และบันทึกภาพเมื่อเจอใบหน้า
        for (x, y, w, h) in detect:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if frame_count % frame_save == 0:
                if image_counter <= images_per_member:
                    # สร้างโฟลเดอร์สำหรับตำแหน่งงานหากยังไม่มี
                    position_folder = os.path.join(output_folder, members_info[member_index][1])  # ใช้ตำแหน่งงาน
                    if not os.path.exists(position_folder):
                        os.makedirs(position_folder)

                    # ตั้งชื่อไฟล์รูปภาพ
                    img_name = f"{image_counter}_{members_info[member_index][0]}.jpg"
                    img_path = os.path.join(position_folder, img_name)
                    cv.imwrite(img_path, frame)
                    print(f"{img_name} Save complete!")
                    image_counter += 1
                
                # หากถ่ายครบ 10 รูปแล้ว เปลี่ยนไปถ่ายสมาชิกคนถัดไป
                if image_counter > images_per_member:
                    print(f"Completed taking 10 photos for {members_info[member_index][0]}")
                    
                    # เช็คว่าถึงคนสุดท้ายหรือยัง
                    if member_index < len(members_info) - 1:
                        print("Press 'n' to switch to the next member, or 'e' to exit")
                    else:
                        print("Completed taking photos of everyone!")

                    # ลูปเพื่อรอการเปลี่ยนสมาชิก
                    while True:
                        cv.imshow("Output", frame)  # แสดงผลเฟรมในขณะที่รอ
                        key = cv.waitKey(1) & 0xFF
                        if key == ord('n'):
                            member_index += 1
                            image_counter = 1
                            if member_index >= len(members_info):
                                print("All photos have been taken!")
                                cam.release()
                                cv.destroyAllWindows()
                                exit(0)
                            break
                        elif key == ord('e'):
                            cam.release()
                            cv.destroyAllWindows()
                            exit(0)
        
        # แสดงผลเฟรมปัจจุบัน
        cv.imshow("Output", frame)
        
        # เพิ่มค่า frame_counter
        frame_count += 1

        # กด 'e' เพื่อออก
        if cv.waitKey(1) & 0xFF == ord('e'):
            break
    else:
        break

# ปิดกล้องและทำลายหน้าต่างทั้งหมด
cam.release()
cv.destroyAllWindows()

