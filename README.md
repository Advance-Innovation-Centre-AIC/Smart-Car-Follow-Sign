## Smart car follow signs 

รถเดินตามป้ายจราจรขนาดเล็กบนบอร์ด Raspberry Pi โดยอาศัยเทคโนโลยีอัลกอริทึมตรวจจับวัตถุและใช้กล้องควบคู่กับเทคโนโลยีทำให้ระบบประมวลผลภาพแบบเรียลไทม์จากกล้องในการตรวจจับและจำแนกป้ายจราจร  ที่ติดตั้งในสภาพแวดล้อมที่กำหนดได้อย่างแม่นยำและรวดเร็ว  โดยจำแนกการเคลื่อนที่ออกเป็น 4 รูปแบบ  ได้แก่  เดินหน้า, หยุด, เลี้ยวซ้าย, และเลี้ยวขวา

**สารบัญ**

 - ขั้นตอนการทำ Dataset และ Ai model
 - โค้ดในการทดสอบย่อยต่างๆ 
 - โค้ดที่เสร็จสมบูรณ์
 - ผลการทดลอง

**ขั้นตอนการทำ Dataset และ Ai model**

 - เตรียมรูปที่เราจะทำการสร้าง dataset ใส่ folder โดยแต่ละชนิดของป้ายใช้อย่างต่ำ 10 รูป
 - ติดตั้ง Label studio เพื่อทำการสร้าง dataset

    pip install label-syudio

 - ใส่คำสั่ง start เพื่อทำการเปิด label studio


    label-studio start
	 

 - เข้าหน้า label studio ได้แล้วทำการสร้างบัญชี และเลือก create
   project
   

 - ตั้งชื่อ project และทำการ upload folder ที่ทำการเตรียมไว้แล้วในขั้นตอนที่ 1    

 - ไปที่ช่อง label setup เลือก label เครื่องบินละลบชื่อคลาสออก ใส่ชื่อที่เราจะทำการสร้างและกด add เพื่อทำการตีกรอบ  
 -  ทำการตีกรอบรูปทั้งหมดและ export โดยเลือก YOLO เพื่อเข้าสู่การเทรนในขั้นตอนต่อไป  
 - เข้าลิงค์เทรนของ google colab ปรับ runtime เลือก T4 GPU และทำตามขั้นตอนในแต่ละบล็อกตามลิงค์
    https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-classification-on-custom-dataset.ipynb

 **โค้ดในการทดสอบย่อยต่างๆ** 
  

 - **Code check open webcam camera**

  เอาไว้ใช้ทดสอบการแสดงผลวิดีโอจากเว็บแคมและควบคุมการตั้งค่าพื้นฐานของกล้อง

    import cv2
    import argparse
    import time
    
    def parse_args():
        ap = argparse.ArgumentParser()
        ap.add_argument("--width", type=int, default=640)
        ap.add_argument("--height", type=int, default=480)
        ap.add_argument("--fps", type=int, default=30)
        ap.add_argument("--index", type=int, default=1, help="Webcam index (usually 0 or 1)")
        return ap.parse_args()
    def main():
        args = parse_args()
        CAMERA_INDEX = args.index
        
        print(f"Opening webcam at index {CAMERA_INDEX}...")
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {CAMERA_INDEX}")
        return

    print(f"Webcam opened. Press 'q' to exit.")

    try:
        while True:
      
            ret, frame = cap.read() 
            
            if not ret:
                print("Error: Failed to capture image from webcam.")
                break


            cv2.imshow('Webcam Feed (Press Q to exit)', frame)
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.01) 

    except KeyboardInterrupt:
        pass
    finally:

        cap.release()
        cv2.destroyAllWindows()
        print("Webcam viewer stopped.")
    if _name_ == "_main_":
        main()

 - **Code check YOLO**

ใช้ในการตรวจจับวัตถุแบบเรียลไทม์ โดยเฉพาะ เพื่อใช้ทดสอบโมเดล YOLOv8 ที่ผ่านการเทรนแล้วร่วมกับวิดีโอจากเว็บแคม .



    import argparse
    import cv2
    import torch
    from ultralytics import YOLO
    import numpy as np
    import time
    
    def parse_args():
        ap = argparse.ArgumentParser(description="YOLOv8 Real-Time Detection for Testing.")
        ap.add_argument("--model", type=str, default="best.pt", help="Path to YOLO model weights.")
        ap.add_argument("--imgsz", type=int, default=480, help="Inference size (square).")
        ap.add_argument("--conf", type=float, default=0.2, help="Confidence threshold.")
        ap.add_argument("--width", type=int, default=640, help="Camera width.")
        ap.add_argument("--height", type=int, default=480, help="Camera height.")
        ap.add_argument("--fps", type=int, default=30, help="Camera FPS.")
        ap.add_argument("--skip", type=int, default=0, help="Skip N frames between inference.")
        ap.add_argument("--device", type=str, default="cpu", help="Device to use (e.g., 'cpu' or 'cuda:0').")
        return ap.parse_args()
    
    def run_yolo_test():
        args = parse_args()
 
        try:
            model = YOLO(args.model)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please check if 'best.pt' exists in the current directory.")
            return

        device = args.device
        if device != "cpu" and not torch.cuda.is_available():
            device = "cpu"
        model.to(device)


        CAMERA_INDEX = 1  
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

        if not cap.isOpened():
            print(f"Error: Could not open webcam at index {CAMERA_INDEX}")
            return
    
        time.sleep(0.2)

        print("Warming up YOLO model...")
        _ = model(np.zeros((args.height, args.width, 3), dtype=np.uint8),
                  imgsz=args.imgsz, conf=args.conf, verbose=False)
        print("Ready. Press 'q' to exit.")

        frame_i = 0
        start_time = time.time()
        try:
            while True:
                ret, bgr = cap.read() # อ่านภาพ
                if not ret:
                    print("Error: Failed to capture image from webcam.")
                    break
            
                do_infer = (args.skip == 0) or (frame_i % (args.skip + 1) == 0)

                if do_infer:
                    results = model(bgr, imgsz=args.imgsz, conf=args.conf, verbose=False)
                
                    annotated_frame = results[0].plot()

                else:
                    if 'annotated_frame' not in locals():
                        annotated_frame = bgr.copy()
            

                cv2.putText(annotated_frame, f"FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('YOLO Test Display (Press Q to Quit)', annotated_frame)
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
                frame_i += 1

        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nProgram terminated.")
    if _name_ == "_main_":
        run_yolo_test()

  - **Code คำนวณค่าของ F1 score**

  เป็นการทดสอบการตรวจจับวัตถุ YOLOv8 และการจำลองการคำนวณเมตริก F1 Scoreสำหรับการประเมินผลโมเดล
   

   

        import cv2
        import numpy as np
        import time
        import argparse
        import torch
        from ultralytics import YOLO
        import supervision as sv 
        from supervision.metrics import F1Score
        
        def parse_args():
              ap = argparse.ArgumentParser(description="YOLOv8 Real-Time Detection and F1 Score Simulation.")
              ap.add_argument("--model", type=str, default="/home/bosss/best.pt", help="Path to YOLO model weights (e.g., yolov8n.pt or best.pt).")
              ap.add_argument("--imgsz", type=int, default=480, help="Inference size (square).")
              ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
              ap.add_argument("--device", type=str, default="cpu", help="Device to use (e.g., 'cpu' or 'cuda:0').")
              return ap.parse_args()
        def run_yolo_test():
              print("--- 1. เริ่ม YOLO WebCam Inference ---")
             
              try:
                  args = parse_args()
              except SystemExit:
                  class MockArgs:
                        model = "/home/bosss/best.pt" 
                        imgsz = 240
                        conf = 0.25
                        device = "cpu"
                  args = MockArgs()
               try:
                    model = YOLO(args.model)
               except Exception as e:
                    print(f"❌ Error loading model: {e}")
                    print(f"โปรดตรวจสอบว่าไฟล์ '{args.model}' มีอยู่ในไดเร็กทอรีปัจจุบัน หรือเส้นทางถูกต้อง")
                    return
                device = args.device
                if device != "cpu" and not torch.cuda.is_available():
                    device = "cpu"
                print(f"Running on device: {device}")
                model.to(device)
                
                CAMERA_INDEX = 1
                cap = cv2.VideoCapture(CAMERA_INDEX)
               
                if not cap.isOpened():
                    print(f"❌ Error: ไม่สามารถเปิดกล้องได้ที่ดัชนี {CAMERA_INDEX}")
                    return
                print("Warming up YOLO model...")
                _ = model(np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8),
                print("✅ Ready. Press 'q' to exit Webcam Detection.")
                
                try:
                     while True:
                         ret, bgr = cap.read() # อ่านภาพ
                         if not ret:
                             print("Error: Failed to capture image from webcam.")
                             break
                        
                         results = model(bgr, imgsz=args.imgsz, conf=args.conf, verbose=False)
                         annotated_frame = results[0].plot()
                         cv2.imshow('YOLO Real-Time Detection', annotated_frame)
                        
                         if cv2.waitKey(1) & 0xFF == ord('q'):
                              break
                 except KeyboardInterrupt:
                     print("\nProgram interrupted.")
                 finally:
                     cap.release()
                     cv2.destroyAllWindows()
                     print("--- สิ้นสุด YOLO WebCam Inference ---")
        
         def simulate_f1_score():
             print("\n--- 2. เริ่มจำลองการคำนวณ F1 Score (Validation) ---")
             class_ids_to_evaluate = [0, 1] 
             f1_calculator = F1Score(
                 class_ids=class_ids_to_evaluate, 
                 iou_threshold=0.5
             )
             targets_1 = sv.Detections(
                 xyxy=np.array([[10, 10, 100, 100], [200, 200, 300, 300]]),
                 class_id=np.array([0, 1])
             )
             detections_1 = sv.Detections(
                 xyxy=np.array([[15, 15, 105, 105], [205, 205, 305, 305]]),
                 class_id=np.array([0, 1]),
                 confidence=np.array([0.9, 0.8])
              )
              targets_2 = sv.Detections(
                  xyxy=np.array([[10, 10, 100, 100], [400, 400, 500, 500]]), # GT มี 2 คน
                  class_id=np.array([0, 0])
              )
              detections_2 = sv.Detections(
                  xyxy=np.array([[15, 15, 105, 105], [600, 600, 700, 700]]), # ทำนายถูก 1 คน และทำนายผิดเป็นจักรยาน 1 คัน (FP)
                  class_id=np.array([0, 1]), 
                  confidence=np.array([0.9, 0.7])
              )
                
              f1_calculator.update(detections_1, targets_1)
              f1_calculator.update(detections_2, targets_2) # <--- ถูกต้องแล้ว
              f1_result = f1_calculator.result()
    
              precision = f1_result['precision']
              recall = f1_result['recall']
              f1_score = f1_result['f1']
    
              print("\n--- ผลลัพธ์ F1 Score จากชุดข้อมูลจำลอง (IoU 0.5) ---")
              print(f"🎯 Precision (ความแม่นยำ): {precision:.4f}")
              print(f"🎯 Recall (ความสมบูรณ์): {recall:.4f}")
              print(f"⭐ F1 Score: {f1_score:.4f}") 
    
              print("\n*F1 Score ใช้ในการประเมินโมเดลบน Validation Data เท่านั้น*")
              print("--- สิ้นสุด F1 Score Simulation ---")
         
         if _name_ == "_main_":
             run_yolo_test()
             simulate_f1_score()
    
**โค้ดที่เสร็จสมบูรณ์**

    
            

     

          
    
    


 
    

   


    
   





        
        
    
   

    







 

    







 




