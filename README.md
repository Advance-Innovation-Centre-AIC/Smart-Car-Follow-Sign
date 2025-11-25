## Smart car follow signs 

รถเดินตามป้ายจราจรขนาดเล็กบนบอร์ด Raspberry Pi โดยอาศัยเทคโนโลยีอัลกอริทึมตรวจจับวัตถุและใช้กล้องควบคู่กับเทคโนโลยีทำให้ระบบประมวลผลภาพแบบเรียลไทม์จากกล้องในการตรวจจับและจำแนกป้ายจราจร  ที่ติดตั้งในสภาพแวดล้อมที่กำหนดได้อย่างแม่นยำและรวดเร็ว  โดยจำแนกการเคลื่อนที่ออกเป็น 4 รูปแบบ  ได้แก่  เดินหน้า, หยุด, เลี้ยวซ้าย, และเลี้ยวขวา

**สารบัญ**

 - ขั้นตอนการทำ Dataset และ Ai model
 - โค้ดในการทดสอบย่อยต่างๆ 
 - โค้ดที่เสร็จสมบูรณ์

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

    
       import time
       import atexit
       import argparse, cv2, torch
       from ultralytics import YOLO
       import numpy as np
       from skimage.metrics import structural_similarity as ssim


       MODEL = "best.pt"
       CONF = 0.20 # ใช้ค่าเริ่มต้นที่ต่ำลงตาม args
       IMG_SZ = 480 # ใช้ค่าเริ่มต้นตาม args
       TPL_R = "1.png" # ต้องมีไฟล์ภาพป้ายขวา
       TPL_L = "2.png" # ต้องมีไฟล์ภาพป้ายซ้าย
       PAD = 0.03
       COMPARE_SZ = (160,160)
       USE_ORB = False
       W_SSIM, W_EDGE, W_ORB = 0.56, 0.44, 0.0
       ALLOWED = {s.upper() for s in ("left", "right", "stop", "go")}


       IN1, IN2 = 17, 27
       IN3, IN4 = 22, 23
       ENA, ENB = 18, 19
       PWM_FREQ = 1000

       A_INVERT = False
       B_INVERT = False

       pwmA = None
       pwmB = None


       SPEED_FWD = 60
       SPEED_TURN = 60
       TURN_DUR = 0.5 
       STOP_DUR = 1.0
       COMMAND_HOLD = 0.0 

       ACTION_DURATIONS = {
           "LEFT": TURN_DUR,
           "RIGHT": TURN_DUR,
           "STOP": STOP_DUR,
           "GO": 0.0, 
       }


       current_action = None
       action_in_progress = False
       action_end_time = 0.0

 
       def _clamp_speed(p):
           return max(0, min(100, int(p)))

       try:
           import RPi.GPIO as GPIO
           IS_PI = True
       except Exception:
           IS_PI = False

           class _MockGPIO:
               BCM = "BCM"; OUT = "OUT"; HIGH = 1; LOW = 0
               def setmode(self, m): print("[MOCK GPIO] setmode", m)
               def setwarnings(self, v): pass
               def setup(self, pins, mode): print("[MOCK GPIO] setup", pins, mode)
               def output(self, pins, val): print("[MOCK GPIO] output", pins, val)
               def PWM(self, pin, freq):
                   class _PWM:
                       def __init__(self, pin): self.pin = pin
                       def start(self, dc): pass
                       def ChangeDutyCycle(self, dc): pass
                       def stop(self): pass
                   return _PWM(pin)
               def cleanup(self): print("[MOCK GPIO] cleanup")
           GPIO = _MockGPIO()
       def setup():
           global pwmA, pwmB
           GPIO.setmode(GPIO.BCM)
           GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)
           GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
           pwmA = GPIO.PWM(ENA, PWM_FREQ)
           pwmB = GPIO.PWM(ENB, PWM_FREQ)
           pwmA.start(0)
           pwmB.start(0)
           atexit.register(_safe_shutdown)

       def _ensure_pwm():
           if pwmA is None or pwmB is None:
               raise RuntimeError("PWM not initialized: call setup() before driving motors")

       def _drive_motor_a(direction: int, speed: int):
           _ensure_pwm()
           s = _clamp_speed(speed)
           d = direction
           if A_INVERT: d = -d
           if d > 0: GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
           elif d < 0: GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.HIGH)
           else: GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.LOW)
           pwmA.ChangeDutyCycle(s if d != 0 else 0)

       def _drive_motor_b(direction: int, speed: int):
           _ensure_pwm()
           s = _clamp_speed(speed)
           d = direction
           if B_INVERT: d = -d
           if d > 0: GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
           elif d < 0: GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.HIGH)
           else: GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.LOW)
           pwmB.ChangeDutyCycle(s if d != 0 else 0)

       def forward(speed=SPEED_FWD):
           _drive_motor_a(-1, speed)
           _drive_motor_b(1, speed)

       def backward(speed=SPEED_FWD):
           _drive_motor_a(1, speed)
           _drive_motor_b(-1, speed)

       def left(speed=SPEED_TURN):
           _drive_motor_a(1, speed)
           _drive_motor_b(1, speed)

       def right(speed=SPEED_TURN):
           _drive_motor_a(-1, speed)
           _drive_motor_b(-1, speed)

       def stop(brake=False):
           try:
               if pwmA is None or pwmB is None:
                  GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
                  return
           except Exception: pass

           if brake:
               GPIO.output([IN1, IN2, IN3, IN4], GPIO.HIGH)
           else:
               GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
           pwmA.ChangeDutyCycle(0)
           pwmB.ChangeDutyCycle(0)

       def _safe_shutdown():
           global pwmA, pwmB
           try:
               if pwmA: pwmA.stop()
           except Exception: pass
           try:
               if pwmB: pwmB.stop()
           except Exception: pass
           time.sleep(0.05)
           try:
                  GPIO.cleanup()
           except Exception: pass



       def gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       def edges(g): return cv2.Canny(g, 50, 150)

       def ssim_score(a,b):
           if a.shape != b.shape: b = cv2.resize(b, (a.shape[1], a.shape[0]))
           s,_ = ssim(a,b,full=True)
           return float((s+1)/2)

       def edge_corr(a,b):
           ae, be = edges(a), edges(b)
           if ae.shape != be.shape: be = cv2.resize(be,(ae.shape[1], ae.shape[0]))
           r = cv2.matchTemplate(ae, be, cv2.TM_CCOEFF_NORMED)
           return float(((r.max() if r.size else 0)+1)/2)

       def orb_score(a,b,n=200):
            return 0.0

       def combined_fast(crop, tpl_gray_small, tpl_edge_small, tpl_orb_small=None):
           cg = cv2.resize(gray(crop), COMPARE_SZ, interpolation=cv2.INTER_AREA)
           s1 = ssim_score(cg, tpl_gray_small)
           s2 = edge_corr(cg, tpl_edge_small)
           s3 = orb_score(cg, tpl_orb_small) if USE_ORB and tpl_orb_small is not None else 0.0
           combined = W_SSIM*s1 + W_EDGE*s2 + W_ORB*s3
           return combined, (s1, s2, s3)


       try:
           tpl_r = cv2.imread(TPL_R)
           tpl_l = cv2.imread(TPL_L)
           if tpl_r is None or tpl_l is None:
               print(f"WARNING: Template files {TPL_R}/{TPL_L} not found. LEFT/RIGHT detection will use YOLO confidence only.")
               tpl_r = np.zeros((COMPARE_SZ[1], COMPARE_SZ[0], 3), dtype=np.uint8)
               tpl_l = np.zeros((COMPARE_SZ[1], COMPARE_SZ[0], 3), dtype=np.uint8)

           tpl_r_small_gray = cv2.resize(gray(tpl_r), COMPARE_SZ, interpolation=cv2.INTER_AREA)
           tpl_l_small_gray = cv2.resize(gray(tpl_l), COMPARE_SZ, interpolation=cv2.INTER_AREA)
           tpl_r_small_edge = edges(tpl_r_small_gray)
           tpl_l_small_edge = edges(tpl_l_small_gray)

           tpl_r_orb = tpl_l_orb = None
       except Exception as e:
           print(f"Error preparing templates: {e}")
           tpl_r_small_gray = tpl_l_small_gray = np.zeros(COMPARE_SZ, dtype=np.uint8)
           tpl_r_small_edge = tpl_l_small_edge = np.zeros(COMPARE_SZ, dtype=np.uint8)
           tpl_r_orb = tpl_l_orb = None

       def start_action(action):
           global action_in_progress, action_end_time, current_action
           action = action.upper()
           dur = ACTION_DURATIONS.get(action, 0.0)
           print(f"[{time.strftime('%H:%M:%S')}] START ACTION -> {action} (duration {dur:.2f}s)")
           if action == "GO":
               forward(SPEED_FWD)
           elif action == "STOP":
               stop(brake=True)
           elif action == "LEFT":
               left(SPEED_TURN)
           elif action == "RIGHT":
               right(SPEED_TURN)
           else:
               stop(brake=True)

           current_action = action
           if dur > 0:
               action_in_progress = True
               action_end_time = time.time() + dur
           else:
               action_in_progress = False
               action_end_time = 0.0
       def finish_action():
           global action_in_progress, action_end_time, current_action
           print(f"[{time.strftime('%H:%M:%S')}] ACTION FINISHED -> {current_action}")
           if current_action in ("LEFT", "RIGHT"):
                start_action("GO")
           else:
                stop(brake=True)
                current_action = None
           action_in_progress = False
           action_end_time = 0.0


       def parse_args():
           ap = argparse.ArgumentParser()
           ap.add_argument("--model", type=str, default=MODEL) # ใช้ MODEL เป็นค่าเริ่มต้น
           ap.add_argument("--imgsz", type=int, default=480)
           ap.add_argument("--conf", type=float, default=0.2)
           ap.add_argument("--width", type=int, default=640)
           ap.add_argument("--height", type=int, default=480)
           ap.add_argument("--fps", type=int, default=30)
           ap.add_argument("--skip",type=int, default=0)
           ap.add_argument("--device", type=str, default="cpu")
           ap.add_argument("--show", action="store_true", help="Display webcam feed with detection boxes.") 
           return ap.parse_args()

       DEFAULT_NAMES = {0:"stop", 1:"left", 2:"right", 3:"go"}

       def main():
           global action_in_progress, action_end_time, current_action
           args = parse_args()
           model = YOLO(args.model)
           device = args.device
           if device != "cpu" and not torch.cuda.is_available():
                device = "cpu"
           model.to(device)
           class_names = model.names if model.names is not None else DEFAULT_NAMES

           CAMERA_INDEX = 0
           cap = cv2.VideoCapture(CAMERA_INDEX)
           cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
           cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
           cap.set(cv2.CAP_PROP_FPS, args.fps)
           if not cap.isOpened():
              print(f"Error: Could not open webcam at index {CAMERA_INDEX}")
              _safe_shutdown(); exit()
           time.sleep(0.2)
           _ = model(np.zeros((args.height, args.width, 3), dtype=np.uint8), imgsz=args.imgsz, conf=args.conf, verbose=False)
          
           start_action("GO")
           print("Initial command: GO")

           frame_i = 0

           try:
              while True:
               ret, bgr = cap.read()
               if not ret:
                       time.sleep(0.01)
                       continue
               now = time.time()

               if action_in_progress:
                if now >= action_end_time:
                          finish_action()
               else:
                          if args.show:
                                 cv2.putText(bgr, f"CMD: {current_action} (HOLD {action_end_time - now:.2f}s)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                 cv2.imshow('YOLO Traffic Sign Control', bgr)
                                 if cv2.waitKey(1) & 0xFF == ord('q'): break
                          continue 
               do_infer = (args.skip == 0) or (frame_i % (args.skip + 1) == 0)
               chosen_action = current_action # เริ่มต้นด้วย action ที่กำลังทำอยู่
               if do_infer:
                       results = model(bgr, imgsz=args.imgsz, conf=args.conf, verbose=False)
                       detection_status = "No object detected."
                       if len(results[0].boxes) > 0:
                           conf_scores = results[0].boxes.conf
                           if isinstance(conf_scores, torch.Tensor):
                               best_index = torch.argmax(conf_scores).item() 
                       else:
                               best_index = np.argmax(conf_scores)

                       best_box = results[0].boxes[best_index]
                       cls_id = int(best_box.cls.item())
                       conf = float(best_box.conf.item())
                       action_name_yolo = class_names.get(cls_id, "go")

                       if action_name_yolo in ("left", "right"):

                          x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist()) 

                          crop = bgr[y1:y2, x1:x2].copy()
                          if crop.size > 0:
                              comb_r, _ = combined_fast(crop, tpl_r_small_gray, tpl_r_small_edge, tpl_r_orb)
                              comb_l, _ = combined_fast(crop, tpl_l_small_gray, tpl_l_small_edge, tpl_l_orb)
                              chosen_action = "right" if comb_r > comb_l else "left"
                              detection_status = f"ID {cls_id} (YOLO:{action_name_yolo}) Refined by TM: {chosen_action} (Conf: {conf:.2f} R:{comb_r:.2f} L:{comb_l:.2f})"
                          else:
                              chosen_action = action_name_yolo
                              detection_status = f"ID {cls_id} ({chosen_action}) detected (Conf: {conf:.2f})."
                        else:
                          chosen_action = action_name_yolo
                          detection_status = f"ID {cls_id} ({chosen_action}) detected (Conf: {conf:.2f})."


              if chosen_action.upper() != current_action:
                      start_action(chosen_action)
                      print(f"[{frame_i}] AI: {detection_status} -> CMD: {chosen_action.upper()} (EXECUTE)")
              else:
                      if chosen_action.upper() == "GO":
                              forward(SPEED_FWD)
                      print(f"[{frame_i}] AI: {detection_status} -> CMD: {chosen_action.upper()} (CONTINUE)")

                  if args.show:
                      display_frame = results[0].plot(img=bgr.copy(), line_width=2, font_size=0.7, labels=True, conf=True)
                  else:
                      display_frame = bgr.copy()

              if args.show:
                  if 'display_frame' not in locals():
                      display_frame = bgr.copy()

                  cv2.putText(display_frame, f"CMD: {current_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                  cv2.imshow('YOLO Traffic Sign Control', display_frame)
                
                  if cv2.waitKey(1) & 0xFF == ord('q'):
                      break
            
                  frame_i += 1

          except KeyboardInterrupt:
              pass
          finally:
              stop(brake=True)
              cap.release()
              if args.show: cv2.destroyAllWindows()
              print("\nProgram terminated. Safely shutting down.")

       if _name_ == "_main_":
           setup()
           try:
               main()
           finally:
             _safe_shutdown() 
    
            

     

          
    
    


 
    

   


    
   





        
        
    
   

    







 




            

     

          
    
    


 
    

   


    
   





        
        
    
   

    







 

    







 




