import cv2
from ultralytics import YOLO
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

def main():
    rclpy.init()
    node = rclpy.create_node('cone_publisher')
    pub = node.create_publisher(Float32MultiArray, 'cone_coordinates', 10)

    USE_WEBCAM = True  
    TARGET_COLOR_NAME = "blue"  # Options: "yellow", "blue", "orange", "large_orange", "unknown"

    # Paths
    test_image_path = '..data_split/test/images/amz_00001.jpg'
    model_path = '/home/rover/detection_ws/model/best.pt'
    conf_threshold = 0.5

    COLOR_TO_ID = {
        "yellow": 0,
        "blue": 1,
        "orange": 2,
        "large_orange": 3,
        "unknown": 4
    }

    if TARGET_COLOR_NAME not in COLOR_TO_ID:
        print(f"Error: '{TARGET_COLOR_NAME}' is not a valid color.")
        return
    TARGET_ID = COLOR_TO_ID[TARGET_COLOR_NAME]

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
    model = YOLO(model_path)

    def get_specific_color_centers(frame, results, target_id):
        detected_centers = []
        annotated_frame = frame.copy()

        if results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id == target_id:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detected_centers.append((cx, cy, float(cls_id)))

                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(annotated_frame, f"{cx},{cy}", (cx, cy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return annotated_frame, detected_centers

    if USE_WEBCAM:
        cap = cv2.VideoCapture(4,cv2.CAP_V4L2)
        if not cap.isOpened(): 
            print("Cannot open webcam")
            return

        print("Press 'q' to quit...")
        while rclpy.ok():
            ret, frame = cap.read()
            if not ret: break

            results = model(frame, conf=conf_threshold, stream=True)

            for r in results:
                final_frame, centers = get_specific_color_centers(frame, [r], TARGET_ID)

                # Publish detected centers to ROS 2
                if centers:
                    msg = Float32MultiArray()
                    data = []
                    for (cx, cy, cls_id) in centers:
                        data.extend([float(cx), float(cy), float(cls_id)])
                    msg.data = data
                    pub.publish(msg)
                    node.get_logger().info(f"Published cones: {data}")

                cv2.imshow(f"Tracking: {TARGET_COLOR_NAME.upper()}", final_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        frame = cv2.imread(test_image_path)
        results = model(frame, conf=conf_threshold)

        final_frame, centers = get_specific_color_centers(frame, results, TARGET_ID)

        if centers:
            msg = Float32MultiArray()
            data = []
            for (cx, cy, cls_id) in centers:
                data.extend([float(cx), float(cy), float(cls_id)])
            msg.data = data
            pub.publish(msg)
            node.get_logger().info(f"Published cones: {data}")

        cv2.imshow("Result", final_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
