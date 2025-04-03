import cv2
import time
import streamlit as st
import numpy as np
import pandas as pd
import os
from datetime import datetime
from detector import ObjectDetector
from inventory import InventoryTracker
from config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, PRODUCTS, REPORT_FREQUENCY, VIDEO_PATH

# Setup page config
st.set_page_config(
    page_title="Retail Inventory Tracking",
    page_icon="📦",
    layout="wide"
)

# Initialize detector and tracker
@st.cache_resource
def get_detector():
    return ObjectDetector()

@st.cache_resource
def get_tracker():
    return InventoryTracker()

detector = get_detector()
tracker = get_tracker()

# Global variables
last_report_time = datetime.now()

# UI Components
def main():
    # Title
    st.title("📦 Retail Inventory Tracking System")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Input Source")
        
        # Choose input source
        input_source = st.radio(
            "Select input source:",
            ["Webcam", "Sample Video", "Upload Video"],
            index=0
        )
        
        # Handle different input sources
        if input_source == "Webcam":
            source_type = "webcam"
            camera_id = st.selectbox("Select camera", [0, 1, 2, 3], index=0)
            video_source = camera_id
            loop_video = False
        
        elif input_source == "Sample Video":
            source_type = "video"
            video_path = VIDEO_PATH
            if not os.path.exists(video_path):
                st.warning(f"Sample video not found: {video_path}. Please run download_sample_video.py first.")
                video_source = None
            else:
                video_source = video_path
                loop_video = st.checkbox("Loop video", value=True)
        
        elif input_source == "Upload Video":
            source_type = "video"
            uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file is not None:
                # Create temp directory if it doesn't exist
                save_dir = os.path.join(os.getcwd(), "temp_videos")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                # Save the file
                video_path = os.path.join(save_dir, uploaded_file.name)
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                video_source = video_path
                st.success(f"Video uploaded successfully: {uploaded_file.name}")
                loop_video = st.checkbox("Loop video", value=True)
            else:
                video_source = None
                loop_video = False
        
        # Manual report generation
        st.divider()
        if st.button("Generate Inventory Report"):
            report_path = tracker.save_report()
            if report_path:
                st.success(f"Report saved to {report_path}")
            else:
                st.warning("No data available to generate report")
        
        st.divider()
        
        # Display tracked products
        st.subheader("Tracked Products")
        product_df = pd.DataFrame([
            {
                "Product": PRODUCTS[p]["name"],
                "Low Stock Threshold": PRODUCTS[p]["threshold"]
            }
            for p in PRODUCTS if PRODUCTS[p]["alert"]
        ])
        st.dataframe(product_df, hide_index=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display feed
        if source_type == "webcam":
            st.subheader("Webcam Feed")
        else:
            st.subheader("Video Feed")
            
        feed_placeholder = st.empty()
        
        # Current frame
        try:
            # Only process video frames if a valid source is selected
            if video_source is not None and st.button("Start Processing", use_container_width=True):
                # Create inventory and alerts placeholders in column 2
                with col2:
                    inventory_container = st.container()
                    with inventory_container:
                        st.subheader("Current Inventory")
                        inventory_placeholder = st.empty()
                        
                    alerts_container = st.container()
                    with alerts_container:
                        st.subheader("Low Stock Alerts")
                        alerts_placeholder = st.empty()
                
                process_video_feed(
                    video_source, 
                    feed_placeholder, 
                    source_type, 
                    loop_video, 
                    inventory_placeholder,
                    alerts_placeholder
                )
            else:
                feed_placeholder.info("Press 'Start Processing' to begin")
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
    
    with col2:
        # Placeholders for inventory and alerts that will be updated
        st.subheader("Current Inventory")
        st.info("Inventory data will appear when detection starts")
        
        st.subheader("Low Stock Alerts")
        st.info("No alerts at the moment")

def process_video_feed(source, feed_placeholder, source_type, loop_video=False, 
                       inventory_placeholder=None, alerts_placeholder=None):
    """Process video feed without using threads"""
    global last_report_time
    
    # Initialize video capture
    if source_type == "webcam":
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        delay = 0.03  # ~30 FPS for webcam
    else:
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps > 0 else 0.03
    
    # Check if video opened successfully
    if not cap.isOpened():
        st.error(f"Error: Could not open video source {source}")
        return
    
    # Create progress bar
    progress_bar = st.progress(0)
    stop_button = st.button("Stop Processing")
    
    # Get total frames for video (not applicable for webcam)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source_type == "video" else 0
    frame_count = 0
    
    # Process frames
    while not stop_button:
        # Read frame
        ret, frame = cap.read()
        
        # If frame reading failed
        if not ret:
            if source_type == "video" and loop_video:
                # Reset video to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                continue
            else:
                break
                
        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        processed_frame, detections = detector.detect(frame_rgb)
        
        # Update inventory
        tracker.update(detections)
        
        # Display the processed frame
        feed_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
        
        # Update progress for video files
        if source_type == "video" and total_frames > 0:
            frame_count += 1
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
        
        # Update inventory display
        update_inventory_display(inventory_placeholder, alerts_placeholder)
        
        # Check if we need to save a report
        now = datetime.now()
        if REPORT_FREQUENCY == "hourly" and now.hour != last_report_time.hour:
            tracker.save_report()
            last_report_time = now
        elif REPORT_FREQUENCY == "daily" and now.day != last_report_time.day:
            tracker.save_report()
            last_report_time = now
            
        # Add a small delay to control frame rate and not overwhelm the CPU
        time.sleep(delay)
        
        # Check if stop button was clicked
        if stop_button:
            break
    
    # Release resources
    cap.release()
    st.info("Processing stopped")
    progress_bar.empty()

def update_inventory_display(inventory_placeholder, alerts_placeholder):
    """Update the inventory and alerts display"""
    # Get inventory data
    if tracker.inventory and inventory_placeholder is not None:
        inventory_data = []
        for product, count in tracker.get_inventory().items():
            product_key = next((p for p in PRODUCTS if PRODUCTS[p]["name"] == product), None)
            if product_key and PRODUCTS[product_key]["alert"]:
                threshold = PRODUCTS[product_key]["threshold"]
                status = "🔴 Low" if count < threshold else "🟢 OK"
                inventory_data.append({"Product": product, "Count": count, "Status": status})
            else:
                inventory_data.append({"Product": product, "Count": count, "Status": "⚪ N/A"})
                
        inventory_df = pd.DataFrame(inventory_data)
        inventory_placeholder.dataframe(inventory_df, hide_index=True, use_container_width=True)
    
    # Get alerts data
    if alerts_placeholder is not None:
        if tracker.alerts:
            alert_data = []
            for alert in tracker.alerts:
                alert_data.append({
                    "Product": alert["product"],
                    "Current Count": alert["count"],
                    "Threshold": alert["threshold"],
                    "Time": alert["timestamp"].strftime("%H:%M:%S")
                })
            
            alert_df = pd.DataFrame(alert_data)
            alerts_placeholder.dataframe(alert_df, hide_index=True, use_container_width=True)
        else:
            alerts_placeholder.info("No alerts at the moment")

if __name__ == "__main__":
    main() 