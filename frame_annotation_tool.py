#!/usr/bin/env python3
import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import sys


class FrameAnnotationTool:
    def __init__(self, video_folder_path: str):
        """
        Initialize the annotation tool
        
        Args:
            video_folder_path: Path to video_xxx folder containing timestamp folders
        """
        self.video_folder = Path(video_folder_path)
        self.frames = []
        self.current_frame_idx = 0
        self.current_person_id = None
        
        # Polygon drawing state
        self.polygon_points = []
        self.is_drawing = False
        
        # Pen thickness settings
        self.pen_thickness = 1  # Start with thin
        self.pen_thickness_options = [1, 2, 3, 4, 5, 8, 10]  # Very thin to thick
        self.pen_thickness_idx = 0
        
        # Zoom state
        self.zoom_level = 1.0
        self.zoom_center = None  # (x, y) in original image coordinates
        self.is_zoomed = False
        self.roi_rect = None  # (x, y, w, h) for the region of interest
        
        # Display settings
        self.window_name = "Frame Annotation Tool"
        self.display_img = None
        self.original_img = None
        self.display_scale = 1.0
        
        # Colors
        self.DRAWING_COLOR = (255, 255, 255)  # White
        self.POINT_COLOR = (0, 255, 0)  # Green for vertices
        
        # Load frames
        self._load_frames()
        
    def _load_frames(self):
        """Load all frame paths from the video folder in sorted order"""
        timestamp_folders = sorted([d for d in self.video_folder.iterdir() 
                                   if d.is_dir() and not d.name.startswith('annotations')])
        
        if not timestamp_folders:
            raise ValueError(f"No timestamp folders found in {self.video_folder}")
        
        for ts_folder in timestamp_folders:
            # Look for .jpg files (frames are named like 1119716761570_00.jpg)
            frame_files = sorted([f for f in ts_folder.glob("*.jpg")])
            for frame_file in frame_files:
                self.frames.append(frame_file)
        
        if not self.frames:
            raise ValueError(f"No frames found in {self.video_folder}")
        
        print(f"✓ Loaded {len(self.frames)} frames from {len(timestamp_folders)} timestamp folders")
    
    def _get_annotation_paths(self, frame_path: Path, person_id: str) -> Tuple[Path, Path]:
        """Get the contour and mask output paths for a given frame and person"""
        # Extract the timestamp from the filename (e.g., 1119716761570 from 1119716761570_00.jpg)
        frame_timestamp = frame_path.stem.split('_')[0]
        
        # Create annotations/person_id structure
        annotation_dir = self.video_folder / "annotations" / person_id
        annotation_dir.mkdir(parents=True, exist_ok=True)
        
        contour_path = annotation_dir / f"{person_id}_{frame_timestamp}_contour.jpg"
        mask_path = annotation_dir / f"{person_id}_{frame_timestamp}_mask.jpg"
        
        return contour_path, mask_path
    
    def _is_frame_annotated(self, frame_path: Path, person_id: str) -> bool:
        """Check if a frame is already annotated for a person"""
        try:
            contour_path, mask_path = self._get_annotation_paths(frame_path, person_id)
            return contour_path.exists() and mask_path.exists()
        except Exception as e:
            print(f"Warning: Could not check annotation status: {e}")
            return False
    
    def _load_and_prepare_frame(self):
        """Load current frame and prepare for display"""
        try:
            frame_path = self.frames[self.current_frame_idx]
            self.original_img = cv2.imread(str(frame_path))
            
            if self.original_img is None:
                raise ValueError(f"Could not load frame: {frame_path}")
            
            # Reset zoom when loading new frame
            self.is_zoomed = False
            self.zoom_level = 1.0
            self.zoom_center = None
            self.roi_rect = None
            
            # Calculate display scale to fit screen (lightweight scaling)
            h, w = self.original_img.shape[:2]
            screen_h, screen_w = 1080, 1920  # Assume standard screen
            
            scale_w = screen_w / w
            scale_h = screen_h / h
            self.display_scale = min(scale_w, scale_h, 1.0)  # Don't upscale
            
            self._update_display()
        except Exception as e:
            print(f"ERROR loading frame: {e}")
            raise
    
    def _update_display(self):
        """Update the display image based on current zoom and ROI"""
        try:
            if self.is_zoomed and self.roi_rect is not None:
                # Extract and display ROI
                x, y, w, h = self.roi_rect
                roi = self.original_img[y:y+h, x:x+w].copy()
                
                # Scale ROI to screen
                display_h, display_w = roi.shape[:2]
                scale = min(1920 / display_w, 1080 / display_h, 1.0)
                
                if scale < 1.0:
                    new_w = int(display_w * scale)
                    new_h = int(display_h * scale)
                    self.display_img = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    self.display_img = roi
            else:
                # Normal display with scaling
                h, w = self.original_img.shape[:2]
                if self.display_scale < 1.0:
                    new_w = int(w * self.display_scale)
                    new_h = int(h * self.display_scale)
                    self.display_img = cv2.resize(self.original_img, (new_w, new_h), 
                                                 interpolation=cv2.INTER_AREA)
                else:
                    self.display_img = self.original_img.copy()
        except Exception as e:
            print(f"ERROR updating display: {e}")
    
    def _display_to_original_coords(self, x: int, y: int) -> Tuple[int, int]:
        """Convert display coordinates to original image coordinates"""
        try:
            if self.is_zoomed and self.roi_rect is not None:
                # Account for ROI offset and scaling
                roi_x, roi_y, roi_w, roi_h = self.roi_rect
                display_h, display_w = self.display_img.shape[:2]
                
                scale = roi_w / display_w
                orig_x = int(x * scale + roi_x)
                orig_y = int(y * scale + roi_y)
            else:
                # Account for display scaling
                scale = 1.0 / self.display_scale
                orig_x = int(x * scale)
                orig_y = int(y * scale)
            
            return orig_x, orig_y
        except Exception as e:
            print(f"ERROR converting coords: {e}")
            return x, y
    
    def _original_to_display_coords(self, x: int, y: int) -> Tuple[int, int]:
        """Convert original image coordinates to display coordinates"""
        try:
            if self.is_zoomed and self.roi_rect is not None:
                roi_x, roi_y, roi_w, roi_h = self.roi_rect
                
                # Check if point is within ROI
                if not (roi_x <= x < roi_x + roi_w and roi_y <= y < roi_y + roi_h):
                    return None, None
                
                # Convert to ROI-relative coordinates
                rel_x = x - roi_x
                rel_y = y - roi_y
                
                # Scale to display
                display_h, display_w = self.display_img.shape[:2]
                scale = display_w / roi_w
                disp_x = int(rel_x * scale)
                disp_y = int(rel_y * scale)
            else:
                disp_x = int(x * self.display_scale)
                disp_y = int(y * self.display_scale)
            
            return disp_x, disp_y
        except Exception as e:
            print(f"ERROR converting coords: {e}")
            return None, None
    
    def _draw_polygon_overlay(self):
        """Draw current polygon on display image"""
        try:
            if not self.polygon_points:
                return self.display_img.copy()
            
            overlay = self.display_img.copy()
            
            # Convert points to display coordinates
            display_points = []
            for px, py in self.polygon_points:
                dx, dy = self._original_to_display_coords(px, py)
                if dx is not None and dy is not None:
                    display_points.append((dx, dy))
            
            if not display_points:
                return overlay
            
            # Calculate display thickness based on zoom level
            display_thickness = max(1, int(self.pen_thickness / self.zoom_level)) if self.is_zoomed else self.pen_thickness
            
            # Draw lines between points
            for i in range(len(display_points)):
                pt1 = display_points[i]
                pt2 = display_points[(i + 1) % len(display_points)] if i < len(display_points) - 1 else display_points[0]
                
                if i < len(display_points) - 1:
                    cv2.line(overlay, pt1, pt2, self.DRAWING_COLOR, display_thickness, cv2.LINE_AA)
            
            # Draw vertices (always visible regardless of thickness)
            vertex_radius = max(2, display_thickness + 1)
            for pt in display_points:
                cv2.circle(overlay, pt, vertex_radius, self.POINT_COLOR, -1, cv2.LINE_AA)
            
            return overlay
        except Exception as e:
            print(f"ERROR drawing polygon: {e}")
            return self.display_img.copy()
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for polygon drawing and zooming"""
        try:
            # Zoom with mouse wheel
            if event == cv2.EVENT_MOUSEWHEEL:
                # Convert display coords to original coords for zoom center
                orig_x, orig_y = self._display_to_original_coords(x, y)
                
                if flags > 0:  # Scroll up - zoom in
                    if not self.is_zoomed:
                        self.is_zoomed = True
                        self.zoom_center = (orig_x, orig_y)
                        self.zoom_level = 1.5
                        self._update_zoom_roi()
                    else:
                        # Already zoomed, zoom in more
                        self.zoom_level = min(self.zoom_level * 1.3, 8.0)
                        self._update_zoom_roi()
                else:  # Scroll down - zoom out
                    if self.is_zoomed:
                        self.zoom_level = max(self.zoom_level / 1.3, 1.0)
                        if self.zoom_level <= 1.05:
                            self.is_zoomed = False
                            self.roi_rect = None
                            self.zoom_level = 1.0
                        else:
                            self._update_zoom_roi()
                
                self._update_display()
                return
            
            # Polygon drawing
            if event == cv2.EVENT_LBUTTONDOWN:
                # Convert to original image coordinates
                orig_x, orig_y = self._display_to_original_coords(x, y)
                
                # Clamp to image bounds
                h, w = self.original_img.shape[:2]
                orig_x = max(0, min(orig_x, w - 1))
                orig_y = max(0, min(orig_y, h - 1))
                
                self.polygon_points.append((orig_x, orig_y))
                self.is_drawing = True
        except Exception as e:
            print(f"ERROR in mouse callback: {e}")
    
    def _update_zoom_roi(self):
        """Update the ROI rectangle based on zoom level and center"""
        try:
            if not self.is_zoomed or self.zoom_center is None:
                return
            
            h, w = self.original_img.shape[:2]
            cx, cy = self.zoom_center
            
            # Calculate ROI size based on zoom level
            roi_w = int(w / self.zoom_level)
            roi_h = int(h / self.zoom_level)
            
            # Calculate ROI position (centered on zoom_center)
            roi_x = cx - roi_w // 2
            roi_y = cy - roi_h // 2
            
            # Clamp ROI to image bounds
            roi_x = max(0, min(roi_x, w - roi_w))
            roi_y = max(0, min(roi_y, h - roi_h))
            
            self.roi_rect = (roi_x, roi_y, roi_w, roi_h)
        except Exception as e:
            print(f"ERROR updating zoom ROI: {e}")
    
    def _save_annotation(self):
        """Save the current polygon annotation"""
        try:
            if len(self.polygon_points) < 3:
                print("⚠ Polygon must have at least 3 points. Not saving.")
                return False
            
            frame_path = self.frames[self.current_frame_idx]
            contour_path, mask_path = self._get_annotation_paths(frame_path, self.current_person_id)
            
            print(f"DEBUG: Saving to {contour_path}")
            print(f"DEBUG: Saving to {mask_path}")
            
            # Create contour image (original with white polygon)
            contour_img = self.original_img.copy()
            pts = np.array(self.polygon_points, dtype=np.int32)
            cv2.polylines(contour_img, [pts], isClosed=True, color=self.DRAWING_COLOR, thickness=2)
            
            # Create mask image (black background, white filled polygon)
            h, w = self.original_img.shape[:2]
            mask_img = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.fillPoly(mask_img, [pts], self.DRAWING_COLOR)
            
            # Save both images
            success1 = cv2.imwrite(str(contour_path), contour_img)
            success2 = cv2.imwrite(str(mask_path), mask_img)
            
            if success1 and success2:
                print(f"✓ Saved: {contour_path.name} and {mask_path.name}")
                return True
            else:
                print(f"ERROR: Failed to save images. Success: contour={success1}, mask={success2}")
                return False
                
        except Exception as e:
            print(f"ERROR saving annotation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _cycle_pen_thickness(self):
        """Cycle through pen thickness options"""
        self.pen_thickness_idx = (self.pen_thickness_idx + 1) % len(self.pen_thickness_options)
        self.pen_thickness = self.pen_thickness_options[self.pen_thickness_idx]
        print(f"✏ Pen thickness: {self.pen_thickness}px")
    
    def _show_help(self):
        """Display help overlay"""
        help_text = [
            "=== FRAME ANNOTATION TOOL ===",
            "",
            "NAVIGATION:",
            "  n - Next frame",
            "  b - Previous frame",
            "  q - Quit",
            "",
            "DRAWING:",
            "  Left Click - Add polygon point",
            "  t - Change pen thickness (1-10px)",
            "  p - Reset/clear polygon",
            "  s - Save annotation (auto-advances)",
            "",
            "ZOOM:",
            "  Mouse Wheel Up - Zoom in",
            "  Mouse Wheel Down - Zoom out",
            "  Zoom centered on cursor",
            "",
            "TIPS:",
            "  - Polygon auto-closes on save",
            "  - Use thin pen when zoomed in",
            "  - Already annotated frames show checkmark",
            "",
            "Press any key to continue..."
        ]
        
        # Create help overlay
        overlay = np.zeros((700, 550, 3), dtype=np.uint8)
        y_offset = 30
        for line in help_text:
            cv2.putText(overlay, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 25
        
        cv2.imshow("Help", overlay)
        cv2.waitKey(0)
        cv2.destroyWindow("Help")
    
    def _get_status_text(self) -> List[str]:
        """Generate status text for display"""
        try:
            frame_path = self.frames[self.current_frame_idx]
            annotated = "✓" if self._is_frame_annotated(frame_path, self.current_person_id) else "✗"
            zoom_text = f"{self.zoom_level:.1f}x" if self.is_zoomed else "1.0x"
            
            status_lines = [
                f"Frame: {self.current_frame_idx + 1}/{len(self.frames)} | Person: {self.current_person_id} | Points: {len(self.polygon_points)}",
                f"Zoom: {zoom_text} | Pen: {self.pen_thickness}px | Annotated: {annotated} | [h]=help [t]=thickness"
            ]
            
            return status_lines
        except Exception as e:
            return [f"ERROR: {e}"]
    
    def run(self):
        """Main annotation loop"""
        # Get person ID at start
        print("\n=== Frame Annotation Tool ===")
        print(f"Video folder: {self.video_folder}")
        print(f"Total frames: {len(self.frames)}")
        
        self.current_person_id = input("\nEnter person ID (e.g., person_01): ").strip()
        
        if not self.current_person_id:
            print("Error: Person ID cannot be empty")
            return
        
        # Show help
        print("\nControls:")
        print("  n/b - Next/Previous frame")
        print("  t - Change pen thickness")
        print("  s - Save")
        print("  p - Reset polygon")
        print("  h - Help")
        print("  q - Quit")
        
        # Load first frame
        self._load_and_prepare_frame()
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\nStarting annotation loop...")
        
        running = True
        frame_count = 0
        while running:
            try:
                # Draw current state
                display = self._draw_polygon_overlay()
                
                # Add status text
                status_lines = self._get_status_text()
                y_pos = 25
                for line in status_lines:
                    cv2.putText(display, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    y_pos += 20
                
                cv2.imshow(self.window_name, display)
                
                # Handle keyboard input
                key = cv2.waitKey(20) & 0xFF
                
                if key == 255:  # No key pressed
                    continue
                
                # Debug key press
                if key != 255:
                    print(f"DEBUG: Key pressed: {key} ('{chr(key) if 32 <= key < 127 else '?'}')")
                
                if key == ord('q'):
                    print("Quitting...")
                    running = False
                
                elif key == ord('h'):
                    self._show_help()
                
                elif key == ord('t'):
                    self._cycle_pen_thickness()
                
                elif key == ord('n'):
                    if self.current_frame_idx < len(self.frames) - 1:
                        self.current_frame_idx += 1
                        self._load_and_prepare_frame()
                        self.polygon_points = []
                        print(f"→ Next: Frame {self.current_frame_idx + 1}/{len(self.frames)}")
                    else:
                        print("⚠ Already at last frame")
                
                elif key == ord('b'):
                    if self.current_frame_idx > 0:
                        self.current_frame_idx -= 1
                        self._load_and_prepare_frame()
                        self.polygon_points = []
                        print(f"← Previous: Frame {self.current_frame_idx + 1}/{len(self.frames)}")
                    else:
                        print("⚠ Already at first frame")
                
                elif key == ord('s'):
                    print("Attempting to save...")
                    if self._save_annotation():
                        # Auto-advance to next frame
                        if self.current_frame_idx < len(self.frames) - 1:
                            self.current_frame_idx += 1
                            self._load_and_prepare_frame()
                            self.polygon_points = []
                            print(f"Saved! Moving to frame {self.current_frame_idx + 1}")
                        else:
                            print("✓ Last frame annotated!")
                            self.polygon_points = []
                    else:
                        print("Save failed!")
                
                elif key == ord('p'):
                    self.polygon_points = []
                    print("⟲ Polygon reset")
                
                frame_count += 1
                
            except Exception as e:
                print(f"ERROR in main loop: {e}")
                import traceback
                traceback.print_exc()
        
        print("Closing windows...")
        cv2.destroyAllWindows()
        print("Done!")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        print("=== Frame Annotation Tool ===")
        print("\nPlease enter the path to your video_xxx folder:")
        print("(e.g., C:\\Users\\USER\\Desktop\\Fast_Moving_Object_Database\\video_001)")
        folder_path = input("\nPath: ").strip().strip('"')
    
    if not folder_path:
        print("Error: No folder path provided")
        return
    
    try:
        tool = FrameAnnotationTool(folder_path)
        tool.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
