"""
Quick Start Example - Fusion Detection System
=============================================

This script demonstrates how to use the fusion detection system.
Run this to process a video with both pothole and road sign detection.
"""

import subprocess
import sys
from pathlib import Path

def run_fusion_detection():
    """Run the fusion detection on a video"""
    
    print("="*80)
    print("FUSION DETECTION QUICK START")
    print("="*80)
    
    # Check if models exist
    pothole_model = Path("pothole_weights/best.pt")
    sign_model = Path("sign_weights/best.pt")
    
    if not pothole_model.exists():
        print(f"❌ Error: Pothole model not found at {pothole_model}")
        print("   Please ensure the model weights are in the correct directory.")
        return
    
    if not sign_model.exists():
        print(f"❌ Error: Sign model not found at {sign_model}")
        print("   Please ensure the model weights are in the correct directory.")
        return
    
    print("✅ Models found!")
    print(f"   Pothole Model: {pothole_model}")
    print(f"   Sign Model: {sign_model}")
    
    # Get video path from user
    print("\n" + "="*80)
    video_path = input("Enter the path to your video file: ").strip().strip('"')
    
    if not video_path:
        print("❌ No video path provided. Exiting.")
        return
    
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"❌ Error: Video not found at {video_path}")
        return
    
    print(f"✅ Video found: {video_file}")
    
    # Get output path
    output_default = f"fusion_output_{video_file.stem}.mp4"
    output_path = input(f"Enter output path (press Enter for '{output_default}'): ").strip().strip('"')
    
    if not output_path:
        output_path = output_default
    
    print(f"📹 Output will be saved to: {output_path}")
    
    # Ask about display
    show_display = input("\nShow live video display? (y/n, default=y): ").strip().lower()
    no_display_flag = [] if show_display != 'n' else ["--no-display"]
    
    # Build command
    cmd = [
        sys.executable,  # Python executable
        "fusion_detection.py",
        "--video", str(video_path),
        "--output", output_path,
        "--pothole-model", str(pothole_model),
        "--sign-model", str(sign_model)
    ] + no_display_flag
    
    print("\n" + "="*80)
    print("STARTING FUSION DETECTION...")
    print("="*80)
    print(f"\nCommand: {' '.join(cmd)}")
    print("\nPress Ctrl+C to stop processing")
    print("Press 'q' in the video window to quit")
    print("\n" + "="*80 + "\n")
    
    try:
        # Run the detection
        subprocess.run(cmd, check=True)
        
        print("\n" + "="*80)
        print("✅ DETECTION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nOutput saved to: {output_path}")
        print("\nYou can now:")
        print("1. Open the output video to view results")
        print("2. Run detection on another video")
        print("3. Check the console for detailed statistics")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during detection: {e}")
    except KeyboardInterrupt:
        print("\n⚠️  Detection interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

def main():
    """Main function"""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║          FUSION DETECTION - QUICK START SCRIPT                ║
    ║                                                                ║
    ║  Combines Pothole Detection + Road Sign Detection             ║
    ║  - Dual model inference                                        ║
    ║  - Distance estimation for potholes                            ║
    ║  - 21 road sign classes + generic fallback                     ║
    ║  - Adaptive frame skipping for speed                           ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    while True:
        run_fusion_detection()
        
        print("\n" + "="*80)
        again = input("\nProcess another video? (y/n): ").strip().lower()
        if again != 'y':
            print("\n✅ Thank you for using Fusion Detection!")
            print("="*80)
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Exiting...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
