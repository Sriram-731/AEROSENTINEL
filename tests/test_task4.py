import io
import os
import sys
import unittest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detector import load_yolo_model, run_inference
from streamlit.testing.v1 import AppTest


class TestTask4RealYOLOv8Integration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = load_yolo_model("yolov8n.pt")
        cls.sample_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "sample_drone_recon.jpg"
        )
        cls.sample_img = Image.open(cls.sample_path)

    def test_01_model_instance(self):
        self.assertIsNotNone(self.model)
        self.assertTrue(hasattr(self.model, "predict"))
        print("PASS: Test 1 - Real YOLOv8 model instance validated.")

    def test_02_real_inference_on_sample_image(self):
        annotated_img, detections, duration_ms = run_inference(
            self.model, self.sample_img, conf_threshold=0.40
        )

        # Verify duration
        self.assertGreater(duration_ms, 0)

        # Verify detections
        self.assertGreater(len(detections), 0)
        for det in detections:
            self.assertIn("class", det)
            self.assertIn("confidence", det)
            self.assertIn("confidence_pct", det)
            self.assertIn("box", det)
            self.assertGreaterEqual(det["confidence"], 0.40)
            self.assertEqual(len(det["box"]), 4)

        # Verify annotated image
        self.assertEqual(annotated_img.size, self.sample_img.size)
        print(
            f"PASS: Test 2 - Real inference executed ({duration_ms:.1f}ms, {len(detections)} objects detected: {[d['class'] for d in detections]})."
        )

    def test_03_zero_detections_handling(self):
        # Setting confidence to 0.99 should filter out all objects
        annotated_img, detections, duration_ms = run_inference(
            self.model, self.sample_img, conf_threshold=0.99
        )
        self.assertGreater(duration_ms, 0)
        self.assertIsInstance(detections, list)
        self.assertEqual(annotated_img.size, self.sample_img.size)
        print(f"PASS: Test 3 - High threshold / zero detections handled gracefully ({len(detections)} detected).")

    def test_04_streamlit_app_real_inference(self):
        at = AppTest.from_file("../app.py").run()
        self.assertEqual(len(at.exception), 0)

        # Switch to Detection Workspace
        at.button[1].click().run()
        self.assertEqual(at.session_state["active_page"], "Detection Workspace")
        self.assertEqual(len(at.exception), 0)

        # Upload benchmark sample image
        with open(self.sample_path, "rb") as f:
            sample_bytes = f.read()

        at.file_uploader[0].upload("sample_drone_recon.jpg", sample_bytes, "image/jpeg").run()
        self.assertEqual(len(at.exception), 0)

        # Click 'Run Detection' button (button 4)
        self.assertIn("Run Detection", at.button[4].label)
        at.button[4].click().run()
        self.assertEqual(len(at.exception), 0)

        # Verify detection results are stored in session state
        self.assertIn("detection_result", at.session_state)
        det_res = at.session_state["detection_result"]
        self.assertIsNotNone(det_res)
        self.assertGreater(len(det_res["detections"]), 0)
        self.assertGreater(det_res["duration_ms"], 0)

        print(
            f"PASS: Test 4 - Streamlit AppTest executed real inference successfully ({det_res['duration_ms']:.1f}ms, {len(det_res['detections'])} objects) with 0 exceptions!"
        )


if __name__ == "__main__":
    unittest.main()
