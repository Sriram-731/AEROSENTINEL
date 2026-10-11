import io
import os
import sys
import unittest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detector import load_yolo_model, run_inference, highlight_specific_object, image_to_png_bytes
from streamlit.testing.v1 import AppTest


class TestTask5DetectionDashboard(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = load_yolo_model("yolov8n.pt")
        cls.sample_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "sample_drone_recon.jpg"
        )
        cls.sample_img = Image.open(cls.sample_path)

    def test_01_highlight_specific_object(self):
        # Run inference to get real box
        _, dets, _ = run_inference(self.model, self.sample_img, conf_threshold=0.40)
        self.assertGreater(len(dets), 0)
        first_det = dets[0]

        highlighted = highlight_specific_object(
            self.sample_img,
            first_det["box"],
            first_det["class"],
            first_det["confidence_pct"],
        )
        self.assertIsNotNone(highlighted)
        self.assertEqual(highlighted.size, self.sample_img.size)
        print("PASS: Test 1 - Target object highlighting executed with matching dimensions.")

    def test_02_image_to_png_bytes(self):
        png_bytes = image_to_png_bytes(self.sample_img)
        self.assertIsInstance(png_bytes, bytes)
        self.assertGreater(len(png_bytes), 0)
        # Verify decoding the generated PNG bytes produces valid image
        decoded_img = Image.open(io.BytesIO(png_bytes))
        self.assertEqual(decoded_img.format, "PNG")
        self.assertEqual(decoded_img.size, self.sample_img.size)
        print("PASS: Test 2 - In-memory PNG byte serialization and validation verified.")

    def test_03_apptest_full_dashboard_workflow(self):
        at = AppTest.from_file("../app.py").run()
        self.assertEqual(len(at.exception), 0)

        # Switch to Detection Workspace
        at.button[1].click().run()
        self.assertEqual(at.session_state["active_page"], "Detection Workspace")

        # Upload benchmark sample image
        with open(self.sample_path, "rb") as f:
            sample_bytes = f.read()

        at.file_uploader[0].upload("sample_drone_recon.jpg", sample_bytes, "image/jpeg").run()
        self.assertEqual(len(at.exception), 0)

        # Click Run Detection
        self.assertIn("Run Detection", at.button[4].label)
        at.button[4].click().run()
        self.assertEqual(len(at.exception), 0)

        # Check session state
        self.assertIn("detection_result", at.session_state)
        det_res = at.session_state["detection_result"]
        self.assertIsNotNone(det_res)
        self.assertGreater(len(det_res["detections"]), 0)

        # Verify Download button is present
        download_btns = [b for b in at.download_button if "Download Annotated Image" in b.label]
        self.assertEqual(len(download_btns), 1)

        # Verify Detection History recorded exactly 1 entry
        self.assertIn("detection_history", at.session_state)
        history = at.session_state["detection_history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["Filename"], "sample_drone_recon.jpg")
        self.assertEqual(history[0]["Objects Detected"], len(det_res["detections"]))

        # Verify an ordinary rerun does not duplicate history
        at.run()
        self.assertEqual(len(at.session_state["detection_history"]), 1)

        print("PASS: Test 3 - Streamlit dashboard with download button & deduplicated history verified.")

    def test_04_target_selection_and_class_filtering(self):
        at = AppTest.from_file("../app.py").run()
        at.button[1].click().run()

        with open(self.sample_path, "rb") as f:
            sample_bytes = f.read()

        at.file_uploader[0].upload("sample_drone_recon.jpg", sample_bytes, "image/jpeg").run()
        at.button[4].click().run()

        # Check selectbox for target highlight is available
        self.assertGreater(len(at.selectbox), 0)
        # Select Target #1
        if len(at.selectbox[0].options) > 1:
            at.selectbox[0].select(at.selectbox[0].options[1]).run()
            self.assertEqual(len(at.exception), 0)

        # Check multiselect class filter
        self.assertGreater(len(at.multiselect), 0)
        # Select first available class only
        available_classes = at.multiselect[0].options
        if len(available_classes) > 0:
            at.multiselect[0].select([available_classes[0]]).run()
            self.assertEqual(len(at.exception), 0)

        print("PASS: Test 4 - Target selection and multiselect class filtering executed smoothly.")

    def test_05_clear_history_action(self):
        at = AppTest.from_file("../app.py").run()
        at.button[1].click().run()

        with open(self.sample_path, "rb") as f:
            sample_bytes = f.read()

        at.file_uploader[0].upload("sample_drone_recon.jpg", sample_bytes, "image/jpeg").run()
        at.button[4].click().run()

        self.assertEqual(len(at.session_state["detection_history"]), 1)

        # Find Clear Detection History button
        clear_btns = [b for b in at.button if "Clear Detection History" in b.label]
        self.assertEqual(len(clear_btns), 1)
        clear_btns[0].click().run()

        # Verify history is now empty
        self.assertEqual(len(at.session_state["detection_history"]), 0)
        self.assertEqual(len(at.exception), 0)

        print("PASS: Test 5 - Clear detection history action verified.")


if __name__ == "__main__":
    unittest.main()
