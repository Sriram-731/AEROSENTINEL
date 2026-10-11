import io
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
from streamlit.testing.v1 import AppTest
from image_utils import validate_and_load_image, extract_metadata, MAX_UPLOAD_SIZE_BYTES


class TestTask3ImageUploadAndWorkspace(unittest.TestCase):

    def setUp(self):
        # 1. Create a sample JPG in memory
        self.jpg_bytes = io.BytesIO()
        img_jpg = Image.new("RGB", (800, 600), color=(30, 45, 60))
        img_jpg.save(self.jpg_bytes, format="JPEG")
        self.jpg_data = self.jpg_bytes.getvalue()

        # 2. Create a sample PNG in memory (with RGBA)
        self.png_bytes = io.BytesIO()
        img_png = Image.new("RGBA", (1024, 768), color=(0, 229, 255, 180))
        img_png.save(self.png_bytes, format="PNG")
        self.png_data = self.png_bytes.getvalue()

        # 3. Create a sample WEBP in memory
        self.webp_bytes = io.BytesIO()
        img_webp = Image.new("RGB", (640, 480), color=(16, 24, 39))
        img_webp.save(self.webp_bytes, format="WEBP")
        self.webp_data = self.webp_bytes.getvalue()

        # 4. Dummy mock uploaded file helper
        class MockUploadedFile:
            def __init__(self, name, data):
                self.name = name
                self.data = data
                self.size = len(data)

            def getvalue(self):
                return self.data

        self.MockFile = MockUploadedFile

    def test_01_valid_jpg_upload(self):
        f = self.MockFile("drone_recon_01.jpg", self.jpg_data)
        img, meta, err = validate_and_load_image(f)
        self.assertIsNone(err)
        self.assertIsNotNone(img)
        self.assertEqual(meta["width"], 800)
        self.assertEqual(meta["height"], 600)
        self.assertEqual(meta["format"], "JPEG")
        print("PASS: Test 1 - Valid JPG decoded and metadata extracted.")

    def test_02_valid_png_upload(self):
        f = self.MockFile("drone_survey_02.png", self.png_data)
        img, meta, err = validate_and_load_image(f)
        self.assertIsNone(err)
        self.assertIsNotNone(img)
        self.assertEqual(meta["width"], 1024)
        self.assertEqual(meta["height"], 768)
        self.assertEqual(meta["format"], "PNG")
        print("PASS: Test 2 - Valid PNG decoded and metadata extracted.")

    def test_03_valid_webp_upload(self):
        f = self.MockFile("aerial_stream_03.webp", self.webp_data)
        img, meta, err = validate_and_load_image(f)
        self.assertIsNone(err)
        self.assertIsNotNone(img)
        self.assertEqual(meta["width"], 640)
        self.assertEqual(meta["height"], 480)
        self.assertEqual(meta["format"], "WEBP")
        print("PASS: Test 3 - Valid WEBP decoded and metadata extracted.")

    def test_04_unsupported_file_type(self):
        # Text/binary file claiming to be an image
        bad_data = b"%PDF-1.4 dummy pdf document content"
        f = self.MockFile("flight_log.pdf", bad_data)
        img, meta, err = validate_and_load_image(f)
        self.assertIsNone(img)
        self.assertIsNotNone(err)
        self.assertTrue("Unsupported" in err or "unrecognized" in err or "Corrupted" in err)
        print("PASS: Test 4 - Unsupported file type safely rejected.")

    def test_05_corrupted_image(self):
        # Truncated or mangled bytes
        corrupted_data = self.jpg_data[:50] + b"\x00\xff\x00\xaa" * 10
        f = self.MockFile("corrupted_feed.jpg", corrupted_data)
        img, meta, err = validate_and_load_image(f)
        self.assertIsNone(img)
        self.assertIsNotNone(err)
        print(f"PASS: Test 5 - Corrupted image handled gracefully ({err}).")

    def test_06_file_size_limit_exceeded(self):
        # Exceeds 10 MB limit
        huge_file = self.MockFile("huge_panorama.jpg", b"0" * 100)
        huge_file.size = 11 * 1024 * 1024  # 11 MB
        img, meta, err = validate_and_load_image(huge_file)
        self.assertIsNone(img)
        self.assertIn("exceeds maximum upload limit", err)
        print("PASS: Test 6 - File exceeding 10 MB rejected with clear error.")

    def test_07_streamlit_app_navigation_and_upload(self):
        at = AppTest.from_file("../app.py").run()
        self.assertEqual(len(at.exception), 0)

        # Navigate to Detection Workspace
        at.button[1].click().run()
        self.assertEqual(at.session_state["active_page"], "Detection Workspace")
        self.assertEqual(len(at.exception), 0)

        # Upload image into the workspace
        at.file_uploader[0].upload("drone_flight_alpha.jpg", self.jpg_data, "image/jpeg").run()
        self.assertEqual(len(at.exception), 0)

        # Verify Detection Controls and status
        # Button 4 is 'Run Detection'
        self.assertIn("Run Detection", at.button[4].label)
        at.button[4].click().run()
        self.assertEqual(len(at.exception), 0)
        # Verify detection results are produced
        self.assertIn("detection_result", at.session_state)
        self.assertIsNotNone(at.session_state["detection_result"])

        print("PASS: Test 7 - Streamlit workspace upload & real detection verified.")

    def test_08_replace_and_remove_image(self):
        at = AppTest.from_file("../app.py").run()
        at.button[1].click().run()

        # 1. Upload first image
        at.file_uploader[0].upload("first_image.png", self.png_data, "image/png").run()
        self.assertEqual(len(at.exception), 0)

        # 2. Replace with second image
        at.file_uploader[0].upload("second_image.webp", self.webp_data, "image/webp").run()
        self.assertEqual(len(at.exception), 0)

        # 3. Click 'Remove Image' button (Button 5)
        self.assertIn("Remove Image", at.button[5].label)
        at.button[5].click().run()
        self.assertEqual(len(at.exception), 0)
        print("PASS: Test 8 - Replace and remove image actions verified.")

    def test_09_confidence_threshold_slider(self):
        at = AppTest.from_file("../app.py").run()
        at.button[1].click().run()

        # Check default threshold is 40%
        self.assertEqual(at.session_state["confidence_threshold"], 40)

        # Upload image
        at.file_uploader[0].upload("recon.jpg", self.jpg_data, "image/jpeg").run()

        # Change slider to 75%
        at.slider[0].set_value(75).run()
        self.assertEqual(at.session_state["confidence_threshold"], 75)
        self.assertEqual(len(at.exception), 0)
        print("PASS: Test 9 - Confidence slider modified to 75% and persisted across reruns.")


if __name__ == "__main__":
    unittest.main()
