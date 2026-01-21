from locust import HttpUser, task, between
from io import BytesIO
from PIL import Image

class EmotionAPIUser(HttpUser):
    wait_time = between(1, 4)

    # Pre-create a dummy image to send during the test
    def on_start(self):
        image = Image.new("RGB", (224, 224), color="blue")
        self.img_byte_arr = BytesIO()
        image.save(self.img_byte_arr, format='JPEG')

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(5)
    def predict_emotion(self):
        self.img_byte_arr.seek(0) # Reset pointer to start of file
        self.client.post(
            "/predict",
            files={"file": ("load_test.jpg", self.img_byte_arr, "image/jpeg")}
        )
