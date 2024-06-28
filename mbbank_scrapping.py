import requests
import base64
import os
from PIL import Image
from io import BytesIO
from anticaptchaofficial.imagecaptcha import imagecaptcha

# URL and body for the POST request
url = "https://ebank.mbbank.com.vn/corp/common/generateCaptcha"
body = {
    "refNo": "2024062400251990", 
    "deviceId": "737a264c94e159fb79858bf979db3ad2"
}

# Create the directory to save images if it doesn't exist
save_directory = "scrapped_images"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Initialize the captcha solver
solver = imagecaptcha()
solver.set_verbose(1)
solver.set_key("f3a44e66302c61ffec07c80f4732baf3")

# Generate 5 images and solve the captchas
for i in range(10):
    response = requests.post(url, json=body)

    # Check if the request was successful
    if response.status_code == 200:
        # Decode the Base64 encoded image from the response
        image_base64 = response.json().get("imageBase64")
        image_data = base64.b64decode(image_base64)
        
        # Create an image from the decoded bytes
        image = Image.open(BytesIO(image_data))
        
        # Save the image temporarily to solve the captcha
        temp_image_path = os.path.join(save_directory, "temp_image2.png")
        image.save(temp_image_path)
        
        # Solve the captcha
        captcha_text = solver.solve_and_return_solution(temp_image_path)
        if captcha_text != 0:
            # Save the image with the captcha text as the filename
            final_image_path = os.path.join(save_directory, f"{captcha_text}.png")
            image.save(final_image_path)
            print(f"Image {i+1} saved as {final_image_path} with text '{captcha_text}'")
        else:
            print(f"Failed to solve captcha for image {i+1}. Error: {solver.error_code}")
    else:
        print(f"Failed to retrieve captcha image. Status code: {response.status_code}")
