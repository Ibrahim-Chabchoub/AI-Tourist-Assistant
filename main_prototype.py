import cv2 as cv
import torch
import time
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests
import numpy as np
from io import BytesIO
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
monuments = [
    "Eiffel Tower",
    "Big Ben",
    "Colosseum",
    "Brandenburg Gate",
    "Statue of Liberty"
]
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
headers = {
    "User-Agent": "IbrahimTouristApp/1.0"
}
def preprocess_frame(frame):
    # 1. resize (helps consistency)
    frame = cv.resize(frame, (640, 480))

    # 2. slight blur to reduce noise
    frame = cv.GaussianBlur(frame, (5, 5), 0)

    # 3. improve contrast (CLAHE = very useful for AI)
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv.merge((l, a, b))
    frame = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

    return frame

def predict_monument(frame):
    image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    inputs = processor(
        text=monuments,
        images=image,
        return_tensors="pt",
        padding=True
    )

    outputs = model(**inputs)
    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1)

    best_index = torch.argmax(probs).item()
    confidence = probs[0][best_index].item()

    return monuments[best_index], confidence


def search_monument(monument):
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={monument}&format=json"
    response = requests.get(url, headers=headers)
    data = response.json()
    results = data["query"]["search"]
    if not results:
        return None
    return results[0]["title"]


def get_current_image(title):
    url = f"https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=pageimages&format=json&pithumbsize=1000"
    response = requests.get(url, headers=headers)
    data = response.json()
    pages = data["query"]["pages"]
    page = list(pages.values())[0]
    if "thumbnail" not in page:
        return None
    return page["thumbnail"]["source"]


def get_historical_image(title):

    url = f"https://commons.wikimedia.org/w/api.php?action=query&list=search&srsearch={title} historical&srnamespace=6&format=json"

    response = requests.get(url, headers=headers)
    data = response.json()

    results = data.get("query", {}).get("search", [])

    if not results:
        return None

    for r in results:

        file_title = r["title"]

        # get information about this file
        url2 = f"https://commons.wikimedia.org/w/api.php?action=query&titles={file_title}&prop=imageinfo&iiprop=url|mime&format=json"

        response2 = requests.get(url2, headers=headers)
        data2 = response2.json()

        pages = data2["query"]["pages"]
        page = list(pages.values())[0]

        if "imageinfo" not in page:
            continue

        image_info = page["imageinfo"][0]

        mime = image_info.get("mime", "")

        # accept only real images
        if mime.startswith("image/"):

            return image_info["url"]

    return None

def display_images(current_url, historical_url):

    if not current_url:
        print("No current image.")
        return

    try:

        response1 = requests.get(current_url, headers=headers)

        print("Current image:")
        print(response1.status_code)
        print(response1.headers.get("Content-Type"))

        img1 = Image.open(BytesIO(response1.content)).convert("RGB")


        if historical_url:

            response2 = requests.get(historical_url, headers=headers)

            print("Historical image:")
            print(response2.status_code)
            print(response2.headers.get("Content-Type"))

            img2 = Image.open(BytesIO(response2.content)).convert("RGB")

        else:
            print("No historical image found.")
            return


        img1 = cv.cvtColor(np.asarray(img1), cv.COLOR_RGB2BGR)
        img2 = cv.cvtColor(np.asarray(img2), cv.COLOR_RGB2BGR)


        img1 = cv.resize(img1, (500,400))
        img2 = cv.resize(img2, (500,400))


        combined = np.hstack((img1,img2))


        cv.imshow("Current vs Historical", combined)
        cv.waitKey(0)
        cv.destroyAllWindows()


    except Exception as e:
        print("Image display error:")
        print(e)

def get_description(title):
    url = f"https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=extracts&exintro=1&explaintext=1&format=json"
    response = requests.get(url, headers=headers)
    data = response.json()
    pages = data["query"]["pages"]
    page = list(pages.values())[0]
    return page.get("extract", "No description found")


def main():
    cap = cv.VideoCapture(0)

    last_name = None
    stable_count = 0
    last_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv.imshow("Camera", frame)

        # press q to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # run AI every 1 second
        if time.time() - last_time < 1:
            continue

        last_time = time.time()

        frame_processed = preprocess_frame(frame)
        name, confidence = predict_monument(frame_processed)
        print(f"Detected: {name} ({confidence:.2f})")

        # stability check
        if name == last_name:
            stable_count += 1
        else:
            stable_count = 0
            last_name = name

        # only accept stable predictions
        if stable_count >= 2 and confidence > 0.7:
            print("\nFINAL DETECTION:", name)

            title = search_monument(name)

            if title:
                current_img = get_current_image(title)
                historical_img = get_historical_image(title)
                description = get_description(title)
                print("Wikipedia title:", title)
                print("Current image URL:", current_img)
                print("Historical image URL:", historical_img)
                display_images(current_img, historical_img)

                print("\n--- DESCRIPTION ---\n")
                print(description[:1000])
                break
            stable_count = 0

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
