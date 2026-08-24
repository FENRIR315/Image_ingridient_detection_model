# Image Ingredient Detection Model

A YOLO-based ingredient detection API. Upload a photo of food and get back the detected ingredients as JSON — built to power recipe-recommendation apps.

Fine-tuned from **YOLO26s** on a curated Roboflow dataset: **64 ingredient classes**, 39k+ training images.

## Test-split performance (3,031 unseen images)

| Precision | Recall | mAP50 | mAP50-95 |
|-----------|--------|-------|----------|
| 0.849     | 0.882  | **0.892** | 0.723 |

Inference speed: ~10–70 ms per image on an RTX 4050 GPU (CPU also works, just slower).

## Repo structure

```
├── api/
│   └── main.py        # FastAPI server
├── models/
│   └── best.pt        # Trained weights
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.9+ (tested on 3.14).

```bash
pip install -r requirements.txt
```

For GPU inference, install a CUDA build of PyTorch first (see https://pytorch.org for the right command for your CUDA version). CPU-only works out of the box.

## Run the API

```bash
cd api
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Interactive docs open at http://localhost:8000/docs

## Endpoints

### `POST /detect` — JSON detections

```bash
curl -F "file=@photo.jpg" http://localhost:8000/detect
```

```json
{
  "count": 3,
  "inference_ms": 72,
  "ingredients": [
    {"name": "tomato", "confidence": 0.937},
    {"name": "egg", "confidence": 0.865},
    {"name": "bell pepper", "confidence": 0.813}
  ],
  "detections": [
    {"name": "tomato", "confidence": 0.937, "box": [423, 0, 640, 187]}
  ]
}
```

- `ingredients` — deduplicated list (best confidence per ingredient)
- `detections` — every box with pixel coordinates `[x1, y1, x2, y2]`
- Optional query param: `?conf=0.5` to change the confidence threshold (default 0.30)

### `POST /detect/annotated` — image with boxes drawn

```bash
curl -F "file=@photo.jpg" http://localhost:8000/detect/annotated -o annotated.jpg
```

Returns the input photo as JPEG with labeled bounding boxes rendered on it.

### `GET /health` — status check

## Using from React Native

```js
import * as ImagePicker from "expo-image-picker";

async function detectIngredients(photoUri) {
  const form = new FormData();
  form.append("file", { uri: photoUri, name: "photo.jpg", type: "image/jpeg" });
  const res = await fetch("http://<YOUR-SERVER-IP>:8000/detect", {
    method: "POST",
    body: form,
  });
  const data = await res.json();
  return data.ingredients;
}
```

Notes:
- On a physical device over Wi-Fi, use your PC's LAN IP (e.g. `192.168.x.x`)
- On an Android emulator use `10.0.2.2` instead of `localhost`
- The server sends permissive CORS headers, so web apps work too

## Supported ingredients (64)

**Fruits:** apple, banana, blueberry, cherry, cranberry, cucumber, gooseberry, grape, guava, kumquat, lemon, mango, mulberry, orange, papaya, pear, pineapple, pitaya, pumpkin, raspberry, strawberry

**Vegetables:** artichoke, beet, bell pepper, broccoli, cabbage, carrot, cauliflower, eggplant, garlic, ginger, green chilli pepper, leek, lettuce, okra, onion, potato, radish, tomato, white button mushroom, zucchini

**Proteins:** anchovy, chicken, chickpeas, egg, fish, mince, mutton, pork, red meat, shrimp, tofu, white beans

**Herbs, spices & pantry:** bay leaf, coriander, oil, parsley, rice, salt, turmeric

**Other:** annona (custard apple), palm fruit

## Training

Fine-tuned from pretrained `yolo26s.pt` at 640px, 50 epochs, batch 16, on an NVIDIA RTX 4050 Laptop GPU (~15 hours). Dataset was curated from a public Roboflow ingredient-detection dataset: merged duplicate/Spanish-labeled classes and dropped junk categories down to the final 64.
